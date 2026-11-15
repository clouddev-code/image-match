"""CLIP の画像埋め込みで、アップロードされた画像どうしの類似度を返す。

アップロードされた画像はメモリ上で処理するだけで、保存しない。
"""

import io
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import open_clip
import torch
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

MODEL_NAME = os.environ.get("CLIP_MODEL", "ViT-B-32")
PRETRAINED = os.environ.get("CLIP_PRETRAINED", "laion2b_s34b_b79k")
GALLERY_DIR = Path(os.environ.get("GALLERY_DIR", Path(__file__).parent.parent / "gallery"))
STATIC_DIR = Path(__file__).parent / "static"
MAX_BYTES = 5 * 1024 * 1024
MAX_REQUEST_BYTES = 30 * 1024 * 1024
# 展開後の大きさで制限する。小さな PNG でも展開すると数 GB になる画像（解凍爆弾）があるため
MAX_PIXELS = 25_000_000
THUMB_SIZE = (512, 512)
MAX_CANDIDATES = 20
TOP_K = 5
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

Image.MAX_IMAGE_PIXELS = MAX_PIXELS
# 同時に展開・推論するリクエストを絞り、メモリの使用量を limits に収める
compute_slots = threading.BoundedSemaphore(2)
state: dict = {}


def to_tensor(img: Image.Image) -> torch.Tensor:
    """縮小してから前処理し、展開した画像をすぐ手放せるようにする。"""
    if img.width * img.height > MAX_PIXELS:
        raise ValueError("too many pixels")
    img.draft("RGB", THUMB_SIZE)
    img.thumbnail(THUMB_SIZE)
    return state["preprocess"](img.convert("RGB"))


def embed(tensors: list[torch.Tensor]) -> torch.Tensor:
    with torch.no_grad():
        features = state["model"].encode_image(torch.stack(tensors))
    return features / features.norm(dim=-1, keepdim=True)


def load_upload(file: UploadFile) -> torch.Tensor:
    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"{file.filename}: 5MB 以下の画像にしてください")
    try:
        with Image.open(io.BytesIO(data)) as img:
            return to_tensor(img)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, ValueError):
        raise HTTPException(400, f"{file.filename}: 画像として読めないか、大きすぎます")


@asynccontextmanager
async def lifespan(app: FastAPI):
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "2")))
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    model.eval()
    state["model"], state["preprocess"] = model, preprocess

    paths = sorted(p for p in GALLERY_DIR.glob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    state["gallery_names"] = [p.name for p in paths]
    if paths:
        tensors = []
        for p in paths:
            with Image.open(p) as img:
                tensors.append(to_tensor(img))
        state["gallery_embs"] = embed(tensors)
    yield
    state.clear()


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
if GALLERY_DIR.is_dir():
    app.mount("/gallery", StaticFiles(directory=GALLERY_DIR), name="gallery")


@app.middleware("http")
async def limit_body(request: Request, call_next):
    """アップロードは一時ファイルに書かれるので、受け取る前に全体の大きさで止める。"""
    if request.method == "POST":
        length = request.headers.get("content-length", "")
        if not length.isdigit():
            return JSONResponse({"detail": "Content-Length が必要です"}, status_code=411)
        if int(length) > MAX_REQUEST_BYTES:
            return JSONResponse({"detail": "送る画像の合計が大きすぎます"}, status_code=413)
    return await call_next(request)


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/healthz")
def healthz():
    return {"ok": "model" in state}


@app.get("/api/gallery")
def gallery():
    return {"images": state["gallery_names"]}


@app.post("/api/search")
def search(query: UploadFile = File(...)):
    """同梱のギャラリーから、query に似た画像を返す。"""
    if not state["gallery_names"]:
        raise HTTPException(404, "ギャラリーに画像がありません")
    with compute_slots:
        scores = (embed([load_upload(query)]) @ state["gallery_embs"].T)[0]
    top = scores.topk(min(TOP_K, len(scores)))
    return {
        "results": [
            {"name": state["gallery_names"][i], "score": round(s, 4)}
            for s, i in zip(top.values.tolist(), top.indices.tolist())
        ]
    }


@app.post("/api/compare")
def compare(query: UploadFile = File(...), candidates: list[UploadFile] = File(...)):
    """candidates を query に似ている順に並べて返す。"""
    if len(candidates) > MAX_CANDIDATES:
        raise HTTPException(400, f"比べる画像は {MAX_CANDIDATES} 枚までにしてください")
    with compute_slots:
        embs = embed([load_upload(query)] + [load_upload(c) for c in candidates])
    scores = (embs[:1] @ embs[1:].T)[0].tolist()
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return {"results": [{"index": i, "name": candidates[i].filename, "score": round(s, 4)} for i, s in ranked]}
