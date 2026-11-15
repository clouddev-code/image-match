"""CLIP の画像埋め込みで、アップロードされた画像どうしの類似度を返す。

アップロードされた画像はメモリ上で処理するだけで、保存しない。
"""

import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

import open_clip
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

MODEL_NAME = os.environ.get("CLIP_MODEL", "ViT-B-32")
PRETRAINED = os.environ.get("CLIP_PRETRAINED", "laion2b_s34b_b79k")
GALLERY_DIR = Path(os.environ.get("GALLERY_DIR", Path(__file__).parent.parent / "gallery"))
STATIC_DIR = Path(__file__).parent / "static"
MAX_BYTES = 10 * 1024 * 1024
MAX_CANDIDATES = 20
TOP_K = 5
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

state: dict = {}


def embed(images: list[Image.Image]) -> torch.Tensor:
    batch = torch.stack([state["preprocess"](img) for img in images])
    with torch.no_grad():
        features = state["model"].encode_image(batch)
    return features / features.norm(dim=-1, keepdim=True)


def load_upload(file: UploadFile) -> Image.Image:
    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"{file.filename}: 10MB 以下の画像にしてください")
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, f"{file.filename}: 画像として読めませんでした")


@asynccontextmanager
async def lifespan(app: FastAPI):
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "2")))
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    model.eval()
    state["model"], state["preprocess"] = model, preprocess

    paths = sorted(p for p in GALLERY_DIR.glob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    state["gallery_names"] = [p.name for p in paths]
    if paths:
        state["gallery_embs"] = embed([Image.open(p).convert("RGB") for p in paths])
    yield
    state.clear()


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
if GALLERY_DIR.is_dir():
    app.mount("/gallery", StaticFiles(directory=GALLERY_DIR), name="gallery")


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
    embs = embed([load_upload(query)] + [load_upload(c) for c in candidates])
    scores = (embs[:1] @ embs[1:].T)[0].tolist()
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return {"results": [{"index": i, "name": candidates[i].filename, "score": round(s, 4)} for i, s in ranked]}
