FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.2 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    HF_HOME=/models \
    PATH=/app/.venv/bin:$PATH

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# 後から chown するとモデルの層が重複するので、先にユーザーを作ってそのユーザーで取り込む
RUN useradd --uid 1000 --no-create-home app && mkdir /models && chown app /models
USER 1000

# モデルの重みをイメージに入れ、起動時にダウンロードしない
RUN python -c "import open_clip; open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')"

COPY app ./app
COPY gallery ./gallery

ENV HF_HUB_OFFLINE=1 \
    TORCH_THREADS=2

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
