"""Real-time inference gateway — СКЕЛЕТ.

Готовы /health и /metrics. Остальные ручки реализуйте сами по контракту
(API_CONTRACT.md в корне этого каталога):
    /triton/health, /vllm/health, /metrics/triton, /metrics/vllm, /metrics/node,
    /info (Bearer, с полем gpu), /vision (Bearer, модель вашего варианта — make variant),
    /chat (Bearer, SSE).

Как структурировать код (auth, конфиг, инференс) — на ваше усмотрение.
"""
from __future__ import annotations
import io
import os
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import jwt
import numpy as np
import tritonclient.http as httpclient
from fastapi import (
    FastAPI,
    Header,
    HTTPException,
    Depends,

    Response,
    UploadFile,
    File,
    Request,)
from fastapi.responses import StreamingResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from prometheus_fastapi_instrumentator import Instrumentator
from starlette.concurrency import run_in_threadpool
JWT_SECRET = os.getenv("JWT_SECRET", "")
STUDENT_NAME = os.getenv("STUDENT_NAME", "")
STUDENT_EMAIL = os.getenv("STUDENT_EMAIL", "")
CV_MODEL = os.getenv("CV_MODEL", "")
LLM_MODEL = os.getenv("LLM_MODEL", "")
GPU_NAME = os.getenv("GPU_NAME", "")

TRITON_URL = os.getenv("TRITON_URL", "http://triton:8000")
VLLM_URL = os.getenv("VLLM_URL", "http://vllm:8000")
TRITON_METRICS_URL = os.getenv(
    "TRITON_METRICS_URL", "http://triton:8002/metrics"
)
VLLM_METRICS_URL = os.getenv(
    "VLLM_METRICS_URL", "http://vllm:8000/metrics"
)
NODE_METRICS_URL = os.getenv(
    "NODE_METRICS_URL", "http://node-exporter:9100/metrics"
)
def require_auth(authorization: str | None = Header(default=None)) -> dict:
    """Проверяет Bearer JWT и возвращает его payload."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")

    if not JWT_SECRET:
        raise HTTPException(status_code=500, detail="JWT_SECRET is not configured")

    token = authorization.removeprefix("Bearer ").strip()

    try:
        payload = jwt.decode(
            token,
            JWT_SECRET,
            algorithms=["HS256"],
        )
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    return payload
app = FastAPI(title="Real-time Inference Gateway")

Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
@app.get("/info")
def info(_: dict = Depends(require_auth)) -> dict:
    if not GPU_NAME:
        raise HTTPException(status_code=503, detail="GPU_NAME is not configured")

    return {
        "student": {
            "name": STUDENT_NAME,
            "email": STUDENT_EMAIL,
        },
        "models": {
            "vision": CV_MODEL,
            "llm": LLM_MODEL,
        },
        "gpu": GPU_NAME,
        "version": "1.0",
    }

async def fetch_metrics(url: str) -> str:
    """Получает исходные Prometheus-метрики сервиса."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.text
    except httpx.HTTPError:
        raise HTTPException(
            status_code=502,
            detail="Metrics backend is unavailable",
        )


@app.get("/triton/health")
async def triton_health() -> dict:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(
                f"{TRITON_URL}/v2/health/ready"
            )
            response.raise_for_status()

        return {
            "backend": "triton",
            "ready": True,
            "model": CV_MODEL,
        }
    except httpx.HTTPError:
        raise HTTPException(
            status_code=503,
            detail="Triton is unavailable",
        )


@app.get("/vllm/health")
async def vllm_health() -> dict:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{VLLM_URL}/v1/models")
            response.raise_for_status()
            data = response.json()

        models = [
            model["id"]
            for model in data.get("data", [])
            if "id" in model
        ]

        return {
            "backend": "vllm",
            "ready": True,
            "models": models,
        }
    except (httpx.HTTPError, ValueError, AttributeError):
        raise HTTPException(
            status_code=503,
            detail="vLLM is unavailable",
        )


@app.get("/metrics/triton")
async def triton_metrics():
    return Response(
        content=await fetch_metrics(TRITON_METRICS_URL),
        media_type="text/plain",
    )

# Параметры preprocessing из MobileNet_V3_Large_Weights.IMAGENET1K_V2.
IMAGE_SIZE = 224
RESIZE_SIZE = 232

IMAGENET_MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32,
).reshape(1, 1, 3)

IMAGENET_STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32,
).reshape(1, 1, 3)

IMAGENET_CLASSES_PATH = Path(__file__).parent / "imagenet_classes.txt"
IMAGENET_CLASSES = IMAGENET_CLASSES_PATH.read_text(
    encoding="utf-8"
).splitlines()


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Преобразует изображение в формат MobileNet V3 Large.

    Возвращает массив FP32 формы [1, 3, 224, 224].
    """
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image = ImageOps.exif_transpose(image)

            # Обрабатываем прозрачность: накладываем изображение на белый фон.
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, rgba).convert("RGB")

            width, height = image.size

            if width <= 0 or height <= 0:
                raise ValueError("Invalid image dimensions")

            # Resize: меньшая сторона становится равна 232.
            if width <= height:
                new_width = RESIZE_SIZE
                new_height = int(RESIZE_SIZE * height / width)
            else:
                new_height = RESIZE_SIZE
                new_width = int(RESIZE_SIZE * width / height)

            image = image.resize(
                (new_width, new_height),
                resample=Image.Resampling.BILINEAR,
            )

            # CenterCrop: центральная область 224 x 224.
            left = int(round((new_width - IMAGE_SIZE) / 2.0))
            top = int(round((new_height - IMAGE_SIZE) / 2.0))

            image = image.crop(
                (
                    left,
                    top,
                    left + IMAGE_SIZE,
                    top + IMAGE_SIZE,
                )
            )

            # PIL Image -> NumPy [224, 224, 3], диапазон [0, 1].
            array = np.asarray(image, dtype=np.float32) / 255.0

    except (
        UnidentifiedImageError,
        Image.DecompressionBombError,
        OSError,
        ValueError,
    ) as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file",
        ) from exc

    # Нормализация ImageNet.
    array = (array - IMAGENET_MEAN) / IMAGENET_STD

    # HWC -> CHW и добавление batch dimension.
    array = np.transpose(array, (2, 0, 1))
    array = np.expand_dims(array, axis=0)

    return np.ascontiguousarray(array, dtype=np.float32)


def run_triton(tensor: np.ndarray) -> np.ndarray:
    """
    Отправляет изображение в Triton и возвращает логиты MobileNet.
    """
    if not CV_MODEL:
        raise RuntimeError("CV_MODEL is not configured")

    parsed_url = urlsplit(TRITON_URL)
    triton_address = parsed_url.netloc or parsed_url.path

    client = httpclient.InferenceServerClient(
        url=triton_address,
        connection_timeout=5.0,
        network_timeout=10.0,
    )

    try:
        infer_input = httpclient.InferInput(
            "INPUT__0",
            tensor.shape,
            "FP32",
        )
        infer_input.set_data_from_numpy(tensor)

        infer_output = httpclient.InferRequestedOutput("OUTPUT__0")

        result = client.infer(
            model_name=CV_MODEL,
            inputs=[infer_input],
            outputs=[infer_output],
        )

        logits = result.as_numpy("OUTPUT__0")

        if logits is None or logits.shape != (1, 1000):
            raise RuntimeError("Unexpected Triton output shape")

        return logits[0].astype(np.float32)

    finally:
        client.close()


def get_top5(logits: np.ndarray) -> list[dict]:
    """
    Преобразует логиты в вероятности и возвращает пять лучших классов.
    """
    # Численно устойчивый Softmax.
    shifted = logits - np.max(logits)
    exp_values = np.exp(shifted)
    probabilities = exp_values / np.sum(exp_values)

    top_indices = np.argsort(probabilities)[-5:][::-1]

    predictions = []

    for class_id in top_indices:
        predictions.append(
            {
                "class_id": int(class_id),
                "label": IMAGENET_CLASSES[int(class_id)],
                "score": float(probabilities[class_id]),
            }
        )

    return predictions


@app.post("/vision")
async def vision(
    file: UploadFile = File(...),
    _: dict = Depends(require_auth),
) -> dict:
    """
    Классифицирует изображение с помощью MobileNet V3 Large.
    """
    started_at = time.perf_counter()

    image_bytes = await file.read()
    await file.close()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Empty image file",
        )

    tensor = preprocess_image(image_bytes)

    try:
        logits = await run_in_threadpool(run_triton, tensor)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="Triton inference failed",
        ) from exc

    predictions = get_top5(logits)

    latency_ms = (time.perf_counter() - started_at) * 1000

    return {
        "model": CV_MODEL,
        "predictions": predictions,
        "latency_ms": round(latency_ms, 2),
    }
@app.get("/metrics/vllm")
async def vllm_metrics():
    return Response(
        content=await fetch_metrics(VLLM_METRICS_URL),
        media_type="text/plain",
    )


@app.get("/metrics/node")
async def node_metrics():
    return Response(
        content=await fetch_metrics(NODE_METRICS_URL),
        media_type="text/plain",
    )
@app.post("/chat")
async def chat(
    request: Request,
    _: dict = Depends(require_auth),
):
    """
    Отправляет запрос в vLLM.
    Поддерживает SSE streaming и обычный JSON-ответ.
    """
    try:
        body = await request.json()
    except (ValueError, UnicodeDecodeError):
        raise HTTPException(status_code=400, detail="Invalid JSON")

    if (
        not isinstance(body, dict)
        or not isinstance(body.get("messages"), list)
        or not body["messages"]
    ):
        raise HTTPException(
            status_code=400,
            detail="Field 'messages' must be a non-empty list",
        )

    payload = dict(body)
    payload.setdefault("max_tokens", 128)
    payload.setdefault("temperature", 0.0)

    stream = payload.get("stream", True)

    if not isinstance(stream, bool):
        raise HTTPException(
            status_code=400,
            detail="Field 'stream' must be a boolean",
        )

    payload["stream"] = stream
    chat_url = f"{VLLM_URL.rstrip('/')}/v1/chat/completions"

    # Обычный режим: возвращаем JSON от vLLM.
    if not stream:
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    chat_url,
                    json=payload,
                )
        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=502,
                detail="vLLM is unavailable",
            ) from exc

        if response.status_code >= 400:
            status_code = (
                400 if response.status_code < 500 else 502
            )
            raise HTTPException(
                status_code=status_code,
                detail="vLLM rejected the request",
            )

        try:
            return response.json()
        except ValueError as exc:
            raise HTTPException(
                status_code=502,
                detail="Invalid JSON response from vLLM",
            ) from exc

    # Потоковый режим.
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(
            connect=5.0,
            read=None,
            write=30.0,
            pool=5.0,
        )
    )

    try:
        upstream_request = client.build_request(
            "POST",
            chat_url,
            json=payload,
        )
        upstream = await client.send(
            upstream_request,
            stream=True,
        )
    except httpx.HTTPError as exc:
        await client.aclose()
        raise HTTPException(
            status_code=502,
            detail="vLLM is unavailable",
        ) from exc

    # Проверяем статус до начала SSE-ответа клиенту.
    if upstream.status_code >= 400:
        status_code = (
            400 if upstream.status_code < 500 else 502
        )
        await upstream.aclose()
        await client.aclose()

        raise HTTPException(
            status_code=status_code,
            detail="vLLM rejected the request",
        )

    async def event_stream():
        done_received = False
        stream_interrupted = False

        try:
            async for line in upstream.aiter_lines():
                if not line:
                    yield b"\n"
                    continue

                yield (line + "\n").encode("utf-8")

                if line.strip() == "data: [DONE]":
                    done_received = True

        except httpx.HTTPError:
            stream_interrupted = True
            yield (
                b'data: {"error":{"message":"vLLM stream interrupted"}}\n\n'
            )

        finally:
            await upstream.aclose()
            await client.aclose()


    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
