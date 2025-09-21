import json
import traceback
from typing import Any, Dict, Iterable, List, Optional, Union

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse

from illm.logs.logger import Logger
from illm.model.args import config
from illm.model.core import (
    BadRequest,
    CapabilityNotSupported,
    ProviderError,
    provider_registry,
)
from illm.model.providers import register_default_providers


# 注册默认 Provider 工厂
register_default_providers(provider_registry)

# FastAPI 应用
app = FastAPI(title="ILLM API", version="1.0")


# ========== 工具函数 ==========

def _merge_config_with_request(body: Dict[str, Any]) -> Dict[str, Any]:
    """优先级：请求体 > 全局配置"""
    merged = dict(config)
    for key in [
        "framework",
        "model",
        "device",
        "dtype",
        "tgi_base_url",
        "vllm_base_url",
        "sglang_base_url",
        "ollama_base_url",
        "max_tokens",
        "temperature",
        "top_p",
        "top_k",
        "repetition_penalty",
        "stop",
        "embedding_model",
    ]:
        if key in body and body.get(key) is not None:
            merged[key] = body[key]
    # 兼容 transformer/transformers
    fw = (merged.get("framework") or "").lower()
    if fw == "transformer":
        merged["framework"] = "transformers"
    return merged


def _select_provider(body: Dict[str, Any]):
    merged = _merge_config_with_request(body)
    framework = merged.get("framework", "transformers")
    Logger.debug(f"Selecting provider: framework={framework}, model={merged.get('model')}")
    return provider_registry.create(framework, merged)


def _extract_gen_params(body: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "max_tokens",
        "temperature",
        "top_p",
        "top_k",
        "repetition_penalty",
        "stop",
        "stream_options",
        "request_id",
    ]
    return {k: body[k] for k in keys if k in body}


def _messages_from_prompt(prompt: str) -> List[Dict[str, Any]]:
    return [{"role": "user", "content": prompt}]


# ========== 全局异常处理 ==========

@app.exception_handler(ProviderError)
async def provider_error_handler(request: Request, exc: ProviderError):
    status = 422 if isinstance(exc, CapabilityNotSupported) else 400
    Logger.error(f"ProviderError: {exc}")
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "type": exc.__class__.__name__,
                "message": str(exc),
            }
        },
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception):
    Logger.error(f"Unhandled error: {exc}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "type": "InternalServerError",
                "message": str(exc),
            }
        },
    )


# ========== 健康与模型 ==========

@app.get("/healthz")
async def healthz():
    return {
        "status": "ok",
        "framework": config.get("framework"),
        "model": config.get("model"),
        "providers": provider_registry.available(),
    }


@app.get("/models")
async def list_models(framework: Optional[str] = Query(default=None)):
    body: Dict[str, Any] = {}
    if framework:
        body["framework"] = framework
    try:
        provider = _select_provider(body)
        return {
            "object": "list",
            "data": provider.list_models(),
        }
    except Exception as e:
        Logger.warning(f"List models failed: {e}")
        # 返回空列表但不报 500，便于显示可用性
        return {"object": "list", "data": []}


# ========== OpenAI 兼容端点 ==========

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    provider = _select_provider(body)
    model = body.get("model") or config.get("model")
    stream = bool(body.get("stream", False))
    messages = body.get("messages")
    if not messages:
        prompt = body.get("prompt")
        if not prompt:
            raise HTTPException(status_code=400, detail="messages or prompt is required")
        messages = _messages_from_prompt(prompt)

    gen_params = _extract_gen_params(body)
    Logger.info(f"/v1/chat/completions stream={stream}, model={model}")

    result = provider.chat_completions(messages=messages, model=model, stream=stream, gen_params=gen_params)

    if not stream:
        return result

    def event_stream() -> Iterable[str]:
        try:
            for chunk in result:  # type: ignore[assignment]
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        except ProviderError as e:
            err = {"error": {"type": e.__class__.__name__, "message": str(e)}}
            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
        finally:
            # 结束标记
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/v1/completions")
async def completions(request: Request):
    body = await request.json()
    # 兼容旧版：使用 prompt 转换为 messages
    prompt = body.get("prompt")
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required")
    body.setdefault("messages", _messages_from_prompt(prompt))

    provider = _select_provider(body)
    model = body.get("model") or config.get("model")
    stream = bool(body.get("stream", False))
    gen_params = _extract_gen_params(body)
    Logger.info(f"/v1/completions stream={stream}, model={model}")

    result = provider.chat_completions(messages=body["messages"], model=model, stream=stream, gen_params=gen_params)

    if not stream:
        return result

    def event_stream() -> Iterable[str]:
        try:
            for chunk in result:  # type: ignore[assignment]
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        except ProviderError as e:
            err = {"error": {"type": e.__class__.__name__, "message": str(e)}}
            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/v1/embeddings")
async def create_embeddings(request: Request):
    body = await request.json()
    provider = _select_provider(body)
    model = body.get("model") or config.get("embedding_model") or config.get("model")
    inputs = body.get("input") if "input" in body else body.get("inputs")
    if inputs is None:
        raise HTTPException(status_code=400, detail="input (string or array) is required")

    Logger.info(f"/v1/embeddings model={model}")
    resp = provider.embeddings(inputs=inputs, model=model, emb_params=None)
    return resp


@app.post("/v1/audio/transcriptions")
async def audio_transcriptions(file: UploadFile = File(default=None), request: Request = None):
    """
    可选：支持上传文件或在 JSON 中提供 audio_base64。
    当后端不支持时返回 422。
    """
    body: Dict[str, Any] = {}
    if request is not None:
        try:
            body = await request.json()
        except Exception:
            body = {}
    provider = _select_provider(body)
    model = body.get("model") or config.get("model")

    audio_bytes: Optional[bytes] = None
    if file is not None:
        audio_bytes = await file.read()
    elif "audio_base64" in body:
        import base64 as _b64
        audio_bytes = _b64.b64decode(body.get("audio_base64") or "")

    if not audio_bytes:
        raise HTTPException(status_code=400, detail="audio file or audio_base64 is required")

    Logger.info(f"/v1/audio/transcriptions model={model}")
    try:
        return provider.audio_transcriptions(audio_bytes=audio_bytes, model=model, stt_params=None)
    except CapabilityNotSupported as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


# ========== 启动函数（由 server 入口调用） ==========

def start_uvicorn(host: str, port: int):
    try:
        Logger.info(f"Starting server at {host}:{port}")
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=False,
            access_log=True,
            log_level="debug",
        )
    except Exception as e:
        Logger.critical(f"Critical error starting Uvicorn server: {str(e)}")
        raise