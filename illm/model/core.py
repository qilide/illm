from __future__ import annotations

import base64
import io
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple, Union, Callable


# ============== 基础异常与错误类型 ==============

class ProviderError(Exception):
    """通用提供方异常"""


class CapabilityNotSupported(ProviderError):
    """请求的能力不被该后端支持"""


class RemoteEndpointError(ProviderError):
    """远端服务错误"""


class ModelNotFound(ProviderError):
    """模型不存在或不可用"""


class BadRequest(ProviderError):
    """请求参数错误"""


# ============== 能力声明与消息结构 ==============

@dataclass(frozen=True)
class Capabilities:
    supports_text_generation: bool = False
    supports_streaming: bool = False
    supports_embeddings: bool = False
    supports_vision: bool = False
    supports_audio_stt: bool = False


PartType = Dict[str, Any]  # {"type": "text"|"image_url"|"image_base64", ...}
Message = Dict[str, Any]   # {"role": "user"|"assistant"|"system", "content": str|List[PartType]}

# ============== Provider 抽象接口 ==============

class Provider(ABC):
    """统一 Provider 抽象"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def capabilities(self) -> Capabilities:
        ...

    # 文本对话（OpenAI 风格）
    @abstractmethod
    def chat_completions(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        stream: bool = False,
        gen_params: Optional[Dict[str, Any]] = None,
    ) -> Union[Dict[str, Any], Iterable[Dict[str, Any]]]:
        """
        返回：
          - 非流式：OpenAI chat.completion 结构字典
          - 流式：迭代产生 chat.completion.chunk 结构字典
        """
        ...

    # 旧版补全（可选）
    def completions(
        self,
        prompt: str,
        model: Optional[str] = None,
        stream: bool = False,
        gen_params: Optional[Dict[str, Any]] = None,
    ) -> Union[Dict[str, Any], Iterable[Dict[str, Any]]]:
        raise CapabilityNotSupported(f"{self.name} does not support legacy completions")

    # 向量嵌入
    def embeddings(
        self,
        inputs: Union[str, List[str]],
        model: Optional[str] = None,
        emb_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        raise CapabilityNotSupported(f"{self.name} does not support embeddings")

    # 语音转文字
    def audio_transcriptions(
        self,
        audio_bytes: bytes,
        model: Optional[str] = None,
        stt_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        raise CapabilityNotSupported(f"{self.name} does not support audio STT")

    # 模型列表
    def list_models(self) -> List[Dict[str, Any]]:
        return [{"id": self.config.get("model", "unknown"), "object": "model"}]


# ============== Provider 注册表 ==============

class ProviderRegistry:
    """根据配置或请求选择合适 Provider 实例"""

    def __init__(self):
        self._providers: Dict[str, Callable[[Dict[str, Any]], Provider]] = {}

    def register(self, key: str, factory: Callable[[Dict[str, Any]], Provider]) -> None:
        self._providers[key.lower()] = factory

    def has(self, key: str) -> bool:
        return key.lower() in self._providers

    def available(self) -> List[str]:
        return list(self._providers.keys())

    def create(self, framework: str, config: Dict[str, Any]) -> Provider:
        key = (framework or "").lower()
        if key not in self._providers:
            raise BadRequest(f"Unknown framework '{framework}'. Available: {', '.join(self.available())}")
        return self._providers[key](config)


# ============== 多模态辅助 ==============

def extract_text_from_messages(messages: List[Message], allow_vision: bool) -> Tuple[str, List[io.BytesIO]]:
    """
    将 OpenAI 风格 messages 解析为纯文本提示，同时在允许视觉时解析图像为 BytesIO 列表
    """
    text_parts: List[str] = []
    images: List[io.BytesIO] = []

    for msg in messages:
        content = msg.get("content", "")
        role = msg.get("role", "user")
        prefix = f"{role}: "

        if isinstance(content, str):
            text_parts.append(prefix + content)
        elif isinstance(content, list):
            for part in content:
                ptype = part.get("type")
                if ptype == "text":
                    text_parts.append(prefix + str(part.get("text", "")))
                elif ptype in ("image_url", "image_base64"):
                    if not allow_vision:
                        raise CapabilityNotSupported("Vision content not supported by selected backend")
                    img_bytes = _load_image_bytes(part, ptype)
                    if img_bytes:
                        images.append(img_bytes)
                else:
                    # 忽略未知类型
                    continue
        else:
            # 非法 content，忽略
            continue

    return "\n".join(text_parts), images


def _load_image_bytes(part: Dict[str, Any], ptype: str) -> Optional[io.BytesIO]:
    """
    安全加载图像，返回 BytesIO（不做解码入内存大图像，留给后端自行处理）
    """
    try:
        if ptype == "image_url":
            import requests
            url = part.get("image_url") or part.get("url")
            if not url:
                return None
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            # 限制下载大小（例如 8MB）
            if len(resp.content) > 8 * 1024 * 1024:
                raise BadRequest("Image too large (> 8MB)")
            return io.BytesIO(resp.content)
        else:
            b64 = part.get("image_base64") or part.get("b64_json")
            if not b64:
                return None
            raw = base64.b64decode(b64)
            if len(raw) > 8 * 1024 * 1024:
                raise BadRequest("Image too large (> 8MB)")
            return io.BytesIO(raw)
    except Exception as e:
        raise BadRequest(f"Failed to load image: {e}") from e


# ============== OpenAI 风格响应构造 ==============

def now_ts() -> int:
    return int(time.time())


def build_chat_completion(
    model: str,
    content: str,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "id": request_id or f"chatcmpl-{int(time.time() * 1000)}",
        "object": "chat.completion",
        "created": now_ts(),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


def build_chat_chunk_delta(content_part: Optional[str]) -> Dict[str, Any]:
    return {
        "role": "assistant",
        "content": content_part,
    }


def build_chat_chunk(model: str, delta_content: Optional[str], request_id: Optional[str] = None) -> Dict[str, Any]:
    return {
        "id": request_id or f"chatcmpl-{int(time.time() * 1000)}",
        "object": "chat.completion.chunk",
        "created": now_ts(),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": build_chat_chunk_delta(delta_content),
                "finish_reason": None,
            }
        ],
    }


def build_embeddings_response(
    model: str,
    vectors: List[List[float]],
) -> Dict[str, Any]:
    return {
        "object": "list",
        "model": model,
        "model_replica": "default",
        "data": [
            {"index": i, "object": "embedding", "embedding": vec}
            for i, vec in enumerate(vectors)
        ],
        "usage": {
            "prompt_tokens": 0,
            "total_tokens": 0,
        },
    }


# ============== 全局默认注册表实例（供 API 层使用） ==============

provider_registry = ProviderRegistry()