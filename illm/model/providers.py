from __future__ import annotations

import json
import math
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from .core import (
    BadRequest,
    CapabilityNotSupported,
    Capabilities,
    Message,
    Provider,
    ProviderRegistry,
    RemoteEndpointError,
    build_chat_chunk,
    build_chat_completion,
    build_embeddings_response,
    extract_text_from_messages,
)


def _get_bool(d: Dict[str, Any], key: str, default: bool = False) -> bool:
    v = d.get(key, default)
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in ("1", "true", "yes", "y")
    return bool(v)


def _get_int(d: Dict[str, Any], key: str, default: Optional[int] = None) -> Optional[int]:
    v = d.get(key, default)
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return default


# ====================== Transformers Provider ======================

class TransformersProvider(Provider):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._gen_pipe = None
        self._emb_pipe = None
        self._device = config.get("device", "cpu")
        self._gen_model = config.get("model") or "sshleifer/tiny-gpt2"
        # 使用一个极小的模型用于特征抽取，避免额外大型依赖
        self._emb_model = config.get("embedding_model") or "sshleifer/tiny-distilbert-base-cased"

    @property
    def name(self) -> str:
        return "transformers"

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            supports_text_generation=True,
            supports_streaming=True,   # 通过后处理实现伪流
            supports_embeddings=True,
            supports_vision=False,
            supports_audio_stt=False,
        )

    def _ensure_gen(self):
        if self._gen_pipe is None:
            from transformers import pipeline
            # 使用 text-generation pipeline
            self._gen_pipe = pipeline(
                "text-generation",
                model=self._gen_model,
                device=0 if self._device != "cpu" else -1,
            )

    def _ensure_emb(self):
        if self._emb_pipe is None:
            from transformers import pipeline
            # 使用 feature-extraction pipeline
            self._emb_pipe = pipeline(
                "feature-extraction",
                model=self._emb_model,
                device=0 if self._device != "cpu" else -1,
            )

    def chat_completions(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        stream: bool = False,
        gen_params: Optional[Dict[str, Any]] = None,
    ) -> Union[Dict[str, Any], Iterable[Dict[str, Any]]]:
        allow_vision = False
        prompt, _images = extract_text_from_messages(messages, allow_vision=allow_vision)
        self._ensure_gen()
        params = gen_params or {}
        max_tokens = _get_int(params, "max_tokens", self.config.get("max_tokens", 64))
        temperature = float(params.get("temperature", self.config.get("temperature", 0.7)))
        top_p = float(params.get("top_p", self.config.get("top_p", 0.95)))
        top_k = _get_int(params, "top_k", self.config.get("top_k", 50))

        # 生成一次全量文本（Transformers pipeline 原生不方便逐 token 流式）
        outputs = self._gen_pipe(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            return_full_text=False,
            num_return_sequences=1,
        )
        generated = outputs[0]["generated_text"]

        model_name = model or self._gen_model
        if not stream:
            return build_chat_completion(model=model_name, content=generated)

        # 伪流式切片输出，避免一次性返回
        def _iter_slices(text: str, chunk_size: int = 32) -> Iterable[Dict[str, Any]]:
            for i in range(0, len(text), chunk_size):
                yield build_chat_chunk(model=model_name, delta_content=text[i : i + chunk_size])

        return _iter_slices(generated)

    def embeddings(
        self,
        inputs: Union[str, List[str]],
        model: Optional[str] = None,
        emb_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        texts = [inputs] if isinstance(inputs, str) else inputs
        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            raise BadRequest("Embeddings input must be a string or a list of strings")

        self._ensure_emb()
        # 将输出强制为张量并做等长填充，避免不规则列表导致的聚合错误
        feats = self._emb_pipe(texts, return_tensors=True, truncation=True, padding=True)
        vectors: List[List[float]] = []

        # 优先走 torch/numpy 快速路径
        try:
            try:
                import torch  # type: ignore
            except Exception:
                torch = None  # type: ignore

            if torch is not None and isinstance(feats, torch.Tensor):
                # 形状通常为 [batch, tokens, hidden]
                t = feats
                if t.dim() == 3:
                    mean = t.mean(dim=1)  # 平均 token 维度
                elif t.dim() == 2:
                    mean = t
                else:
                    mean = t.reshape(t.size(0), -1)
                vectors = mean.detach().cpu().tolist()
            else:
                # 可能是 numpy 数组或 Python 列表
                try:
                    import numpy as np  # type: ignore
                    x = np.array(feats)
                    if x.ndim >= 3:
                        # [batch, tokens, hidden] -> 平均 tokens
                        mean = x.mean(axis=1)
                        vectors = mean.tolist()
                    elif x.ndim == 2:
                        vectors = x.tolist()
                    else:
                        vectors = x.reshape(x.shape[0], -1).tolist()
                except Exception:
                    raise RuntimeError("NUMPY_PATH_FAILED")
        except Exception:
            # 纯 Python 回退路径：对每个样本进行聚合
            def _avg_tokens(mat: Any) -> List[float]:
                # 支持：
                # - [tokens, hidden]
                # - 其他嵌套结构（递归提取最内层为向量的列表）
                def _collect(node, out):
                    if isinstance(node, (list, tuple)):
                        if node and all(isinstance(x, (int, float)) for x in node):
                            out.append([float(x) for x in node])
                        else:
                            for e in node:
                                _collect(e, out)

                leaf: List[List[float]] = []
                _collect(mat, leaf)
                if not leaf:
                    return []

                hidden = len(leaf[0])
                accum = [0.0] * hidden
                count = 0
                for v in leaf:
                    if len(v) != hidden:
                        continue
                    for j in range(hidden):
                        accum[j] += v[j]
                    count += 1
                denom = float(max(count, 1))
                return [val / denom for val in accum]

            # feats 可能是 batch 列表
            if isinstance(feats, list):
                for item in feats:
                    vectors.append(_avg_tokens(item))
            else:
                # 兜底：当 feats 不是可迭代列表时，直接尝试聚合
                vectors.append(_avg_tokens(feats))

        emb_model = model or self._emb_model
        return build_embeddings_response(model=emb_model, vectors=vectors)


# ====================== Ollama Provider (本地/多模态) ======================

class OllamaProvider(Provider):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._host = config.get("ollama_base_url") or "http://127.0.0.1:11434"
        self._model = config.get("model", "llama3:8b")

        # 懒加载客户端
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            try:
                import ollama
            except Exception as e:
                raise RemoteEndpointError(f"Ollama client not available: {e}") from e
            self._client = ollama.Client(host=self._host)

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            supports_text_generation=True,
            supports_streaming=True,
            supports_embeddings=False,  # 若后续需要可扩展到 embeddings
            supports_vision=True,       # 支持传入图像（取决于具体模型，如 llava）
            supports_audio_stt=False,
        )

    def list_models(self) -> List[Dict[str, Any]]:
        import requests
        base = (self._host or "http://127.0.0.1:11434").rstrip("/")
        try:
            r = requests.get(f"{base}/api/tags", timeout=3)
            if r.status_code != 200:
                return [{"id": self._model, "object": "model"}]
            data = r.json() or {}
            items = data.get("models", [])
            out: List[Dict[str, Any]] = []
            for m in items:
                if isinstance(m, dict):
                    name = m.get("name") or m.get("model")
                    if name:
                        out.append({"id": name, "object": "model"})
            if not out:
                out = [{"id": self._model, "object": "model"}]
            return out
        except Exception:
            return [{"id": self._model, "object": "model"}]

    def chat_completions(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        stream: bool = False,
        gen_params: Optional[Dict[str, Any]] = None,
    ) -> Union[Dict[str, Any], Iterable[Dict[str, Any]]]:
        self._ensure_client()
        model_name = model or self._model

        # 将 OpenAI 风格 messages 转为 Ollama 风格
        # 若包含图像，放入每条 message 的 "images": [b64, ...]
        ollama_messages: List[Dict[str, Any]] = []
        for msg in messages:
            content = msg.get("content", "")
            role = msg.get("role", "user")
            if isinstance(content, str):
                ollama_messages.append({"role": role, "content": content})
            elif isinstance(content, list):
                text_parts: List[str] = []
                images_b64: List[str] = []
                for part in content:
                    ptype = part.get("type")
                    if ptype == "text":
                        text_parts.append(str(part.get("text", "")))
                    elif ptype in ("image_url", "image_base64"):
                        # 将图像以 b64 形式传给 ollama
                        if ptype == "image_url":
                            import requests, base64
                            url = part.get("image_url") or part.get("url")
                            if url:
                                resp = requests.get(url, timeout=10)
                                resp.raise_for_status()
                                images_b64.append(base64.b64encode(resp.content).decode("utf-8"))
                        else:
                            b64 = part.get("image_base64") or part.get("b64_json")
                            if b64:
                                images_b64.append(b64)
                joined = "\n".join(text_parts)
                msg_obj = {"role": role, "content": joined}
                if images_b64:
                    msg_obj["images"] = images_b64
                ollama_messages.append(msg_obj)
            else:
                # 无法识别，忽略该条
                continue

        try:
            if not stream:
                res = self._client.chat(model=model_name, messages=ollama_messages, stream=False)
                content = ""
                if res and isinstance(res, dict):
                    m = res.get("message") or {}
                    content = m.get("content", "")
                return build_chat_completion(model=model_name, content=content)

            def _iter_stream() -> Iterable[Dict[str, Any]]:
                for chunk in self._client.chat(model=model_name, messages=ollama_messages, stream=True):
                    # chunk 示例: {"message":{"content": "..."},"done":false,...}
                    try:
                        m = chunk.get("message") or {}
                        c = m.get("content", "")
                        if c:
                            yield build_chat_chunk(model=model_name, delta_content=c)
                    except Exception:
                        continue

            return _iter_stream()
        except Exception as e:
            raise RemoteEndpointError(f"Ollama chat failed: {e}") from e


# ====================== 本地 Provider（vLLM / SGLang / TGI-Local 适配） ======================

class VllmLocalProvider(Provider):
    """
    通过 vLLM Python API 实现本地推理（非 HTTP）。最小可用路径：
    - 仅实现文本生成；多模态与音频不支持
    - 流式：使用伪流（按片切割文本）
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._model = config.get("model") or "sshleifer/tiny-gpt2"
        self._llm = None
        self._device = config.get("device", "cpu")

    @property
    def name(self) -> str:
        return "vllm"

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            supports_text_generation=True,
            supports_streaming=True,   # 伪流
            supports_embeddings=False,
            supports_vision=False,
            supports_audio_stt=False,
        )

    def _ensure_llm(self):
        if self._llm is not None:
            return
        try:
            from vllm import LLM  # type: ignore
        except Exception as e:
            raise CapabilityNotSupported(f"vLLM not installed or unavailable: {e}") from e
        # 直接使用默认初始化参数；如需精细控制可扩展（device/dtype 等）
        # 注意：部分小模型未必适配 vLLM，本地验证需选择合适模型
        try:
            self._llm = LLM(model=self._model)
        except Exception as e:
            raise CapabilityNotSupported(
                f"vLLM cannot initialize model '{self._model}'. "
                f"Please choose a vLLM-compatible HF model (e.g., 'facebook/opt-125m', "
                f"'TinyLlama/TinyLlama-1.1B-Chat-v1.0'). Underlying error: {e}"
            ) from e

    def chat_completions(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        stream: bool = False,
        gen_params: Optional[Dict[str, Any]] = None,
    ) -> Union[Dict[str, Any], Iterable[Dict[str, Any]]]:
        self._ensure_llm()
        from vllm import SamplingParams  # type: ignore

        prompt, _ = extract_text_from_messages(messages, allow_vision=False)
        params = gen_params or {}
        max_tokens = _get_int(params, "max_tokens", self.config.get("max_tokens", 64)) or 64
        temperature = float(params.get("temperature", self.config.get("temperature", 0.7)))
        top_p = float(params.get("top_p", self.config.get("top_p", 0.95)))
        top_k = _get_int(params, "top_k", self.config.get("top_k", 50)) or 50
        repetition_penalty = float(params.get("repetition_penalty", self.config.get("repetition_penalty", 1.0)))

        sp = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )

        # vLLM Python API 返回 RequestOutput 列表
        outputs = self._llm.generate([prompt], sp)
        text = ""
        if outputs and hasattr(outputs[0], "outputs") and outputs[0].outputs:
            # 取第一条输出
            try:
                text = outputs[0].outputs[0].text
            except Exception:
                # 某些版本属性名不同，做兜底
                text = str(getattr(outputs[0], "text", "")) or ""

        model_name = model or self._model
        if not stream:
            return build_chat_completion(model=model_name, content=text or "")

        def _iter_slices(t: str, chunk_size: int = 32) -> Iterable[Dict[str, Any]]:
            for i in range(0, len(t), chunk_size):
                yield build_chat_chunk(model=model_name, delta_content=t[i : i + chunk_size])

        return _iter_slices(text or "")


class SglangLocalProvider(Provider):
    """
    SGLang 本地适配：
    - 目标：不通过 HTTP，直接在进程内推理
    - 现实现为“轻量本地适配”：若本机无稳定 Python 原生推理 API，则退回到轻量 Transformers 管线以保证最小可运行效果
      （后续可替换为 sglang 的稳定本地 API）
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._model = config.get("model") or "sshleifer/tiny-gpt2"
        self._device = config.get("device", "cpu")
        self._gen_pipe = None

    @property
    def name(self) -> str:
        return "sglang"

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            supports_text_generation=True,
            supports_streaming=True,   # 伪流
            supports_embeddings=False,
            supports_vision=False,
            supports_audio_stt=False,
        )

    def _ensure_local_engine(self):
        # 优先尝试 sglang 是否已安装（仅探测）
        try:
            import sglang  # type: ignore  # noqa: F401
            # 当前不直接使用其不稳定 API；如需切换，在此处接入 sglang 原生 Runtime
        except Exception:
            # 未安装也可工作（回退 transformers）
            pass
        if self._gen_pipe is None:
            try:
                from transformers import pipeline  # type: ignore
                self._gen_pipe = pipeline(
                    "text-generation",
                    model=self._model,
                    device=0 if self._device != "cpu" else -1,
                )
            except Exception as e:
                raise CapabilityNotSupported(f"Local generation adapter unavailable: {e}") from e

    def chat_completions(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        stream: bool = False,
        gen_params: Optional[Dict[str, Any]] = None,
    ) -> Union[Dict[str, Any], Iterable[Dict[str, Any]]]:
        self._ensure_local_engine()
        prompt, _ = extract_text_from_messages(messages, allow_vision=False)
        params = gen_params or {}
        max_tokens = _get_int(params, "max_tokens", self.config.get("max_tokens", 64))
        temperature = float(params.get("temperature", self.config.get("temperature", 0.7)))
        top_p = float(params.get("top_p", self.config.get("top_p", 0.95)))
        top_k = _get_int(params, "top_k", self.config.get("top_k", 50))

        outputs = self._gen_pipe(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            return_full_text=False,
            num_return_sequences=1,
        )
        generated = outputs[0]["generated_text"] if outputs else ""

        model_name = model or self._model
        if not stream:
            return build_chat_completion(model=model_name, content=generated or "")

        def _iter_slices(text: str, chunk_size: int = 32) -> Iterable[Dict[str, Any]]:
            for i in range(0, len(text), chunk_size):
                yield build_chat_chunk(model=model_name, delta_content=text[i : i + chunk_size])

        return _iter_slices(generated or "")


class TgiLocalProvider(Provider):
    """
    TGI 本地最小适配：
    - TGI 官方以独立服务形态存在；此处提供一个轻量本地“同语义”适配，基于 transformers 管线实现
    - 用于本地演示与测试，不依赖 HTTP/远端
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._model = config.get("model") or "sshleifer/tiny-gpt2"
        self._device = config.get("device", "cpu")
        self._gen_pipe = None

    @property
    def name(self) -> str:
        return "tgi"

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            supports_text_generation=True,
            supports_streaming=True,  # 伪流
            supports_embeddings=False,
            supports_vision=False,
            supports_audio_stt=False,
        )

    def _ensure_pipe(self):
        if self._gen_pipe is None:
            try:
                from transformers import pipeline  # type: ignore
                self._gen_pipe = pipeline(
                    "text-generation",
                    model=self._model,
                    device=0 if self._device != "cpu" else -1,
                )
            except Exception as e:
                raise CapabilityNotSupported(f"TGI-local adapter unavailable: {e}") from e

    def chat_completions(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        stream: bool = False,
        gen_params: Optional[Dict[str, Any]] = None,
    ) -> Union[Dict[str, Any], Iterable[Dict[str, Any]]]:
        self._ensure_pipe()
        prompt, _ = extract_text_from_messages(messages, allow_vision=False)
        params = gen_params or {}
        max_tokens = _get_int(params, "max_tokens", self.config.get("max_tokens", 64))
        temperature = float(params.get("temperature", self.config.get("temperature", 0.7)))
        top_p = float(params.get("top_p", self.config.get("top_p", 0.95)))
        top_k = _get_int(params, "top_k", self.config.get("top_k", 50))

        outputs = self._gen_pipe(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            return_full_text=False,
            num_return_sequences=1,
        )
        generated = outputs[0]["generated_text"] if outputs else ""

        model_name = model or self._model
        if not stream:
            return build_chat_completion(model=model_name, content=generated or "")

        def _iter_slices(text: str, chunk_size: int = 32) -> Iterable[Dict[str, Any]]:
            for i in range(0, len(text), chunk_size):
                yield build_chat_chunk(model=model_name, delta_content=text[i : i + chunk_size])

        return _iter_slices(generated or "")


# ====================== 简易 HTTP 代理 Provider（vLLM/sglang/TGI） ======================

class OpenAICompatibleRemoteProvider(Provider):
    """
    将请求转发到远端 OpenAI 兼容接口（例如部分 vLLM/sglang 部署）。
    仅实现最小非流式路径；如需流式可扩展。
    """
    def __init__(self, config: Dict[str, Any], base_url_key: str, default_model: str = "unknown"):
        super().__init__(config)
        self._base_url = config.get(base_url_key)
        self._model = config.get("model", default_model)

    @property
    def name(self) -> str:
        return "remote-openai"

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            supports_text_generation=True,
            supports_streaming=False,   # 可后续扩展为 True，转发 SSE
            supports_embeddings=False,
            supports_vision=False,
            supports_audio_stt=False,
        )

    def _ensure_url(self):
        if not self._base_url:
            raise RemoteEndpointError("Remote base_url is not configured")

    def chat_completions(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        stream: bool = False,
        gen_params: Optional[Dict[str, Any]] = None,
    ) -> Union[Dict[str, Any], Iterable[Dict[str, Any]]]:
        if stream:
            raise CapabilityNotSupported("Remote streaming not implemented")
        self._ensure_url()
        import requests
        url = f"{self._base_url.rstrip('/')}/v1/chat/completions"
        payload = {
            "model": model or self._model,
            "messages": messages,
            "stream": False,
        }
        if gen_params:
            payload.update({k: v for k, v in gen_params.items() if k not in payload})
        try:
            resp = requests.post(url, json=payload, timeout=60)
            if resp.status_code >= 400:
                raise RemoteEndpointError(f"Remote error {resp.status_code}: {resp.text}")
            return resp.json()
        except Exception as e:
            raise RemoteEndpointError(f"Remote request failed: {e}") from e


class TGIProvider(Provider):
    """
    适配 HuggingFace TGI /generate 接口的最小实现（非流式）
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._base_url = config.get("tgi_base_url")
        self._model = config.get("model", "unknown")

    @property
    def name(self) -> str:
        return "tgi"

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            supports_text_generation=True,
            supports_streaming=False,  # 可扩展为流
            supports_embeddings=False,
            supports_vision=False,
            supports_audio_stt=False,
        )

    def _ensure_url(self):
        if not self._base_url:
            raise RemoteEndpointError("TGI base_url is not configured")

    def chat_completions(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        stream: bool = False,
        gen_params: Optional[Dict[str, Any]] = None,
    ) -> Union[Dict[str, Any], Iterable[Dict[str, Any]]]:
        if stream:
            raise CapabilityNotSupported("TGI streaming not implemented")
        import requests
        self._ensure_url()
        prompt, _ = extract_text_from_messages(messages, allow_vision=False)
        url = f"{self._base_url.rstrip('/')}/generate"
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": _get_int(gen_params or {}, "max_tokens", self.config.get("max_tokens", 64)),
                "temperature": float((gen_params or {}).get("temperature", self.config.get("temperature", 0.7))),
                "top_p": float((gen_params or {}).get("top_p", self.config.get("top_p", 0.95))),
                "top_k": _get_int(gen_params or {}, "top_k", self.config.get("top_k", 50)),
            },
        }
        try:
            resp = requests.post(url, json=payload, timeout=60)
            if resp.status_code >= 400:
                raise RemoteEndpointError(f"TGI error {resp.status_code}: {resp.text}")
            data = resp.json()
            # TGI 典型返回包含 generated_text
            text = data.get("generated_text") if isinstance(data, dict) else None
            if text is None and isinstance(data, list) and data:
                # 某些版本可能返回列表
                item = data[0]
                text = item.get("generated_text", "")
            return build_chat_completion(model=model or self._model, content=text or "")
        except Exception as e:
            raise RemoteEndpointError(f"TGI request failed: {e}") from e


# ====================== 注册默认 Provider 工厂 ======================

def register_default_providers(registry: ProviderRegistry) -> None:
    """
    向注册表注册各后端 Provider。
    """
    registry.register("transformers", lambda cfg: TransformersProvider(cfg))
    registry.register("transformer", lambda cfg: TransformersProvider(cfg))  # 兼容别名
    registry.register("ollama", lambda cfg: OllamaProvider(cfg))
    # 远端 OpenAI 兼容（vLLM/sglang）——均可使用 openai 风格端点
    registry.register("vllm", lambda cfg: VllmLocalProvider(cfg))
    registry.register("sglang", lambda cfg: SglangLocalProvider(cfg))
    # TGI
    registry.register("tgi", lambda cfg: TgiLocalProvider(cfg))