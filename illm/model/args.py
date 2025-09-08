import argparse
import os
from typing import Any, Dict, List, Optional, Union

from illm.model.common import Command

# 全局配置对象（被 server 与 API 使用）
config: Dict[str, Any] = {}


def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    return os.environ.get(key, default)


def _env_float(key: str, default: Optional[float]) -> Optional[float]:
    v = os.environ.get(key)
    if v is None:
        return default
    try:
        return float(v)
    except Exception:
        return default


def _env_int(key: str, default: Optional[int]) -> Optional[int]:
    v = os.environ.get(key)
    if v is None:
        return default
    try:
        return int(v)
    except Exception:
        return default


def _choices() -> List[str]:
    # 扩展兼容：原有枚举 + "transformers" 别名
    enum_vals = [e.value for e in Command]
    if "transformers" not in enum_vals:
        enum_vals.append("transformers")
    return enum_vals


def parse_arguments():
    """
    解析命令行参数到全局 config，带环境变量与默认值回退。
    优先级：命令行 > 环境变量 > 默认值
    """
    global config

    parser = argparse.ArgumentParser(description="启动 ILLM 推理服务")

    # 基础运行配置
    parser.add_argument(
        "framework",
        nargs="?",
        choices=_choices(),
        default=_env("ILLM_FRAMEWORK", "transformers"),
        help="选择推理框架：transformers | vllm | sglang | tgi | ollama 等",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=_env("ILLM_MODEL", "sshleifer/tiny-gpt2"),
        help="指定模型（名称或路径）",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=_env("ILLM_HOST", "0.0.0.0"),
        help="服务监听地址",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=_env_int("ILLM_PORT", 8000),
        help="服务端口",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=_env("ILLM_DEVICE", "cpu"),
        choices=["cpu", "cuda", "mps"],
        help="推理设备：cpu/cuda/mps",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=_env("ILLM_DTYPE", None),
        help="数据类型（如 float16/bfloat16/float32），不同后端可选",
    )

    # 远端端点配置
    parser.add_argument("--tgi_base_url", type=str, default=_env("ILLM_TGI_BASE_URL", None), help="TGI 基础 URL")
    parser.add_argument("--vllm_base_url", type=str, default=_env("ILLM_VLLM_BASE_URL", None), help="vLLM 基础 URL")
    parser.add_argument("--sglang_base_url", type=str, default=_env("ILLM_SGLANG_BASE_URL", None), help="SGLang 基础 URL")
    parser.add_argument("--ollama_base_url", type=str, default=_env("ILLM_OLLAMA_BASE_URL", "http://127.0.0.1:11434"), help="Ollama 基础 URL")

    # 生成参数（通用）
    parser.add_argument("--max_tokens", type=int, default=_env_int("ILLM_MAX_TOKENS", 64), help="最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=_env_float("ILLM_TEMPERATURE", 0.7), help="采样温度")
    parser.add_argument("--top_p", type=float, default=_env_float("ILLM_TOP_P", 0.95), help="top_p 采样")
    parser.add_argument("--top_k", type=int, default=_env_int("ILLM_TOP_K", 50), help="top_k 采样")
    parser.add_argument("--repetition_penalty", type=float, default=_env_float("ILLM_REPETITION_PENALTY", 1.0), help="重复惩罚因子")
    parser.add_argument(
        "--stop",
        action="append",
        default=None,
        help="停止词（可指定多次）",
    )

    # 可选：单独指定 embedding 模型
    parser.add_argument(
        "--embedding_model",
        type=str,
        default=_env("ILLM_EMBEDDING_MODEL", "sshleifer/tiny-distilbert-base-cased"),
        help="嵌入向量模型（Transformers 默认极小模型）",
    )

    args = parser.parse_args()

    # 标准化 framework（兼容 transformers/transformer）
    framework = (args.framework or "").lower()
    if framework == "transformers":
        framework = "transformers"  # 保持别名（注册表已同时支持 transformer/transformers）
    elif framework == "transformer":
        framework = "transformers"

    # 整合配置
    config.update(
        {
            "framework": framework,
            "model": args.model,
            "host": args.host,
            "port": args.port,
            "device": args.device,
            "dtype": args.dtype,
            "tgi_base_url": args.tgi_base_url,
            "vllm_base_url": args.vllm_base_url,
            "sglang_base_url": args.sglang_base_url,
            "ollama_base_url": args.ollama_base_url,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
            "stop": args.stop,
            "embedding_model": args.embedding_model,
        }
    )