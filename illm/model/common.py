from enum import Enum, unique

# 日志文件路径
LOG_FILE_PATH = "illm.log"


@unique
class Command(str, Enum):
    INFERENCE = "inference"
    VLLM = "vllm"
    OLLAMA = "ollama"
    SGLANG = "sglang"
    TRANSFORMER = "transformer"
    TGI = "tgi"
    VERSION = "version"
    HELP = "help"
    ILLM = "illm"


# ollama的依赖列表
REQUIRED_OLLAMA_PACKAGES = [
    'fastapi',
    'uvicorn',
    'pydantic',
    'transformers',
    'requests',
]

# vLLM的依赖列表
REQUIRED_VLLM_PACKAGES = [
    'fastapi',
    'uvicorn',
    'vllm',
    'pydantic'
]


# transformer的依赖列表
REQUIRED_TRANSFORMERS_PACKAGES = [
    'fastapi',
    'uvicorn',
    'transformers',
    'pydantic'
]


# llama_cpp的依赖列表
REQUIRED_LLAMACPP_PACKAGES = [
    'fastapi',
    'uvicorn',
    'pydantic',
    'requests',
    'llama-cpp-python'
]


# sglang的依赖列表
REQUIRED_SGLANG_PACKAGES = [
    'fastapi',
    'uvicorn',
    'pydantic',
    'sglang'
]


# tgi的依赖列表
REQUIRED_TGI_PACKAGES = [
    'fastapi',
    'uvicorn',
    'pydantic',
    'huggingface_hub'
]