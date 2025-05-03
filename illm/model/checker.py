import subprocess
import sys
import socket
from illm.logs.logger import Logger
from illm.model.common import (
    REQUIRED_OLLAMA_PACKAGES,
    REQUIRED_VLLM_PACKAGES,
    REQUIRED_TRANSFORMERS_PACKAGES,
    REQUIRED_LLAMACPP_PACKAGES,
    REQUIRED_SGLANG_PACKAGES,
    REQUIRED_TGI_PACKAGES,
)


# 根据框架选择对应的依赖列表
def get_required_packages(framework: str):
    """根据框架名称返回对应的依赖列表"""
    if framework == "ollama":
        return REQUIRED_OLLAMA_PACKAGES
    elif framework == "vllm":
        return REQUIRED_VLLM_PACKAGES
    elif framework == "transformers":
        return REQUIRED_TRANSFORMERS_PACKAGES
    elif framework == "llama_cpp":
        return REQUIRED_LLAMACPP_PACKAGES
    elif framework == "sglang":
        return REQUIRED_SGLANG_PACKAGES
    elif framework == "tgi":
        return REQUIRED_TGI_PACKAGES
    else:
        raise ValueError(f"Unknown framework: {framework}")


def install_missing_packages(framework: str):
    """检查并安装缺失的包"""
    required_packages = get_required_packages(framework)
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            Logger.warning(f"Package {package} is missing. Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def is_port_in_use(port):
    """检查端口是否被占用"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0