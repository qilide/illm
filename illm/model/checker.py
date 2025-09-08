import os
import subprocess
import sys
import socket
from typing import List

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
def get_required_packages(framework: str) -> List[str]:
    """根据框架名称返回对应的依赖列表"""
    fw = (framework or "").lower()
    if fw == "ollama":
        return REQUIRED_OLLAMA_PACKAGES
    elif fw == "vllm":
        return REQUIRED_VLLM_PACKAGES
    elif fw in ("transformer", "transformers"):
        return REQUIRED_TRANSFORMERS_PACKAGES
    elif fw == "llama_cpp":
        return REQUIRED_LLAMACPP_PACKAGES
    elif fw == "sglang":
        return REQUIRED_SGLANG_PACKAGES
    elif fw == "tgi":
        return REQUIRED_TGI_PACKAGES
    else:
        raise ValueError(f"Unknown framework: {framework}")


def check_missing_packages(framework: str) -> List[str]:
    """返回缺失的包列表，不执行安装"""
    required_packages = get_required_packages(framework)
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    return missing


def install_missing_packages(framework: str):
    """
    检查并根据环境变量选择是否自动安装缺失的包。
    ILLM_AUTO_INSTALL = 1/true 时自动安装；否则仅提示并给出安装指引。
    """
    auto_install = str(os.environ.get("ILLM_AUTO_INSTALL", "0")).lower() in ("1", "true", "yes", "y")
    missing = check_missing_packages(framework)
    if not missing:
        Logger.info(f"All required packages satisfied for framework={framework}")
        return

    if not auto_install:
        Logger.warning(
            f"Missing packages for framework '{framework}': {missing}. "
            f"Set ILLM_AUTO_INSTALL=1 to auto-install or run: "
            f"python -m pip install " + " ".join(missing)
        )
        return

    for package in missing:
        try:
            Logger.warning(f"Package {package} is missing. Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except Exception as e:
            Logger.error(f"Failed to install package {package}: {e}")


def is_port_in_use(port):
    """检查端口是否被占用"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0