from illm.api.fast_api import start_uvicorn
from illm.logs.logger import initialize_logger,Logger
from illm.model.checker import install_missing_packages, is_port_in_use
from illm.model.args import *


def run():
    # 初始化日志
    initialize_logger()
    # 获取参数
    parse_arguments()
    # 检查包依赖
    install_missing_packages(config.get("framework"))
    # 检查参数
    if is_port_in_use(config.get("port")):
        Logger.error("Error: Port {port} is already in use. Please kill the process or use another port.")
        exit(0)
    # 启动fastApi服务
    start_uvicorn(config.get("host"),config.get("port"))