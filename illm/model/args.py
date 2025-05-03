import argparse
from illm.model.common import Command

config = {}


def parse_arguments():
    """
    解析程序启动命令行参数，并返回解析结果。
    """
    global config

    parser = argparse.ArgumentParser(description="启动模型推理程序")

    # 添加命令参数
    parser.add_argument("framework", choices=[e.value for e in Command],
                        help="选择推理框架，例如：inference, vllm, ollama, sglang, transformer")

    # 添加可选的模型路径参数
    parser.add_argument("--model", type=str, required=True,
                        help="指定模型路径，例如：llama")

    # 添加主机地址参数
    parser.add_argument("--host", type=str, required=True,
                        help="指定主机地址，例如：127.0.0.1")

    # 添加端口参数
    parser.add_argument("--port", type=int, required=True,
                        help="指定端口号，例如：8080")

    # 解析命令行参数
    args = parser.parse_args()

    # 返回解析结果
    config["framework"] = args.framework
    config["model"] = args.model
    config["host"] = args.host
    config["port"] = args.port