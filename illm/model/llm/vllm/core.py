import subprocess
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from vllm import LLM, SamplingParams
import uvicorn
import socket

# 需要检查的依赖列表
required_packages = [
    'fastapi',
    'uvicorn',
    'vllm',
    'pydantic'
]

# 检查并安装缺失的包
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        print(f"Package {package} is missing. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# 启动 vLLM 服务
def start_vllm():
    """启动 vLLM 服务"""
    try:
        subprocess.Popen(
            ["vllm", "serve"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print("vLLM service started...")
    except Exception as e:
        print(f"Error starting vLLM: {e}")

# 停止 vLLM 服务
def stop_vllm():
    """停止 vLLM 服务"""
    try:
        subprocess.call(["pkill", "vllm"])
        print("vLLM service stopped.")
    except Exception as e:
        print(f"Error stopping vLLM: {e}")

# FastAPI 应用
app = FastAPI()

# 生命周期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    """生命周期管理 (替代旧的 on_event)"""
    start_vllm()  # 启动 vLLM 服务
    yield  # 应用运行期间
    stop_vllm()  # 关闭时停止 vLLM 服务

app = FastAPI(lifespan=lifespan)  # 注入生命周期管理器

# 请求模型
class ChatRequest(BaseModel):
    model: str = "mosaicml/mpt-7b"
    prompt: str
    stream: bool = False

# 聊天接口
@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        print(f"Handling chat request: {request}")
        llm = LLM(model=request.model)
        sampling_params = SamplingParams(max_new_tokens=128, top_k=10, top_p=0.95, temperature=0.8)

        if request.stream:
            async def generate():
                try:
                    async for chunk in llm.stream(request.prompt, sampling_params):
                        yield f"data: {chunk['text']}\n"
                except Exception as e:
                    print(f"Error during stream generation: {e}")
                    yield f"data: Error occurred while generating response.\n\n"

            return StreamingResponse(generate(), media_type="text/event-stream")
        else:
            response = await llm.generate(request.prompt, sampling_params)
            return {"response": response['text']}
    except Exception as e:
        print(f"Error handling chat request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 检查端口是否被占用
def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

# 启动 Uvicorn 服务器
if __name__ == "__main__":
    if is_port_in_use(8000):
        print("Error: Port 8000 is already in use. Please kill the process or use another port.")
    else:
        try:
            print("Starting Uvicorn server...")
            uvicorn.run(
                app,
                host="127.0.0.1",
                port=8000,
                reload=False,
                access_log=True,
                log_level="debug"  # 显示详细日志
            )
        except Exception as e:
            print(f"Critical error: {str(e)}")
            stop_vllm()  # 确保关闭 vLLM 进程
