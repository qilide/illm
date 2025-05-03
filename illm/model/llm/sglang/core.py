import subprocess
import sys
import socket
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import requests

# 需要检查的依赖列表
required_packages = [
    'fastapi',
    'uvicorn',
    'pydantic',
    'sglang'
]

# 检查并安装缺失的包
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        print(f"Package {package} is missing. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# sglang 进程管理
sglang_process = None
sglang_port = 30000  # 设置 SGLang 服务的端口

def start_sglang():
    """启动 SGLang 服务"""
    global sglang_process
    try:
        sglang_process = subprocess.Popen(
            [
                "python", "-m", "sglang.launch_server",
                "--model-path", "Qwen/Qwen2.5-0.5B-Instruct",
                "--host", "0.0.0.0",
                "--port", str(sglang_port)
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print("SGLang service started...")
    except Exception as e:
        print(f"Error starting SGLang: {e}")

def stop_sglang():
    """停止 SGLang 服务"""
    global sglang_process
    if sglang_process:
        sglang_process.terminate()
        sglang_process = None
        print("SGLang service stopped.")

def wait_for_server(url, retries=10, delay=2):
    """等待服务器启动"""
    import time
    for _ in range(retries):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                print("Server is up and running.")
                return True
        except requests.RequestException:
            pass
        time.sleep(delay)
    print("Server did not start in time.")
    return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """生命周期管理"""
    start_sglang()
    yield
    stop_sglang()

app = FastAPI(lifespan=lifespan)

class ChatRequest(BaseModel):
    model: str = "HuggingFaceTB/SmolLM2-135M"
    prompt: str
    stream: bool = False

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        print(f"Handling chat request: {request}")
        url = f"http://localhost:{sglang_port}/v1/chat/completions"
        data = {
            "model": request.model,
            "messages": [{"role": "user", "content": request.prompt}],
        }
        if request.stream:
            def generate():
                with requests.post(url, json=data, stream=True) as response:
                    for chunk in response.iter_lines():
                        if chunk:
                            yield f"data: {chunk.decode()}\n\n"
            return StreamingResponse(generate(), media_type="text/event-stream")
        else:
            response = requests.post(url, json=data)
            return {"response": response.json()}
    except Exception as e:
        print(f"Error handling chat request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def is_port_in_use(port):
    """检查端口是否被占用"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

if __name__ == "__main__":
    if is_port_in_use(8000):
        print("Error: Port 8000 is already in use. Please kill the process or use another port.")
    else:
        try:
            print("Starting Uvicorn server...")
            uvicorn.run(app, host="127.0.0.1", port=8000, reload=False, access_log=True, log_level="debug")
        except Exception as e:
            print(f"Critical error: {str(e)}")
            stop_sglang()
