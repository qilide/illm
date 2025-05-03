import subprocess
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import socket
import requests

# 需要检查的依赖列表
required_packages = [
    'fastapi',
    'uvicorn',
    'pydantic',
    'requests',
    'llama-cpp-python'
]

# 检查并安装缺失的包
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        print(f"Package {package} is missing. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# llama.cpp 进程管理
llama_process = None

def start_llama():
    """启动 llama.cpp 服务"""
    global llama_process
    try:
        llama_process = subprocess.Popen(
            ["./llama-server", "-m", "path_to_your_model.gguf", "--host", "0.0.0.0", "--port", "8080"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print("llama.cpp service started...")
    except Exception as e:
        print(f"Error starting llama.cpp: {e}")

def stop_llama():
    """停止 llama.cpp 服务"""
    global llama_process
    if llama_process:
        llama_process.terminate()
        llama_process = None
        print("llama.cpp service stopped.")

client = None  # 不再需要 llama.cpp 客户端

@asynccontextmanager
async def lifespan(app: FastAPI):
    """生命周期管理 (替代旧的 on_event)"""
    start_llama()
    yield
    stop_llama()

app = FastAPI(lifespan=lifespan)

class ChatRequest(BaseModel):
    model: str = "path_to_your_model.gguf"
    prompt: str
    stream: bool = False

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        print(f"Handling chat request: {request}")

        def run_llama_inference(prompt: str):
            """调用 llama.cpp 进行推理"""
            try:
                response = requests.post(
                    "http://localhost:8080/v1/chat/completions",
                    json={"model": request.model, "messages": [{"role": "user", "content": prompt}]}
                )
                return response.json()
            except Exception as e:
                print(f"Error during inference: {e}")
                return {"error": "Inference failed"}

        if request.stream:
            async def generate():
                try:
                    response = run_llama_inference(request.prompt)
                    full_response = response.get("choices", [{}])[0].get("message", {}).get("content", "")
                    yield f"data: {full_response}\n"
                except Exception as e:
                    print(f"Error during stream generation: {e}")
                    yield f"data: Error occurred while generating response.\n\n"

            return StreamingResponse(generate(), media_type="text/event-stream")
        else:
            response = run_llama_inference(request.prompt)
            return {"response": response}

    except Exception as e:
        print(f"Error handling chat request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

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
                log_level="debug"
            )
        except Exception as e:
            print(f"Critical error: {str(e)}")
            stop_llama()  # 确保关闭 llama.cpp 进程
