import subprocess
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from ollama import AsyncClient
import uvicorn
import socket


# 需要检查的依赖列表
required_packages = [
    'fastapi',
    'uvicorn',
    'ollama-python',
    'pydantic'
]

# 检查并安装缺失的包
for package in required_packages:
    try:
        # 尝试导入包，如果没有则捕获异常
        __import__(package)
    except ImportError:
        print(f"Package {package} is missing. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Ollama 进程管理
ollama_process = None

def start_ollama():
    """启动 Ollama 服务"""
    global ollama_process
    try:
        # 启动 Ollama 服务
        ollama_process = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print("Ollama service started...")
    except Exception as e:
        print(f"Error starting Ollama: {e}")

def stop_ollama():
    """停止 Ollama 服务"""
    global ollama_process
    if ollama_process:
        ollama_process.terminate()  # 使用 terminate() 来关闭进程
        ollama_process = None
        print("Ollama service stopped.")

client = AsyncClient(host="http://127.0.0.1:11434")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """生命周期管理 (替代旧的 on_event)"""
    start_ollama()

    try:
        models = (await client.list()).get("models", [])
        print(f"Available models: {models}")
        if not any(m.model.startswith("deepseek-r1") for m in models):
            print("Downloading deepseek-r1...")
            await client.pull("deepseek-r1:1.5b")
    except Exception as e:
        print(f"Error during model setup: {e}")

    yield  # 应用运行期间

    # 关闭时
    stop_ollama()

app = FastAPI(lifespan=lifespan)  # 注入生命周期管理器

class ChatRequest(BaseModel):
    model: str = "deepseek-r1:1.5b"
    prompt: str
    stream: bool = False

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        print(f"Handling chat request: {request}")
        if request.stream:
            async def generate():
                full_response = ""
                try:
                    async for chunk in await client.generate(
                            model=request.model,
                            prompt=request.prompt,
                            stream=True
                    ):
                        chunk_text = chunk["response"]
                        full_response += chunk_text
                    yield f"data: {full_response}\n"
                except Exception as e:
                    print(f"Error during stream generation: {e}")
                    yield f"data: Error occurred while generating response.\n\n"

            return StreamingResponse(generate(), media_type="text/event-stream")
        else:
            response = await client.generate(
                model=request.model,
                prompt=request.prompt,
                stream=False
            )
            return {"response": response["response"]}
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
                log_level="debug"  # 显示详细日志
            )
        except Exception as e:
            print(f"Critical error: {str(e)}")
            stop_ollama()  # 确保关闭 Ollama 进程
