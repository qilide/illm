import subprocess
import sys
import socket
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from huggingface_hub import InferenceClient
import uvicorn

# 检查并安装缺失的包
required_packages = [
    'fastapi',
    'uvicorn',
    'pydantic',
    'huggingface_hub'
]

for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        print(f"Package {package} is missing. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# TGI 进程管理
tgi_process = None

def stop_tgi():
    """停止 TGI 服务"""
    global tgi_process
    if tgi_process:
        tgi_process.terminate()
        tgi_process = None
        print("TGI service stopped.")

# 需要使用docker启动推理服务
# 初始化 InferenceClient
client = InferenceClient(
    base_url="http://127.0.0.1:8080",  # 根据您的 TGI 服务地址进行调整
)

app = FastAPI()

class ChatRequest(BaseModel):
    model: str = "HuggingFaceTB/SmolLM2-135M"  # 默认使用 DeepSeek-R1
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
                    output = client.chat.completions.create(
                        model=request.model,
                        messages=[{"role": "user", "content": request.prompt}],
                        stream=True,
                        max_tokens=1024,
                    )
                    async for chunk in output:
                        chunk_text = chunk.choices[0].delta.content
                        full_response += chunk_text
                        yield f"data: {chunk_text}\n\n"
                except Exception as e:
                    print(f"Error during stream generation: {e}")
                    yield f"data: Error occurred while generating response.\n\n"

            return StreamingResponse(generate(), media_type="text/event-stream")
        else:
            output = client.chat.completions.create(
                model=request.model,
                messages=[{"role": "user", "content": request.prompt}],
                stream=False,
                max_tokens=1024,
            )
            return {"response": output.choices[0].delta.content}
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
            stop_tgi()
