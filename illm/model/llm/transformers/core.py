import subprocess
import sys
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import pipeline
import uvicorn
import socket

# 需要检查的依赖列表
required_packages = [
    'fastapi',
    'uvicorn',
    'transformers',
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

# FastAPI 应用实例
app = FastAPI()

class ChatRequest(BaseModel):
    model: str = "HuggingFaceTB/SmolLM2-135M"
    prompt: str
    stream: bool = False

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        print(f"Handling chat request: {request}")
        generator = pipeline("text-generation", model=request.model)

        if request.stream:
            async def generate():
                full_response = ""
                try:
                    for chunk in generator(request.prompt, max_length=50, num_return_sequences=1):
                        chunk_text = chunk[0]['generated_text']
                        full_response += chunk_text
                        yield f"data: {chunk_text}\n"
                except Exception as e:
                    print(f"Error during stream generation: {e}")
                    yield f"data: Error occurred while generating response.\n\n"

            return StreamingResponse(generate(), media_type="text/event-stream")
        else:
            response = generator(request.prompt, max_length=50, num_return_sequences=1)
            return {"response": response[0]['generated_text']}
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
