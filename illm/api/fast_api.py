import subprocess
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
import uvicorn
from ollama import AsyncClient
from starlette.responses import StreamingResponse
from illm.logs.logger import Logger
from illm.model.args import config
from illm.model.types import ChatRequest


@asynccontextmanager
async def lifespan(app: FastAPI, service_process=None):
    """生命周期管理器，启动与停止推理服务"""
    Logger.info("Starting inference service...")
    service_process = start_service(
        "ollama serve",
        "ollama", 11434
    )
    client = AsyncClient(host="http://127.0.0.1:11434")
    try:
        models = (await client.list()).get("models", [])
        Logger.info(f"Available models: {models}")
        if not any(m.model.startswith(config.get("model")) for m in models):
            Logger.info(f"Downloading {config.get('model')} ...")
            await client.pull(config.get("model"))
    except Exception as e:
        Logger.info(f"Error during model setup: {e}")

    try:
        yield  # FastAPI 应用运行时
    finally:
        Logger.info("Stopping inference service...")
        stop_service(service_process)

# FastAPI 应用实例
app = FastAPI(lifespan=lifespan)  # 注入生命周期管理器


def start_uvicorn(
    host: str,
    port: int,
):
    """启动 Uvicorn 服务器"""
    try:
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=False,
            access_log=True,
            log_level="debug"
        )
    except Exception as e:
        Logger.critical(f"Critical error starting Uvicorn server: {str(e)}")
        raise


# 动态启动服务的工具函数
def start_service(command: str, model_name: str, port: int):
    """启动推理服务"""
    try:
        process = subprocess.Popen(
            command.split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        Logger.info(f"{model_name} service started on port {port}")
        return process
    except Exception as e:
        Logger.error(f"Error starting service: {e}")
        return None


def stop_service(process):
    """停止推理服务"""
    if process:
        process.terminate()
        Logger.info("Service stopped.")
    else:
        Logger.warning("No process to stop.")


@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        Logger.info(f"Handling chat request: {request}")
        client = AsyncClient(host="http://127.0.0.1:11434")
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
                    Logger.info(f"Error during stream generation: {e}")
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
        Logger.info(f"Error handling chat request: {e}")
        raise HTTPException(status_code=500, detail=str(e))