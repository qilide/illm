# ILLM - 统一多后端推理服务

面向多推理框架（Transformers、vLLM、SGLang、TGI、Ollama）的统一后端服务，提供 OpenAI 兼容风格的 API，支持：
- 文本对话与补全（Chat/Completions）
- 向量嵌入（Embeddings）
- 多模态对话（文本 + 图像，优先基于 Ollama 视觉模型）
-（可选）语音转写接口占位（若后端不支持则返回能力不支持）

服务遵循统一 Provider 抽象与注册表机制，可按需扩展更多后端与模态。


## 目录与关键文件
- 入口与 API
  - illm/server.py
  - illm/api/fast_api.py
- 核心模型层
  - illm/model/core.py（Provider 抽象、能力与统一异常、注册表实例）
  - illm/model/providers.py（Transformers/Ollama/TGI/vLLM/SGLang 适配器与注册工厂）
  - illm/model/args.py（CLI/环境变量配置）
  - illm/model/checker.py（依赖检查与可选自动安装）
- 客户端示例
  - illm/client/chat_completions.py
  - illm/client/embeddings.py
  - illm/client/multimodal_chat.py
- 测试
  - tests/test_api_minimal.py（最小可运行路径）
  - tests/test_registry.py（注册表与能力）
  - tests/test_conditional_backends.py（条件后端：Ollama、远端 vLLM/sglang/TGI）
- 设计文档
  - docs/architecture.md


## 支持的后端能力（概览）
- Transformers（本地）
  - 文本生成：支持
  - 流式：支持（片段切片伪流）
  - Embeddings：支持（feature-extraction+均值池化）
  - 多模态：不支持
- Ollama（本地/视觉）
  - 文本生成：支持
  - 流式：支持（SSE 转发）
  - Embeddings：当前未接入
  - 多模态：支持（依赖具体模型，如 llava、minicpm-v）
- TGI（本地轻量适配或远端 /generate）
  - 文本生成：支持（本地轻量适配基于 transformers；亦可连接远端 /generate）
  - 流式：未实现（本地适配为伪流，可扩展）
  - Embeddings/多模态：未实现
- vLLM / SGLang（远端，OpenAI风格）
  - 文本生成：支持（通过远端 /v1/chat/completions）
  - 流式：未实现（可扩展）
  - Embeddings/多模态：未实现


## 安装与依赖

最小验证依赖（本地 Transformers CPU 路径）：
- python -m pip install fastapi uvicorn transformers torch requests pytest

建议：首次运行会下载极小测试模型（如 sshleifer/tiny-gpt2、sshleifer/tiny-distilbert-base-cased），需联网。

按需安装：
- Ollama（本地服务）参见 https://ollama.com
- 远端 vLLM、SGLang、TGI：自行部署对应服务并配置 base_url


## 启动服务

示例（本地 Transformers 最小可运行路径）：
python -m illm.server transformers --model sshleifer/tiny-gpt2 --host 0.0.0.0 --port 8000

环境变量（可替代 CLI 参数，优先级：请求体 > 环境变量/CLI > 默认）：
- ILLM_FRAMEWORK（默认 transformers）
- ILLM_MODEL（默认 sshleifer/tiny-gpt2）
- ILLM_HOST（默认 0.0.0.0）
- ILLM_PORT（默认 8000）
- ILLM_DEVICE（cpu/cuda/mps；默认 cpu）
- ILLM_TGI_BASE_URL、ILLM_VLLM_BASE_URL、ILLM_SGLANG_BASE_URL、ILLM_OLLAMA_BASE_URL
- ILLM_MAX_TOKENS、ILLM_TEMPERATURE、ILLM_TOP_P、ILLM_TOP_K、ILLM_REPETITION_PENALTY
- ILLM_EMBEDDING_MODEL（默认 sshleifer/tiny-distilbert-base-cased）
- ILLM_AUTO_INSTALL（1/true 则自动安装缺失依赖）



## API 端点（OpenAI 风格）

通用说明：
- 请求体可额外附带 framework 与 model 覆盖运行配置
- 流式响应采用标准 SSE data: 行，以 [DONE] 结束
- 出错返回结构化错误体，含 type 与 message

1) POST /v1/chat/completions
- 请求字段（关键）：
  - model：模型名
  - messages：OpenAI 风格消息数组
    - content 可为：
      - 字符串
      - 数组部件：{"type":"text","text":"..."} 或 {"type":"image_url","image_url":"..."} 或 {"type":"image_base64","image_base64":"..."}
  - stream：bool，是否流式
  - 生成参数：max_tokens、temperature、top_p、top_k、repetition_penalty、stop 等
- 响应：OpenAI chat.completion 或 chat.completion.chunk（流式）

2) POST /v1/completions
- 兼容旧版：使用 prompt 转换为 messages 后复用 /v1/chat/completions

3) POST /v1/embeddings
- 字段：
  - input（字符串或字符串数组）
  - model 或 embedding_model（可覆盖）
- 响应：OpenAI Embeddings 风格（object=list, data=[embedding...]）

4) POST /v1/audio/transcriptions（可选）
- file=音频文件 或 audio_base64
- 若选用后端不支持，返回能力不支持错误（422）

5) GET /healthz
- 返回服务状态与可用 provider 列表

6) GET /models
- 返回当前可用模型列表
- Transformers 默认返回当前模型
- Ollama 若本地可用则列出已拉取模型（/api/tags）



## 使用示例（curl）

1) 非流式 Chat（Transformers）
curl -s -X POST http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"framework":"transformers","model":"sshleifer/tiny-gpt2","messages":[{"role":"user","content":"Hello"}],"max_tokens":16,"temperature":0.0,"stream":false}'

2) 流式 Chat（Transformers，片段伪流）
curl -N -s -X POST http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"framework":"transformers","model":"sshleifer/tiny-gpt2","messages":[{"role":"user","content":"Say hi thrice"}],"max_tokens":24,"temperature":0.7,"stream":true}'

3) Embeddings（Transformers）
curl -s -X POST http://127.0.0.1:8000/v1/embeddings -H "Content-Type: application/json" -d '{"framework":"transformers","model":"sshleifer/tiny-gpt2","embedding_model":"sshleifer/tiny-distilbert-base-cased","input":["hello","world"]}'

4) 多模态（Ollama，本机需安装且具备视觉模型）
curl -s -X POST http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"framework":"ollama","model":"llava:7b","messages":[{"role":"user","content":[{"type":"text","text":"Describe this image"},{"type":"image_url","image_url":"https://httpbin.org/image/png"}]}]}'


## Python 客户端示例脚本

非流式/流式 Chat
python illm/client/chat_completions.py --framework transformers --model sshleifer/tiny-gpt2 --prompt "Hello" --max_tokens 16
python illm/client/chat_completions.py --framework transformers --model sshleifer/tiny-gpt2 --prompt "Say hi thrice" --stream --max_tokens 24

Embeddings
python illm/client/embeddings.py --framework transformers --embedding_model sshleifer/tiny-distilbert-base-cased --inputs hello world

多模态（Ollama）
python illm/client/multimodal_chat.py --framework ollama --model llava:7b --text "Describe this image" --image_url https://httpbin.org/image/png



## 远端后端配置（可选）

- vLLM：设置 ILLM_VLLM_BASE_URL，例如 http://127.0.0.1:8001
- SGLang：设置 ILLM_SGLANG_BASE_URL
- TGI：设置 ILLM_TGI_BASE_URL（/generate 接口）
- 客户端请求中可直接指定 framework 为 vllm/sglang/tgi 与 model 字段

联通测试与示例（按需选择）：

1) vLLM / SGLang（OpenAI 兼容端点）
- 环境变量
  export ILLM_VLLM_BASE_URL="http://127.0.0.1:8002"
  export ILLM_SGLANG_BASE_URL="http://127.0.0.1:8003"
- 调用示例（非流式）
  curl -s -X POST http://127.0.0.1:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"framework":"vllm","model":"your-vllm-model","messages":[{"role":"user","content":"hello"}]}'
  curl -s -X POST http://127.0.0.1:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"framework":"sglang","model":"your-sglang-model","messages":[{"role":"user","content":"hello"}]}'
- 说明
  - 未配置 base_url 时将返回错误：{"error":{"type":"RemoteEndpointError","message":"Remote base_url is not configured"}}
  - 当前仅非流式；如需流式可后续扩展为 SSE 转发

2) TGI（/generate）
- 环境变量
  export ILLM_TGI_BASE_URL="http://127.0.0.1:8080"
- 调用示例（非流式）
  curl -s -X POST http://127.0.0.1:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"framework":"tgi","model":"your-tgi-model","messages":[{"role":"user","content":"hello from tgi"}]}'
- 说明
  - 未配置 base_url 时将返回：{"error":{"type":"RemoteEndpointError","message":"TGI base_url is not configured"}}

3) Ollama（本地/多模态）
- 依赖与服务
  - 安装 Python 客户端：python -m pip install ollama
  - 本机需已安装并运行 ollama 服务（参考 https://ollama.com），默认端口 11434
  - 可验证本地服务：curl -s http://127.0.0.1:11434/api/tags
- 模型列出
  - 调用：curl -s "http://127.0.0.1:8000/models?framework=ollama"
  - 若本机未运行 ollama 或无可用模型，将回退为返回默认占位模型
- 文本对话
  curl -s -X POST http://127.0.0.1:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"framework":"ollama","model":"llama3:8b","messages":[{"role":"user","content":"hello"}]}'
- 多模态（示例使用 llava 系列）
  curl -s -X POST http://127.0.0.1:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"framework":"ollama","model":"llava:7b","messages":[{"role":"user","content":[{"type":"text","text":"Describe this image"},{"type":"image_url","image_url":"https://httpbin.org/image/png"}]}]}'
- 说明
  - 若返回 {"error":{"type":"RemoteEndpointError","message":"Ollama client not available: No module named '\''ollama'\''"}}：
    - 需安装 Python 客户端（python -m pip install ollama）并确保本机 ollama 服务正在运行



## 运行测试

最小验证（Transformers）：
pytest -q

条件测试（Ollama、远端）：
- 若本机有 Ollama，将运行多模态与文本用例
- 设置 ILLM_TGI_BASE_URL / ILLM_VLLM_BASE_URL / ILLM_SGLANG_BASE_URL 则各自跑一次基本用例；未配置则自动跳过



## 故障诊断与常见问题

- 首次运行卡顿：首次下载模型耗时较长，建议选择 tiny 系列模型
- 安装依赖失败：可设置 ILLM_AUTO_INSTALL=1 让服务自动安装缺失包，或手动安装
- 端口占用：更换 --port 或释放端口
- 多模态无效：确认使用的 Ollama 模型具备视觉能力（如 llava），并确保本机服务已启动且可访问
- 远端不可用：检查 base_url 正确性与网络连通性



## 设计文档

详见 docs/architecture.md，包含架构说明、能力矩阵、扩展指引与安全/资源控制策略。

## 验证记录与结果

- 完整的实际联通请求与原始输出，已整理至 [`docs/validation.md`](docs/validation.md)
- 快速复现（指向 127.0.0.1:8000/8001 任选其一端口，以下以 8000 为例）：
  - 健康检查：curl -s http://127.0.0.1:8000/healthz
  - 模型列表：curl -s http://127.0.0.1:8000/models
  - Chat（Transformers 非流式）：curl -s -X POST http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"framework":"transformers","model":"sshleifer/tiny-gpt2","messages":[{"role":"user","content":"Hello"}],"max_tokens":8,"temperature":0.0,"stream":false}'
  - 兼容 Completions：curl -s -X POST http://127.0.0.1:8000/v1/completions -H "Content-Type: application/json" -d '{"framework":"transformers","model":"sshleifer/tiny-gpt2","prompt":"Hello legacy","max_tokens":8,"temperature":0.0}'
  - Embeddings：curl -s -X POST http://127.0.0.1:8000/v1/embeddings -H "Content-Type: application/json" -d '{"framework":"transformers","embedding_model":"sshleifer/tiny-distilbert-base-cased","input":["hello","world"]}'
- 远端/本地其他后端的错误路径与联通说明：
  - 未配置 vLLM / SGLang / TGI base_url 时，将返回结构化错误（RemoteEndpointError）
  - 本机未安装或未运行 Ollama / 未安装 Python 客户端时，将返回明确错误提示
  - 详细截图与返回体见 [`docs/validation.md`](docs/validation.md)
