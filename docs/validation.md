# 实际联通与验证记录

本文记录在本机环境中对各后端进行的实际联通与最小验证，便于复现实验与问题诊断。服务由
[`illm/server.py`](illm/server.py) 启动，API 实现在
[`illm/api/fast_api.py`](illm/api/fast_api.py)。以下所有请求默认指向 http://127.0.0.1:8001。

环境与基线
- 框架：transformers（本地最小验证路径）
- 默认模型：sshleifer/tiny-gpt2
- 服务器健康检查已通过

1) 健康检查 GET /healthz
- 请求: curl -s http://127.0.0.1:8001/healthz
- 响应:
{"status":"ok","framework":"transformers","model":"sshleifer/tiny-gpt2","providers":["transformers","transformer","ollama","vllm","sglang","tgi"]}

2) 模型列表 GET /models
- 请求: curl -s http://127.0.0.1:8001/models
- 响应:
{"object":"list","data":[{"id":"sshleifer/tiny-gpt2","object":"model"}]}

3) Chat Completions（Transformers，非流式）
- 请求: curl -s -X POST http://127.0.0.1:8001/v1/chat/completions -H "Content-Type: application/json" -d '{"framework":"transformers","model":"sshleifer/tiny-gpt2","messages":[{"role":"user","content":"Hello from transformers"}],"max_tokens":8,"temperature":0.0,"stream":false}'
- 响应:
{"id":"chatcmpl-1757078342818","object":"chat.completion","created":1757078342,"model":"sshleifer/tiny-gpt2","choices":[{"index":0,"message":{"role":"assistant","content":" stairs stairs stairs stairs stairs stairs stairs stairs"},"finish_reason":"stop"}],"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}

4) 兼容 Completions（POST /v1/completions）
- 请求: curl -s -X POST http://127.0.0.1:8001/v1/completions -H "Content-Type: application/json" -d '{"framework":"transformers","model":"sshleifer/tiny-gpt2","prompt":"Hello legacy","max_tokens":8,"temperature":0.0}'
- 响应:
{"id":"chatcmpl-1757078394045","object":"chat.completion","created":1757078394,"model":"sshleifer/tiny-gpt2","choices":[{"index":0,"message":{"role":"assistant","content":" factors factors factors factors factors factors factors factors"},"finish_reason":"stop"}],"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}

5) Embeddings（Transformers）
- 请求: curl -s -X POST http://127.0.0.1:8001/v1/embeddings -H "Content-Type: application/json" -d '{"framework":"transformers","embedding_model":"sshleifer/tiny-distilbert-base-cased","input":["hello","world"]}'
- 响应:
{"object":"list","model":"sshleifer/tiny-distilbert-base-cased","model_replica":"default","data":[{"index":0,"object":"embedding","embedding":[-1.0,1.0000000794728596]},{"index":1,"object":"embedding","embedding":[-1.0,1.0000000794728596]}],"usage":{"prompt_tokens":0,"total_tokens":0}}

6) Ollama - 模型列表（回退行为）
- 请求: curl -s "http://127.0.0.1:8001/models?framework=ollama"
- 响应:
{"object":"list","data":[{"id":"sshleifer/tiny-gpt2","object":"model"}]}

7) Ollama - Chat（本机未安装 Python 客户端）
- 请求: curl -i -s -X POST http://127.0.0.1:8001/v1/chat/completions -H "Content-Type: application/json" -d '{"framework":"ollama","model":"llava:7b","messages":[{"role":"user","content":"hello"}]}'
- 响应:
HTTP/1.1 400 Bad Request
{"error":{"type":"RemoteEndpointError","message":"Ollama client not available: No module named 'ollama'"}}

8) vLLM/SGLang/TGI - 未配置 base_url 的错误回显
- 请求（示例）:
  - vLLM: POST /v1/chat/completions {"framework":"vllm","model":"placeholder","messages":[{"role":"user","content":"hello"}]}
  - SGLang: POST /v1/chat/completions {"framework":"sglang","model":"placeholder","messages":[{"role":"user","content":"hello"}]}
  - TGI: POST /v1/chat/completions {"framework":"tgi","model":"placeholder","messages":[{"role":"user","content":"hello"}]}
- 响应:
HTTP/1.1 400 Bad Request
{"error":{"type":"RemoteEndpointError","message":"Remote base_url is not configured"}}
HTTP/1.1 400 Bad Request
{"error":{"type":"RemoteEndpointError","message":"Remote base_url is not configured"}}
HTTP/1.1 400 Bad Request
{"error":{"type":"RemoteEndpointError","message":"TGI base_url is not configured"}}

9) Ollama 本地服务探测（可选）
- 请求:
  - curl -s http://127.0.0.1:11434/api/version
  - curl -s http://127.0.0.1:11434/api/tags
- 输出：命令执行但无返回，推测本机未运行或端口未开放（仅作记录）

备注
- 上述输出为本机一次真实跑通结果的直接摘录，用于联通性与错误路径验证。
- 如需实际与远端 vLLM/SGLang/TGI 联通，请设置对应的 ILLM_*_BASE_URL 环境变量，并确保服务可达。
- 如需 Ollama 多模态能力，请安装并运行本机 Ollama 服务，并安装 Python 客户端包（pip install ollama）。
10) vLLM 本地（代码级）验证
- 请求:
curl -i -s -X POST http://127.0.0.1:8002/v1/chat/completions -H "Content-Type: application/json" -d '{"framework":"vllm","model":"sshleifer/tiny-gpt2","messages":[{"role":"user","content":"hello via vllm local"}],"stream":false}'
- 响应（已按实现回退为明确可诊断错误，需选择 vLLM 兼容模型）:
HTTP/1.1 422 Unprocessable Entity
{"error":{"type":"CapabilityNotSupported","message":"vLLM cannot initialize model 'sshleifer/tiny-gpt2'. Please choose a vLLM-compatible HF model (e.g., 'facebook/opt-125m', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'). Underlying error: 'GPT2Config' object has no attribute 'get_text_config'"}}
- 说明：vLLM 对模型支持有特定要求，tiny-gpt2 并不兼容。请选择如 facebook/opt-125m、TinyLlama/TinyLlama-1.1B-Chat-v1.0 等兼容模型后重试。

11) SGLang 本地（代码级，轻量适配）验证
- 请求:
curl -i -s -X POST http://127.0.0.1:8002/v1/chat/completions -H "Content-Type: application/json" -d '{"framework":"sglang","model":"sshleifer/tiny-gpt2","messages":[{"role":"user","content":"hello via sglang local"}],"stream":false}'
- 响应（节选，状态 200 OK，返回 OpenAI 风格）:
HTTP/1.1 200 OK
{"id":"chatcmpl-...","object":"chat.completion","created":...,"model":"sshleifer/tiny-gpt2","choices":[{"index":0,"message":{"role":"assistant","content":" ... "},"finish_reason":"stop"}],"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}
- 说明：当前为本地轻量适配（回退 transformers 管线）以保证最小可运行路径；后续可替换为 sglang 稳定本地 API。

12) TGI 本地（代码级，轻量适配）验证
- 请求:
curl -i -s -X POST http://127.0.0.1:8002/v1/chat/completions -H "Content-Type: application/json" -d '{"framework":"tgi","model":"sshleifer/tiny-gpt2","messages":[{"role":"user","content":"hello via tgi local"}],"stream":false}'
- 响应（节选，状态 200 OK，返回 OpenAI 风格）:
HTTP/1.1 200 OK
{"id":"chatcmpl-...","object":"chat.completion","created":...,"model":"sshleifer/tiny-gpt2","choices":[{"index":0,"message":{"role":"assistant","content":" ... "},"finish_reason":"stop"}],"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}
- 说明：TGI 官方为独立服务形态，本地“同语义”适配基于 transformers 管线，便于离线演示与测试。

小结
- 本地三后端现状：
  - vLLM：已实现 Python API 本地集成；需选择兼容模型才可成功生成（tiny-gpt2 不支持）
  - SGLang：提供本地轻量适配（默认回退 transformers），可成功返回
  - TGI：提供本地轻量适配（默认回退 transformers），可成功返回
- 如需真实 vLLM/SGLang/TGI 行为，可在同一实现下切换至远端部署（或替换本地适配为官方本地 API）并配置对应模型/资源。