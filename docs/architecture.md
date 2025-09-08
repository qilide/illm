# ILLM 架构与设计说明

本文档介绍 ILLM 的总体架构、模块职责、能力矩阵、扩展指引，以及安全与资源控制策略。该系统为多推理框架（Transformers、vLLM、SGLang、TGI、Ollama）提供统一的后端适配层，并对外暴露接近 OpenAI 风格的 API。


## 1. 总体架构

组件分层如下：

- 接入层（FastAPI）
  - 文件：illm/api/fast_api.py
  - 职责：实现 OpenAI 风格 API（/v1/chat/completions、/v1/completions、/v1/embeddings、/v1/audio/transcriptions、/healthz、/models），统一异常处理，SSE 流式响应
- Provider 抽象与注册表
  - 文件：illm/model/core.py
  - 职责：定义 Provider 抽象接口、能力声明、统一异常、通用响应构造、全局注册表实例
- 后端适配器（Providers）
  - 文件：illm/model/providers.py
  - 职责：针对不同框架实现适配器，例如：
    - Transformers（本地 CPU/GPU）
    - Ollama（本地/多模态）
    - TGI（/generate 接口）
    - vLLM/SGLang（远端 OpenAI 兼容端点）
- 参数与依赖
  - 文件：illm/model/args.py：CLI/env 配置解析与合并
  - 文件：illm/model/checker.py：依赖检查、可选自动安装、端口占用检测
- 服务入口
  - 文件：illm/server.py：装配日志、解析参数、依赖检查、启动 FastAPI
- 客户端示例与测试
  - 客户端：illm/client/
  - 测试：tests/

数据流简述：
1) 客户端请求进入 FastAPI 端点（/v1/...），请求体中可覆盖默认配置（framework、model 等）
2) API 层将请求与全局配置合并，依据 framework 从注册表选择对应 Provider 实例
3) 调用 Provider 执行对应能力（生成、嵌入、多模态、或远端转发），统一构造 OpenAI 兼容响应
4) 流式场景通过标准 SSE data: 行输出，并以 [DONE] 结束


## 2. 模块与文件说明

- illm/server.py
  - 启动入口：初始化日志、解析参数、依赖检查、端口检测、启动 Uvicorn

- illm/api/fast_api.py
  - FastAPI 应用与端点实现
  - 统一异常处理（ProviderError、通用异常）
  - SSE 流式输出处理

- illm/model/core.py
  - Provider 抽象接口与能力声明（supports_text_generation、supports_streaming、supports_embeddings、supports_vision、supports_audio_stt）
  - 统一异常类型：BadRequest、CapabilityNotSupported、RemoteEndpointError、ModelNotFound
  - OpenAI 风格响应构造（chat.completion、chat.completion.chunk、embeddings）
  - 全局 ProviderRegistry 实例

- illm/model/providers.py
  - TransformersProvider：本地最小可用路径（text-generation、feature-extraction）
  - OllamaProvider：本地聊天与多模态（图像部件），可列出本地已拉取模型
  - OpenAICompatibleRemoteProvider：将请求转发到远端 OpenAI 风格端点（用于 vLLM/sglang）
  - TGIProvider：对接 /generate 接口（最小非流式）
  - register_default_providers：注册上述 providers

- illm/model/args.py
  - 命令行与环境变量解析；优先级：请求体 > 环境/CLI > 默认
  - 支持项：framework、model、device、dtype、max_tokens、temperature、top_p、top_k、repetition_penalty、stop、embedding_model
  - 远端端点：tgi_base_url、vllm_base_url、sglang_base_url、ollama_base_url

- illm/model/checker.py
  - get_required_packages：按后端返回依赖列表（见 illm/model/common.py）
  - check_missing_packages / install_missing_packages
  - is_port_in_use：端口占用检查

- illm/model/common.py
  - 命令枚举、依赖列表常量、日志配置路径

- 客户端示例（illm/client/）
  - chat_completions.py：非流式与流式聊天
  - embeddings.py：向量嵌入
  - multimodal_chat.py：文本 + 图片部件（优先 Ollama）

- 测试（tests/）
  - test_api_minimal.py：Transformers 最小路径（/healthz、/v1/chat/completions 非流式、/v1/embeddings）
  - test_registry.py：注册表、能力检查
  - test_conditional_backends.py：条件存在时测试 Ollama、多模态、以及远端 vLLM/sglang/TGI（通过环境变量 base_url 控制）


## 3. 能力矩阵（概要）

- Transformers（本地）
  - Chat/Completions：支持
  - 流式：支持（片段切片伪流）
  - Embeddings：支持（feature-extraction + 均值池化）
  - Vision：不支持
  - Audio STT：不支持

- Ollama（本地）
  - Chat/Completions：支持
  - 流式：支持
  - Embeddings：暂未接入
  - Vision：支持（取决于模型，例如 llava、minicpm-v）
  - Audio STT：不支持

- TGI（远端 /generate）
  - Chat/Completions：支持（非流式）
  - 流式：未实现
  - Embeddings：未实现
  - Vision：未实现
  - Audio STT：未实现

- vLLM / SGLang（远端 OpenAI 风格）
  - Chat/Completions：支持（转发 /v1/chat/completions）
  - 流式：未实现（可扩展为转发 SSE）
  - Embeddings：未实现
  - Vision：未实现
  - Audio STT：未实现


## 4. OpenAI 风格数据契约

- /v1/chat/completions（非流式）
  - 返回 object="chat.completion"，包含 choices[0].message.content
- /v1/chat/completions（流式）
  - SSE data: 行，每行 JSON 对象的 object="chat.completion.chunk"
  - 以 data: [DONE] 结束
- /v1/embeddings
  - 返回 object="list"，data=[{"object":"embedding","embedding":[...]}]
- /v1/audio/transcriptions
  - 当后端不支持，返回 HTTP 422 与结构化错误体


## 5. 多模态文本 + 图像部件

在 chat/completions 的 messages[].content 中支持数组部件：
- {"type":"text","text":"..."}
- {"type":"image_url","image_url":"..."}
- {"type":"image_base64","image_base64":"..."} 或 {"b64_json":"..."}

适配策略：
- 仅文本后端收到图像部件将直接返回能力不支持（422）
- Ollama：将图像以 base64 嵌入到每条消息的 images 字段；具体是否支持由模型决定


## 6. 扩展新后端与新模态

扩展步骤：
1) 创建新 Provider 类（建议放在 illm/model/providers.py 或独立模块）
   - 实现至少 chat_completions（非流式）
   - 对于不支持的能力，应抛出 CapabilityNotSupported
2) 在注册函数 register_default_providers 中注册新后端键（如 "mybackend"）
3) 若支持 embeddings/vision/audio，按需实现对应方法或在 chat_completions 中解析图像部件
4) 在 README 与测试中添加条件用例或说明

注意事项：
- 能力声明（Capabilities）需准确
- 不要在 API 层硬编码后端逻辑，统一通过注册表选择
- 远端后端建议通过 base_url 参数配置
- 尽量遵循 OpenAI 风格响应，便于 SDK 复用


## 7. 安全与资源控制

- 输入限额：
  - API 层建议在 Nginx/网关侧限制请求体大小
  - 对图像部件已限制单图像原始数据大小（默认 8MB），避免 OOM
- 超时与重试：
  - 远端调用使用 requests 超时参数；可在部署侧通过网关/服务端设置合理超时
- 最大 tokens：
  - 通过 max_tokens 控制生成长度；Transformers Provider 采用 max_new_tokens
- 日志与错误：
  - 统一日志输出到控制台与文件（illm/logs/logger.py 的 initialize_logger）
  - 统一异常类在 API 层转换为结构化错误体（含类型与可诊断 message）
- 设备与 dtype：
  - 通过 --device（cpu/cuda/mps）与 --dtype 参数控制；不同后端支持程度不同
- 依赖检查与自动安装：
  - 在服务启动时按框架检查依赖；可通过 ILLM_AUTO_INSTALL 启用自动安装


## 8. 运行与验证（最小路径）

- 安装最小依赖：
  - python -m pip install fastapi uvicorn transformers torch requests pytest
- 启动服务（Transformers）：
  - python -m illm.server --framework transformers --model sshleifer/tiny-gpt2 --host 0.0.0.0 --port 8000
- 基本验证：
  - /healthz：返回 ok
  - /v1/chat/completions：非流式请求返回 object=chat.completion
  - /v1/embeddings：返回 object=list 与 data
- 运行测试：
  - pytest -q
- 条件验证：
  - 安装 Ollama 并拉取视觉模型（如 llava）后，可进行多模态请求
  - 设置 ILLM_TGI_BASE_URL / ILLM_VLLM_BASE_URL / ILLM_SGLANG_BASE_URL 时，自动跑基础用例


## 9. 后续改进方向

- 流式转发：为 vLLM/sglang/TGI 实现 SSE 转发，完善 stream=true 路径
- Embeddings 扩展：为 Ollama/远端添加向量化能力（若后端可用）
- STT 能力：接入本地 Whisper（Transformers）或其他 STT 后端
- 更完善的 usage 统计与计费钩子：统一记录 prompt/completion tokens
- 更严格的图像处理与安全策略：可借助 Pillow 验证文件格式、最大像素等（当前采用大小限制与解码前校验策略）