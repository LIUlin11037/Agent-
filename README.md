# Minimal Agent —— 从零实现的最小可用 Agent

一个**不依赖任何 Agent 框架**（LangGraph / OpenHands / AutoGPT 等）的 Agent Runtime。
主循环、工具注册与调用、输出解析、会话隔离、上下文压缩、执行轨迹全部手写，
只用了官方 LLM SDK 和标准库。

```
用户输入 → 装配上下文 → 调 LLM → 解析意图
    ├─ 最终答案 → 落库 → 返回用户
    └─ 工具调用 → Schema 校验 → 执行 → 结果回填 → 继续循环
```

## 亮点

- **协议无关的解析层**：原生 function calling 与文本 JSON 协议双通道，畸形输出可修复重试
- **会话真正隔离**：每个窗口独立上下文，进程重启后可原样恢复
- **上下文三层记忆**：原文 + 滚动摘要 + 结构化状态，超长自动压缩，追问不断片
- **异常不逃逸**：工具报错、参数非法、模型乱输出都回喂给模型自愈，主循环不崩
- **96 个离线测试**：`pytest` 直接跑，不需要 API Key

## 需求对照

| 题目要求 | 实现位置 | 验证用例 |
|---|---|---|
| 主循环：输入 → 决策 → 工具 → 回填 → 继续/返回 | `agent/runtime.py::chat` | `test_loop.py` |
| 解析思考过程 / 工具调用 / 最终答案 | `agent/parser.py` | `test_parser.py` |
| 工具注册（名称、描述、参数 Schema） | `agent/tools/base.py` | `test_registry.py` |
| 至少 3 个工具，todo 必做 | `agent/tools/`（calculator / search / todo / weather / read_docs） | `test_tools.py` |
| 多窗口 session 隔离、可恢复 | `agent/session.py` | `test_session.py`、`test_server.py` |
| 最大轮次限制、追问、上下文压缩 | `agent/runtime.py`、`agent/context.py` | `test_loop.py`、`test_context.py` |
| 异常处理、工具调用 trace | `agent/runtime.py`、`agent/trace.py` | `test_loop.py` |
| 测试用例 | `tests/` | 96 passed / 3 skipped |

## 快速开始

> 需要 Python 3.11+（代码用到 `X | None` 与 `removesuffix`）。

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 先跑测试（不需要密钥）

```bash
pytest
# 96 passed, 3 skipped
```

测试用 `FakeLLM` 脚本化回放模型响应，覆盖循环、解析、异常、会话隔离、追问与压缩。
不需要联网，也不会产生任何费用。

### 3. 配置模型

```bash
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY
```

代码里不存在任何密钥，只从环境变量读取；`.env` 已在 `.gitignore` 中。

### 4. 运行

```bash
# 命令行（推荐）
python cli.py

# 离线演示：不配密钥也能看到完整链路与 trace
python examples/demo_offline.py

# 双窗口演示（需要密钥）
python examples/demo_two_windows.py

# Web 演示
uvicorn server:app --reload --port 8000
```

CLI 常用命令：

| 命令 | 作用 |
|---|---|
| `/new [标题]` | 新建一个会话窗口 |
| `/sessions` | 列出当前用户的所有窗口 |
| `/use <序号\|id>` | 切换窗口，切回去可以接着聊 |
| `/history` | 查看当前窗口的全部原始消息 |
| `/trace [n]` | 查看最近 n 条执行轨迹 |
| `/todo` | 查看待办 |

## 目录结构

```
agent/
├── runtime.py      主循环：唯一的"转圈"逻辑，含终止条件与异常兜底
├── parser.py       输出解析：tool_calls / 文本 JSON / 修复 / 降级
├── llm.py          LLM 客户端抽象：DeepSeekLLM + FakeLLM（测试用）
├── context.py      上下文装配、token 估算、滚动摘要压缩
├── session.py      SQLite 存储：sessions / messages / todos
├── trace.py        JSONL 执行轨迹
└── tools/
    ├── base.py     注册表 + 手写 JSON Schema 校验 + 统一执行入口
    ├── calculator.py  AST 白名单求值，禁用 eval
    ├── search.py      本地语料检索（mock）
    ├── todo.py        待办增删查改
    ├── weather.py     天气（mock，按城市+日期稳定生成）
    └── read_docs.py   读取本地文档
cli.py              多窗口命令行
server.py           可选 Web 演示（FastAPI + 单页）
tests/              96 个离线用例
data/               mock 语料与文档
```

## 系统设计

### 1. 主循环与终止条件

`AgentRuntime.chat()` 是唯一知道"怎么转圈"的地方。单轮对话内的循环：

1. `ContextBuilder.build()` 装配上下文（必要时先压缩）
2. 调 LLM，拿到 `LLMResponse`
3. `parse_response()` 把输出收敛成三种意图之一
4. 工具调用 → Schema 校验 → 执行 → 结果作为 observation 回填 → 回到 1
5. 最终答案 → 落库 → 返回

终止条件有三类，都是显式的：

| 终止原因 | 触发条件 | 用户看到什么 |
|---|---|---|
| `final` | 模型给出最终答案 | 正常回复 |
| `max_steps` | 单轮工具步数超过 `AGENT_MAX_STEPS`（默认 8） | 说明已停止，提示拆小问题 |
| `loop_guard` | 相同 `(工具, 参数)` 连续调用 3 次 | 说明在重复调用，主动停下 |
| `error` | 模型不可用 / 输出两次无法解析 | 友好的失败说明 |

`loop_guard` 是为了防止模型陷入"调用—无进展—再调用"的死循环：第 2 次重复时先注入一条提示，
第 3 次直接打断。

### 2. 输出解析：为什么要两条协议

`parse_response()` 把任意模型输出收敛为 `Decision(kind=final | tool_calls | invalid)`：

- **原生 function calling**（主路径）：读 `tool_calls` 字段，逐个解析 `arguments` JSON
- **文本协议**（兼容路径）：从 ```json 代码块或正文里提取 `{"tool": ..., "arguments": ...}`，
  也支持 `{"action": "final", "answer": ...}` 和 Python 字面量风格（单引号、`True/False`）
- **容错**：先尝试标准 JSON，再试去尾逗号、截取第一个括号配平的对象、`ast.literal_eval`
- **修复**：解析失败时向模型追加一条纠正提示重试一次；仍失败则降级为友好提示

这样做的原因：不是所有模型都稳定支持 function calling；而且"解析逻辑"本身如果不自己写，
等于把决策权交给了框架。trace 里会保留 `protocol` 字段，能看出这次走的是哪条路。

### 3. 工具注册机制

```python
@tool(name="calculator", description="...", parameters={...JSON Schema...})
def calculator(args, ctx) -> str: ...
```

- 装饰器注册，`import agent.tools` 即完成注册
- 每个工具自带 名称 / 描述 / 参数 Schema，`openai_tools()` 自动转成 function calling 描述
- 执行前用**手写校验器**校验参数（type / required / enum / 范围 / 数组项 / 额外字段），
  报错信息是中文且可读，能原样回喂给模型修正
- 任何异常（含未知工具、参数错误、工具内部抛错）都收敛成 `ToolResult(ok=False)`，绝不向上抛

五个工具：`calculator`（AST 白名单，禁用 `eval`）、`search`（本地语料检索）、
`todo`（必做）、`weather`（mock）、`read_docs`（本地文档）。

### 4. Session 管理

三张表对应三种生命周期，这是"窗口隔离"可验证的根本原因：

| 表 | 归属 | 内容 |
|---|---|---|
| `sessions` | session 级 | 标题、滚动摘要、压缩水位线、结构化事实 |
| `messages` | session 级 | 全部原始消息（含已压缩的） |
| `todos` | **user 级**（可配置） | 待办 |

- 每个窗口一个 `session_id`，上下文互不可见
- `ensure_session(user_id, session_id)` 带归属校验：即使拿到别人的 `session_id`，
  也只会得到 `SessionAccessDenied`，不可能接管别人的窗口或读走历史
- `session_id` 走白名单校验（`[A-Za-z0-9_-]{1,64}`）。它会被拼进
  `data/traces/<id>.jsonl`，`../` 这类输入必须挡在入口，否则就是路径穿越
- 进程重启后 `get_session()` 从 SQLite 原样重建，包括摘要与压缩水位线
- 待办默认 user 级共享：**对话上下文隔离，用户资产共享**。设 `AGENT_TODO_SCOPE=session`
  可切换为严格隔离，两种语义都有测试覆盖

### 5. 上下文与记忆（重点）

#### 放置方式：context 里到底放了什么

| 顺序 | 内容 | 是否会被压缩 | 更新时机 |
|---|---|---|---|
| 1 | system：角色 + 工具使用规范 + 输出格式 | 否 | 启动时固定 |
| 2 | 历史对话摘要（滚动合并的短句） | 它本身就是压缩产物 | 触发压缩时重写 |
| 3 | 状态块：当前时间、未完成待办、**上次工具调用**、已知事实 | **永不压缩** | 每轮装配时重建 |
| 4 | 最近 N 轮原文（user / assistant / tool） | 否 | 每轮追加 |

原则是**稳定内容在前、易变内容在后**；结构化状态放在历史之前，让模型"先看事实、再读对话"。

刻意**不**放进上下文的东西：模型的历史思考过程（`reasoning`）。它只在当轮有用，
跨轮保留既费 token 又容易误导，所以只落库、不进上下文。

#### 召回时机：什么时候把记忆取出来

| 时机 | 触发者 | 取什么 |
|---|---|---|
| 每轮装配 | `ContextBuilder.build()` | system + 摘要 + 状态块 + 最近 N 轮 |
| 追问补全参数 | 模型读状态块的"上次工具调用" | 上一轮 `tool_call.name + arguments` |
| 压缩触发 | `maybe_compress()` | 早期轮次原文 → 摘要 |
| 进程重启 / 切换窗口 | `store.get_session()` | 从 SQLite 重建全部状态 |

#### 两类追问是怎么成立的

- **纯对话追问**（"我叫什么"）：最近 N 轮原文还在上下文里，直接可见
- **带工具的追问**（"北京天气" → "那上海呢"）：关键不是记住对话，而是把上一轮的
  `tool_call` 结构化存成 `facts.last_tool_call`（工具名 + 参数 + 结果摘要），
  每轮注入状态块。模型因此知道"上次查的是 weather(city=北京)"，
  才能把"那上海呢"补全成 `weather(city=上海)`

#### 压缩策略

- **触发**：估算 token 超过 `max_context_tokens × compress_threshold`（默认 12000 × 0.7）
- **切分**：按"轮"切（以 user 消息为界），保证 `assistant.tool_calls` 与 `tool` 结果不被拆开
- **保留**：最近 `keep_recent_turns` 轮原文不动，更早的交给 LLM 压成 400 字以内的摘要
- **落库**：推进 `compressed_upto` 水位线；原始消息**不删除**，只是不再进入上下文（可审计、可回放）
- **兜底**：压缩用的 LLM 调用失败时，退化为机械截断摘要，主流程不中断

token 估算用中英文分权（中文 1 字 ≈ 1 token，英文 4 字符 ≈ 1 token），比 tiktoken 便宜且够用。

### 6. 异常处理

| 场景 | 处理方式 |
|---|---|
| 模型超时 / 限流 | 指数退避重试 2 次，仍失败则返回友好提示（`stopped_reason=error`） |
| 输出无法解析 | 追加纠正提示重试一次，二次失败才放弃 |
| 工具不存在 / 参数非法 | 校验失败信息作为 observation 回喂，模型自行修正 |
| 工具内部抛异常 | 捕获并转成 `ToolResult(ok=False)`，循环继续 |
| 工具输出过长 | 入库前按 `AGENT_MAX_TOOL_OUTPUT` 截断，防止一次爆上下文 |
| 死循环 | 重复调用检测（按 `(工具, 参数)` 指纹累计，并行调用也有效）+ `max_steps` 双保险 |
| 跨用户传 session_id | 归属校验拦截，返回 `SessionAccessDenied`，不泄露也不污染原会话 |
| 非法 session_id | 白名单校验拦截（同时防住 trace 文件的路径穿越） |
| 压缩失败 | 机械摘要兜底 |
| trace 写失败 | 静默忽略，绝不影响主流程 |

### 7. 执行轨迹

每次对话写一份 `data/traces/<session_id>.jsonl`，一行一个事件：

```
user_input → llm_request → llm_response → parse → tool_call → tool_result → ... → final
```

每条记录带 `turn` / `step` / 时间戳 / 耗时 / token 用量。`/trace` 命令或 Web 的
"执行轨迹"面板可以直接查看，排障和复盘都用它。

## 测试

```bash
pytest                      # 全部离线，不需要密钥
pytest tests/test_loop.py   # 只跑主循环
AGENT_LIVE_TEST=1 pytest tests/test_live.py   # 真实模型的端到端用例（默认跳过）
```

| 文件 | 覆盖内容 |
|---|---|
| `test_parser.py` | 原生 tool_calls、文本 JSON、思考标签、畸形 JSON、Python 字面量、空响应 |
| `test_registry.py` | 工具注册、Schema 生成、类型/枚举/范围/额外字段校验、异常收敛 |
| `test_tools.py` | 五个工具的正常与异常路径，含代码注入拦截 |
| `test_loop.py` | 直接回答、单工具、多步链、并行工具、max_steps、修复重试、工具报错自愈、死循环打断、LLM 故障 |
| `test_session.py` | 双窗口隔离、重启恢复、待办共享/隔离、跨用户不可见/不可接管、非法 id 拦截 |
| `test_context.py` | 状态块注入、纯对话追问、工具追问、压缩触发、压缩后追问、摘要注入、压缩失败兜底 |
| `test_server.py` | Web 路由的会话鉴权：跨用户 403、路径穿越 404、正常对话 |

## 设计取舍

**为什么 todo 默认 user 级而不是 session 级。**
题目说"两个窗口互不影响"，这句话可以有两种读法。我选择：对话上下文严格隔离，
待办作为用户资产跨窗口共享——这符合真实产品直觉（换台设备打开待办还在）。
但语义分歧不该由实现偷偷决定，所以做成 `AGENT_TODO_SCOPE` 可配置，两种行为都有测试。

**为什么状态块放在历史之前。**
严格说，把易变的状态块放在最前面会破坏前缀缓存。我选择"事实优先"：
让模型先建立当前处境（有哪些待办、上次查了什么），再读对话，追问的准确率更高。
在 12k 上下文规模下，缓存收益不如准确率重要。

**为什么不用向量检索。**
题目只要求 mock 的 search。这里用关键词加权（标题 6 / 标签 3 / 正文 2，中文按 2-gram）
已经足够演示"先检索再读全文"的多步链路，且完全可解释、零依赖。

## 已知限制

- `search` / `weather` 是 mock 数据，不联网
- 压缩只做单层摘要，没有做分层摘要或向量化长期记忆
- SQLite 单机存储，未针对高并发写入优化
- 没有做流式输出（SSE）
- 单轮内不做跨工具的并行执行（多工具是顺序执行的）

## 安全

- 代码中不存在任何 API Key，只从环境变量读取
- `.env` 在 `.gitignore` 中，仓库里只有 `.env.example` 占位符
- `calculator` 用 AST 白名单求值，拒绝 `__import__`、属性访问等任意代码执行
- 工具参数在执行前统一校验，模型无法越权传入额外字段

完整的 12 条开发提示词原文、Agent system prompt 的四版演进、AI 生成代码的 7 处人工修正，
以及 20 条问题复盘见 [docs/AI_PROMPTS.md](docs/AI_PROMPTS.md)。
