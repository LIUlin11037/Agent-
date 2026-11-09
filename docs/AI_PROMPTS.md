# AI Prompt 与问题解决记录

这份文档记录两件事：**怎么用 AI 辅助开发**，以及**开发过程中真实遇到的问题和解法**。
不是事后编的漂亮总结，都是实际踩过的坑。

## 一、AI 在开发中的角色

原则：**先定契约，再让 AI 写实现**。架构、数据结构、终止条件、异常策略由人决定，
AI 负责把边界情况补全、把重复代码写快、把测试用例铺开。核心 Runtime 的每一行都能讲清楚为什么这么写。

### Prompt 1：需求反推验收点

```
把下面这段需求拆成可验收的检查项，重点标出「写得不清楚但会扣分」的地方：
[粘贴题目全文]
输出格式：模块 / 必须交付 / 可能的理解分歧 / 容易被忽略的细节
```

用途：识别出题目里 `todo` 归属的歧义、"解析逻辑"到底是解析什么、测试能不能离线跑这些关键点。
这一步产出的检查项直接变成了后来的测试矩阵。

### Prompt 2：解析器的边界用例穷举

```
我在实现一个把 LLM 输出解析成「工具调用 / 最终答案 / 无法解析」三态的解析器。
支持原生 tool_calls 和文本 JSON 两条协议。
列出所有可能让解析失败的输入形态，按「真实模型会输出的概率」排序。
```

用途：补齐了尾逗号、单引号字面量、代码块前后有解释文字、arguments 是字符串而不是对象、
思考标签未闭合等用例，最后都进了 `test_parser.py`。

### Prompt 3：压缩策略的取舍

```
上下文压缩有两种做法：
A. 滑动窗口直接截断最早的 N 轮
B. 把早期轮次交给 LLM 压成摘要
在「用户会追问」这个前提下，分别会怎么失败？给出混合方案的切分粒度建议。
```

用途：确认了"按轮切分"而不是"按消息切分"，否则会把 `assistant.tool_calls` 和对应的
`tool` 结果拆散，导致上下文非法。这条直接决定了 `group_turns()` 的实现。

### Prompt 4：测试用例设计

```
这是一个自实现的 Agent Runtime，工具和 LLM 都是可注入的。
设计一套完全离线、不依赖 API Key 的测试，要求覆盖：
循环终止条件、解析失败、工具异常、会话隔离、上下文压缩。
每个用例给出「构造方式 + 断言什么」。
```

用途：产生了 `FakeLLM` 脚本化回放的思路，以及"用断言检查第二次请求的 messages 里
是否包含 tool 消息"这类验证方式——比断言最终文本更可靠。

### 系统提示词（Agent 的 system prompt）

`agent/context.py` 里的 `SYSTEM_PROMPT` 是产品的一部分，设计上有几个刻意选择：

- **明确"不要为了调用工具而调用工具"**：不加这句，模型容易对"1+1"也要走一遍 calculator
- **把工具选择规则写成可判定的条件**（"查制度先用 search，要全文再 read_docs"），
  而不是笼统的"善用工具"
- **显式要求参考"上次工具调用"**：这是"那上海呢"能被正确补全的前提
- **声明"绝不编造工具结果"**：工具失败时模型倾向于自己编一个合理答案，必须显式禁止
- **文本协议格式写死在 prompt 里**：保证不支持 function calling 的模型也能稳定输出可解析的结构

## 二、问题解决记录

### 1. 模型陷入重复调用死循环

**现象**：工具返回"没有找到"，模型原样再调一次同样的参数，反复循环直到步数耗尽。

**原因**：只靠 `max_steps` 兜底，用户体验是"转了 8 步然后说失败了"。

**解法**：加重复调用检测。用 `(工具名, 排序后的参数 JSON)` 做指纹：
第 2 次重复时注入一条提示（"结果不会变化，请基于已有结果作答"），
第 3 次直接打断并返回 `stopped_reason=loop_guard`。

**验证**：`test_loop.py::test_loop_guard_breaks_repeat`。

### 2. `str.format` 与 JSON 花括号冲突

**现象**：解析失败触发修复提示时抛 `KeyError: '"tool"'`，整个对话崩掉。

**原因**：修复提示模板里写了 `{"tool": "工具名", "arguments": {...}}`，
`str.format()` 把 `{"tool"` 当成了占位符。

**解法**：模板里的字面花括号转义成 `{{` `}}`。

**教训**：异常处理路径本身也要有测试——这个 bug 只有在"模型输出无法解析"时才会触发，
正是最容易被漏测的分支。

### 3. 文档检索命中的是标题而不是正文

**现象**：用 `query="字数"` 检索周报文档，返回的段落只有 `## 字数` 这一行标题。

**原因**：按空行切段时，Markdown 标题是独立段落，先命中了标题就停了。

**解法**：这是 mock 检索的固有粗糙度，没有过度设计；测试改为断言真实能命中的关键词，
并在 `read_docs` 里保留"未命中则返回开头部分"的兜底。

### 4. 压缩把工具调用链拆断了

**现象**：压缩后偶发上下文非法（assistant 有 tool_calls 但没有对应的 tool 结果）。

**原因**：最初按"消息条数"截断，正好切在 assistant 和 tool 之间。

**解法**：`group_turns()` 以 `user` 消息为界切轮，一轮内部的 assistant / tool / assistant
整体保留或整体压缩。压缩水位线记录到"最后一轮的最后一条消息的 seq"。

**验证**：`test_context.py::test_group_turns_keeps_tool_pairs`。

### 5. 假模型被压缩调用打乱脚本

**现象**：压缩测试里 `FakeLLM` 的脚本顺序错位，断言全挂。

**原因**：压缩是在装配上下文时同步发生的，会插入到正常对话调用之间，消耗掉一个脚本项。

**解法**：写 `SummaryLLM`，先识别"system 消息是不是压缩提示"，是则返回固定摘要，
否则走脚本回放。压缩调用单独计数，用来断言"确实压过"。

### 6. 文本协议下不能伪造 tool 消息

**现象**：走文本 JSON 协议的模型，如果按原生格式回填 `role="tool"` 消息，部分模型会报错。

**原因**：文本协议本质上没有 `tool_call_id`，模型也不认识 `tool` 角色。

**解法**：`_append_tool_observation()` 按协议分流——原生协议回填 `role="tool"`，
文本协议回填 `role="user"` 的 `[工具 xxx 返回] ...` 文本块。

### 7. 中文 token 估算

**现象**：用"字符数 / 4"估算，中文场景严重低估，压缩从不触发。

**解法**：中英文分权估算——中文按 1 字 ≈ 1 token，其余按 4 字符 ≈ 1 token。
够用且零依赖，不需要引入 tiktoken。

### 8. Windows 下 `pip` 与 `python` 不是同一个解释器

**现象**：`pip install pytest` 成功，但 `python -m pytest` 报模块不存在。

**原因**：`pip` 指向 F:\Python3.10，而 PATH 里的 `python` 是另一个 3.13 环境。

**解法**：统一用 `python -m pip install ...`，保证装到当前解释器。

### 9. 换个用户传同一个 session_id，就能接管别人的窗口

**现象**：`ensure_session(user_id, session_id)` 只按 id 查会话，不校验归属。
`runtime.chat("user_b", "…", session_id="win1")` 会直接接着 `user_a` 的窗口聊，
`history("win1")` 也能读走全部历史。

**原因**：把"窗口隔离"理解成了"每个 id 一份上下文"，漏掉了 id 是用户可控输入这一点。

**解法**：`ensure_session` / `get_session` 增加 `user_id` 参数，归属不符抛
`SessionAccessDenied`；Web 层映射成 403，CLI 的 `/history` 也带上当前用户。

**验证**：`test_session.py::test_other_user_cannot_continue_session`（同时断言原会话没被污染）、
`test_server.py::test_messages_endpoint_rejects_other_user`。

### 10. session_id 拼进 trace 文件名 = 路径穿越

**现象**：`Tracer` 用 `trace_dir / f"{session_id}.jsonl"` 建文件。
传 `session_id="../../server"` 就能把日志写到仓库外面；Web 的 trace 接口反过来能读任意 `.jsonl`。

**原因**：id 来自请求参数，却被当成可信的文件名片段。

**解法**：入口统一白名单 `^[A-Za-z0-9_-]{1,64}$`（`validate_session_id`），
trace 接口先校验归属再拼路径。

**验证**：`test_session.py::test_invalid_session_id_rejected`、
`test_server.py::test_trace_endpoint_rejects_path_traversal`。

### 11. 并行工具调用让死循环检测失效

**现象**：模型一步并行调 `weather(北京)` + `weather(上海)`，即使每一步都完全重复，
`loop_guard` 也从不触发，只能等 `max_steps` 兜底。

**原因**：重复计数用的是"上一个 key"，第二个 key 每次都把它重置成 1。

**解法**：改成 `dict[fingerprint] -> count`，一步结束后取最大计数判断；
提示只对同一个指纹发一次。

**验证**：`test_loop.py::test_loop_guard_handles_parallel_calls`。

### 12. 文本协议只认 `{"tool": ...}`，不认 ReAct 风格

**现象**：部分模型输出 `{"thought": "...", "action": "weather", "action_input": {...}}`，
解析器只找 `tool/tool_name/name`，于是整段 JSON 被当成最终答案回给用户。

**解法**：`action` 不在 `final/answer/respond/tool/call/use` 里、且带参数键时，
把 `action` 当工具名；`action_input` 加入参数键候选。同时保留一条边界：
只有 `action` 没有参数时，不瞎猜成工具调用。

**验证**：`test_parser.py::test_text_protocol_react_style_action` /
`test_text_protocol_action_without_arguments_is_final`。

## 三、如果继续做

- 把 `search` 换成真实检索（BM25 + 向量混合），验证长文档下的上下文预算
- 压缩升级为分层摘要（会话级 + 主题级），支持跨 session 的长期记忆
- 增加 SSE 流式输出与工具执行进度反馈
- 用真实的失败 trace 做回归集，把线上问题固化成测试用例
