# Zhuge Agent Demo

诸葛 AI 智能体架构 demo，用于验证“自然语言输入 -> 主智能体路由 -> 子智能体选择 Skill -> 确定性工具执行 -> 证据链 -> LLM 组织回答 -> 结果校验”的平台设计。

项目里保留了早期设计资料和 HTML 原型，当前可运行代码位于 `zhuge_agent/`。

## 核心目标

本项目不是一个简单的 Prompt demo，而是一个可扩展的 Agent / Skill / Tool 运行框架雏形：

1. 业务人员用自然语言提问。
2. MasterAgent 判断整体执行模式：普通寒暄或通用问答直接回答，简单业务问题委派一个子智能体，复合独立任务并行委派多个子智能体，有依赖或人审要求的任务进入 graph workflow。
3. 复合问题可以拆成多个子智能体并行处理，复杂 pipeline 可以通过 workflow JSON 表达串行、并行、汇总和人审节点。
4. 子智能体从“公共 Skills + 自己的私有 Skills”中选择具体 Skill。
5. 子智能体根据自己的 prompt 模板规划并执行具体 Skill。
6. Runtime 按步骤调用确定性工具、模型、规则、知识库或数仓。
7. Evidence Builder 收集证据链。
8. Verifier 校验权限、口径、结构化结果和 `no_llm` 边界。
9. 子智能体的 Composer 调用 LLM 生成可读回答。
10. Post-check 校验回答中的关键数字是否来自上游工具输出。

## 当前支持的 Demo

### 1. 工业 DCS 查数

示例问题：

```text
现在 2# 反应釜的温度、压力和冷却水流量分别是多少？
```

对应 Skill：

```text
zhuge_agent/skills/industrial/query_tag_value/
```

执行链路：

```text
A1 MasterAgent / Router
-> A2 SubAgent / Skill Router
-> B Context Resolver
-> C Planner
-> D1 DCS 当前位号值
-> E Evidence Builder
-> F Verifier
-> G Composer
-> H Post-check
```

### 2. 工业温度归因诊断

示例问题：

```text
2# 反应釜最近 3 天温度一直偏高，帮我看看是什么原因，应该怎么调？
```

对应 Skill：

```text
zhuge_agent/skills/industrial/reactor_temp_forecast_adjust/
```

执行链路包含 DCS 历史数据、趋势预测模型、规则引擎、知识库案例检索和后验校验。

### 3. 企业经营指标分析

示例问题：

```text
本月华东区域销售额、环比多少，按城市排序
```

对应 Skill：

```text
zhuge_agent/skills/business/business_metric_analysis/
```

执行链路：

```text
A1 MasterAgent / Router
-> A2 SubAgent / Skill Router
-> B Context Resolver
-> C Planner
-> D1 企业指标语义库
-> D2 查询计划生成器
-> D3 权限 / 口径校验
-> D4 数仓 / 业务库计算
-> D5 可视化结果生成器
-> D6 原因洞察
-> E Evidence Builder
-> F Verifier
-> G Composer
-> H Post-check
```

这条链路强调企业级指标问数不是简单 NL2SQL，而是先绑定统一业务口径，再生成可校验查询计划，权限和口径校验通过后才执行计算。

### 4. Graph Workflow 编排

示例问题：

```text
本月华东区域销售额、环比多少？同时查询现在 2# 反应釜温度和压力是多少？
```

对应 Workflow：

```text
zhuge_agent/workflows/business_metric_dcs_workflow.json
```

执行图：

```text
经营指标分析节点 ─┐
                 ├─> 人审检查点 -> 汇总输出节点
DCS 当前值节点 ───┘
```

Workflow 层借鉴 ADK 的图编排思想，但不依赖 Google Cloud 或 ADK 包。它负责声明 DAG、并发节点和检查点；每个 agent 节点内部仍复用当前子智能体、Skill、Tool、Verifier 和 Post-check 链路。

## 目录结构

```text
.
├── zhuge_agent/
│   ├── __init__.py
│   ├── llm.py
│   ├── runtime.py
│   ├── schema.py
│   ├── agent_loader.py
│   ├── skill_loader.py
│   ├── workflow_loader.py
│   ├── tools/
│   │   ├── base.py
│   │   ├── registry.py
│   │   └── demo/
│   │       ├── industrial.py
│   │       └── business.py
│   ├── utils/
│   │   ├── config.py
│   │   ├── logging_utils.py
│   │   ├── prompt_templates.py
│   │   └── text.py
│   ├── agents/
│   ├── workflows/
│   └── skills/
│       ├── industrial/
│       │   ├── query_tag_value/
│       │   └── reactor_temp_forecast_adjust/
│       └── business/
│           └── business_metric_analysis/
├── skills/
│   └── README.md
├── tests/
│   ├── cases/
│   └── test_runtime.py
├── zhuge_agent_demo.py
├── config.ini              # 本地配置，已忽略提交
├── config.example.ini      # 可提交的配置示例
└── draft/
```

说明：

- `zhuge_agent/`：当前主代码包。
- `zhuge_agent/utils/`：包内公共基础设施，包括配置加载、日志上下文、Prompt 模板和文本匹配工具；项目代码统一从该目录导入公共工具。
- `zhuge_agent/agents/`：子智能体配置，包含角色、专属 `skill_ids` 和 Jinja2 prompt 模板路径。
- `zhuge_agent/workflows/`：graph-based workflow 配置，声明跨智能体 DAG、并行节点、人审检查点和汇总节点。
- `zhuge_agent/skills/`：平台内置 Skills，随代码包发布。
- `skills/`：外部扩展 Skills 目录，预留给客户或项目定制能力。
- `tests/cases/`：demo 输入案例和测试夹具。
- `draft/`：废弃设计草稿，仅保留为历史资料，不作为当前实现入口。

## Skill 目录规范

每个 Skill 是一个独立文件夹，参考 Anthropic Skills 的组织方式：

```text
some_skill/
├── SKILL.md
└── skill.json
```

`SKILL.md` 面向 LLM 和开发者，包含能力说明、适用问题和执行原则。

`skill.json` 面向 Runtime，描述可执行图：

```json
{
  "id": "query_tag_value",
  "name": "DCS 位号查询",
  "description": "查询工业装置当前或最近时刻的温度、压力、流量等位号数值。",
  "mode": "问数",
  "result_type": "tag_query",
  "visibility": "public",
  "allowed_agents": [],
  "intent_examples": ["现在 2# 反应釜温度是多少"],
  "composer_template": "数值清单 + 时间 + 数据来源 + 数据质量",
  "steps": [
    {
      "id": "D1",
      "name": "Executor / DCS Current Values",
      "kind": "data",
      "tool_id": "dcs.current_values",
      "input_keys": ["context"],
      "output_key": "current_values",
      "no_llm": true
    }
  ]
}
```

Runtime 会递归加载：

1. `zhuge_agent/skills/`
2. `ZHUGE_SKILLS_DIR` 指定的目录
3. 项目根目录 `skills/`

外部 Skill 的 `id` 不能和内置 Skill 重复。

Skill 可见性规则：

- `visibility: "public"`：公共 Skill，对所有子智能体可见。
- `visibility: "private"`：私有 Skill，只对在子智能体 `skill_ids` 中显式声明它的子智能体可见。
- `allowed_agents`：兼容字段，外部旧版 Skill 可临时使用；内置 Skill 不依赖它表达归属。

子智能体定义位于 `zhuge_agent/agents/*.json`：

```json
{
  "id": "business_metric_agent",
  "name": "经营指标分析智能体",
  "role": "business_metric_analyst",
  "description": "负责企业经营指标语义解析、口径校验、查询计划、数仓计算、可视化和洞察组织。",
  "skill_ids": ["business_metric_analysis"],
  "prompt_template": "templates/business_metric_agent.md.j2"
}
```

`prompt_template` 指向一个 Markdown / Jinja2 模板文件。模板文件按注释段落声明多个系统提示词：

```text
<!-- skill_router_system -->
子智能体选择可见 Skill 的提示词。

<!-- planner_system -->
子智能体规划提示词。

<!-- composer_system -->
子智能体回答生成提示词。
```

Runtime 构建子智能体时加载模板文件，但使用 `LazyPrompt` 延迟渲染；只有 LLM 客户端真正消费 system prompt 时，Jinja2 模板才会 render。

## 代码架构

### `schema.py`

定义核心数据结构：

- `Skill`
- `SkillStep`
- `ToolSpec`
- `TraceEvent`
- `Evidence`
- `StepResult`
- `StructuredAnswer`
- `ParallelStructuredAnswer`

这些结构是前端展示、测试断言和后续 API 输出的共同契约。

### `skill_loader.py`

递归加载 Skill 文件夹：

- 读取 `SKILL.md` frontmatter 和正文说明。
- 读取 `skill.json` 执行图。
- 转换成 Runtime 使用的 `Skill` 对象。

### `agent_loader.py`

递归加载子智能体配置：

- 读取 `zhuge_agent/agents/*.json`
- 每个子智能体声明自己的私有 `skill_ids`
- 每个子智能体声明一个 `prompt_template` Markdown / Jinja2 文件
- 公共 Skill 自动对所有子智能体可见

### `llm.py`

封装 LLM 调用：

- `OpenAICompatibleLLMClient`：真实 LLM API 客户端。
- `FixtureLLMClient`：离线测试和 demo 用的确定性客户端。

LLM 通过项目根目录的 `config.ini` 配置，无需设置环境变量：

```ini
[llm]
provider = openai
api_key =
base_url = https://api.openai.com/v1
model = gpt-4o-mini
timeout_seconds = 60
transport = urllib
verify_ssl = true
extra_body =
```

在本地 `config.ini` 的 `api_key` 填写自己的密钥；其他 OpenAI 兼容服务可修改 `base_url` 和 `model`。`base_url` 填 API 根地址，不需要包含 `/chat/completions`。

LiteLLM 或企业内网网关如果需要使用 OpenAI SDK 和自定义 `httpx.Client`，可以改为：

```ini
[llm]
provider = openai
api_key = 本地密钥
base_url = https://litellm.mybigai.ac.cn/
model = 网关中注册的模型名
timeout_seconds = 60
transport = openai_sdk
verify_ssl = false
```

`transport = openai_sdk` 会按 `openai.OpenAI(base_url=..., api_key=..., http_client=httpx.Client(...))` 的方式调用；`verify_ssl = false` 只建议用于内网或自签证书环境。

OpenAI 标准 Chat Completions 参数之外的模型或网关扩展参数可写入 `extra_body`，值必须是 JSON 对象。urllib transport 会把这些字段合并到请求体；OpenAI SDK transport 会通过 SDK 的 `extra_body` 参数传入。比如 Qwen3 / vLLM 关闭 thinking：

```ini
[llm]
provider = openai
api_key = 本地密钥
base_url = http://localhost:8000/v1
model = qwen3-8b
transport = openai_sdk
extra_body = {"chat_template_kwargs": {"enable_thinking": false}}
```

`provider = openai` 调用真实 API，配置缺失或无效时直接报错，不自动切换为模拟响应。`provider = fixture` 明确选择离线演示，不需要密钥。启动时会显示实际使用的模式和真实模式的模型名称，不输出密钥。

`config.ini` 已加入 `.gitignore`，只提交不含真实密钥的 `config.example.ini`。仓库新检出后可参考示例创建本地 `config.ini`。LLM 配置由 `zhuge_agent/utils/config.py` 使用标准库读取和校验，支持 Python 3.10。

代码调用可使用 `ZhugeAgentRuntime(config_path="其他配置.ini")`。默认路径相对于项目位置确定，不受运行目录影响；显式传入的相对路径以当前工作目录为基准。测试注入 `FixtureLLMClient` 时不读取本地配置。

### 日志

`zhuge_agent/utils/logging_utils.py` 统一配置 Python 标准库 Logger。控制台日志写入 stderr，Demo 的业务展示仍在 stdout；本地配置默认同时写入 `logs/zhuge_agent.log`，日志目录已加入 `.gitignore`。

```ini
[logging]
level = INFO
file = logs/zhuge_agent.log
```

`file` 留空只输出控制台；相对路径以配置文件所在目录为基准。没有 `[logging]` 段时默认 INFO 和控制台输出。DEBUG 额外记录模板加载、解析和注册表查询，WARNING/ERROR 可减少日常输出。

每条日志格式为：

```text
agent_id session_id YYYY-MM-DD HH:MM:SS.mmm LEVEL module@lineno - message
```

主路由使用 `MasterAgent`，子任务使用配置中的子智能体 ID。未进入请求的启动日志使用 `-` 占位。调用方可通过 `answer(..., session_id="session-001")`、`answer_parallel(...)`、`iter_answer(...)`、`iter_answer_parallel(...)` 及对应字典入口传入会话 ID，未提供时自动生成 UUID；并行和流式线程共享本次会话并隔离各自智能体身份，退出或异常后恢复上层上下文。

关键函数记录输入、输出摘要、耗时及异常，包括路由、规划、工具调用、证据、校验、回答生成和配置加载。`@log_call("参数名", ...)` 提供统一记录方式，日志的模块和行号指向被记录函数定义；普通 `logger.info(...)` 指向实际调用位置。API 密钥、Authorization、密码等字段及已登记密钥会脱敏，长列表和文本会截取摘要并记录数量，不修改实际结果。记录提示词对象时只显示类型，不会提前消费 `LazyPrompt`。

嵌入其他应用时可调用 `configure_logging(level, log_file=...)` 和 `log_context(agent_id=..., session_id=...)`。配置仅替换本项目创建的 Handler，宿主已有的 Handler 由宿主应用管理。日志级别和文件路径属于全局设置，应在启动时配置。

### `tools/`

实现确定性工具层。`base.py` 定义 Tool 接口，`registry.py` 管理工具注册；当前内置的工业和经营分析能力放在 `demo/industrial.py`、`demo/business.py` 中，明确表示它们是为了离线 demo 和测试闭环准备的示例实现。

当前 demo 工具覆盖：

- DCS 历史数据读取
- DCS 当前位号值读取
- 趋势预测模型
- SOP 规则引擎
- 知识库案例检索
- 企业指标语义库
- 查询计划生成
- 权限 / 口径校验
- 数仓计算
- 可视化配置生成
- 原因洞察生成

### `runtime.py`

实现 Agent 生命周期：

```text
MasterAgent / Router
SubAgent / Skill Router
B Context Resolver
SubAgent / C Planner
D Executor
E Evidence Builder
F Verifier
SubAgent / G Composer
H Post-check
```

其中 MasterAgent 接收用户的一般性问题，只负责选择子智能体；具体业务能力由子智能体执行。MasterAgent 每次路由时会从 `AgentRegistry` 读取当前已注册子智能体，并把 `agent_id`、名称、角色、功能描述和私有 Skill 列表作为能力清单交给 LLM 判断。子智能体从公共 Skills 和自己的私有 Skills 中选择具体 Skill，并使用 `zhuge_agent/agents/templates/*.md.j2` 中的 prompt 模板执行规划和回答生成。

MasterAgent 的兜底策略是“宁可说明能力边界，也不误触发业务链路”，并区分两类情况：

- `invalid_route`：LLM 返回了未注册的 `agent_id`、非法 `agent_ids` 或无效 `workflow_id`。Runtime 会把合法能力清单、上一次错误响应和非法目标重新发给 LLM 重试一次；如果重试后仍非法，返回 `skill_id=direct_answer`，说明路由非法并列出当前可用子智能体。
- `capability_gap`：LLM 判断当前没有子智能体可以可靠承接用户诉求。Runtime 不再强行词匹配到某个业务 Agent，而是返回 `skill_id=direct_answer`，说明现有子智能体能力不足，并列出当前可用子智能体及功能简介。

`AgentRegistry.route()` 保留为注册表的简单文本匹配能力，但主路由链路不会在无有效候选时强行把问题塞给某个子智能体。

Runtime 提供以下入口：

- `answer_auto(request, progress_callback=...)`：由 MasterAgent 先判断 `direct / single / parallel / workflow`，再自动调用对应入口，是产品层默认推荐入口。
- `answer(request, progress_callback=...)`：单子智能体链路，保持向后兼容；传入回调时会流式输出 `AgentProgressEvent`。
- `iter_answer(request)`：普通问答的流式入口，先输出主智能体和子智能体进度事件，再输出最终回答的 `AnswerTokenEvent`，最后输出 `StructuredAnswer`。
- 对于“你好”“你好吗”等无需工具的问题，`iter_answer(request)` 只输出 MasterAgent 路由进度，并返回 `skill_id=direct_answer` 的直接回答。
- `answer_parallel(request)`：MasterAgent 选择多个子智能体后并行执行，各子智能体独立完成 Skill Router、Planner、Executor、Verifier、Composer 和 Post-check，最后返回 `ParallelStructuredAnswer`。
- `iter_answer_parallel(request)`：流式输出 `AgentProgressEvent` 和各子智能体最终回答的 `AnswerTokenEvent`，最后输出 `ParallelStructuredAnswer`，便于前端、SSE 或 WebSocket 展示子智能体工作进度。
- `answer_workflow(request, workflow_id=...)`：执行 graph-based workflow。当前节点类型支持 `agent`、`human_review` 和 `join`；同一 DAG 层级的节点会并发执行，返回 `WorkflowStructuredAnswer`。
- `iter_answer_workflow(request, workflow_id=...)`：流式输出 workflow 节点进度、子智能体进度和最终回答 token，最后返回 `WorkflowStructuredAnswer`。
- `answer_request(AgentRunRequest(...))`：面向产品原型或后端服务的结构化入口，可携带 `session_id`、`tenant_id`、`user_id`、`role`、`data_scope` 和 `context`。默认 `mode=auto` 时由 MasterAgent 自动选择 `direct / single / parallel / workflow`；显式传入 `single|parallel|workflow` 时直接使用指定入口。
- `iter_run_events(AgentRunRequest(...))`：统一事件流入口，输出 `RuntimeEvent(type=progress|token|final|error)`，便于后端直接映射为 SSE、WebSocket 或消息队列事件。

Router、Planner、Composer 可以调用 LLM；数据读取、计算、规则、权限和校验默认使用确定性工具。
Router 和 Planner 仍使用结构化 JSON 完整返回，便于路由、权限和校验；token 级流式只用于 MasterAgent 直接回答和子智能体 Composer 的最终自然语言文案。

### 产品/后端接入边界

当前仓库定位为智能体运行内核，不包含完整后端平台。服务化、持久化、账号权限、发布部署、观测审计和人工审批闭环建议由后端架构层实现；本内核提供稳定的 Agent/Skill/Tool/Workflow 编排能力，以及结构化请求、进度事件、token 事件、最终结果和错误协议。

后端接入时建议把已鉴权的业务上下文写入 `AgentRunRequest.context` 或 `data_scope`，Runtime 会透传给 Context Resolver 和确定性工具；真正的权限/口径/数据访问判断仍应落在后端服务或 Tool Adapter 中，不应写死到 `runtime.py`。

长对话记忆遵循同样边界：后端负责保存完整 chat history、做权限隔离、脱敏、摘要和检索；Runtime 通过 `AgentRunRequest.recent_messages`、`conversation_summary` 和 `retrieved_memories` 消费已筛选上下文，并统一放入 `context["conversation"]`。运行结果中的 `memory_write_candidates` 只表示建议写入的记忆候选，是否持久化仍由后端裁决。

### Agent Memory 设计理念

Agent Memory 不应让 Runtime 直接变成聊天后端。长对话记忆牵涉会话持久化、租户隔离、用户权限、敏感信息脱敏、审计、删除权和 token budget 控制，这些更适合由后端平台或独立 Memory Service 负责。当前 Runtime 的职责是定义稳定协议、消费后端筛选后的上下文，并在未来需要时输出可写入记忆的候选。

推荐分工：

- 后端负责保存完整 chat history、生成会话摘要、检索用户/租户/业务域记忆、做权限过滤和脱敏。
- 后端把本轮需要消费的上下文写入 `AgentRunRequest.recent_messages`、`conversation_summary`、`retrieved_memories` 或 `context`。
- Runtime 将这些内容规范化到 `context["conversation"]`，供 Context Resolver、Planner、Tool 和 Composer 消费。
- Agent 可以在结果中返回 `memory_write_candidates`，表达“建议写入长期记忆”的结构化候选。
- 后端根据权限、策略、置信度和人工确认决定是否真正写入数据库、向量库或会话存储。

示例：

```python
AgentRunRequest(
    request="继续分析刚才那个反应釜问题",
    session_id="s-001",
    user_id="u-001",
    tenant_id="t-001",
    recent_messages=[
        {"role": "user", "content": "刚才我们在看 2# 反应釜。"},
        {"role": "assistant", "content": "已记录当前分析对象为 2# 反应釜。"},
    ],
    conversation_summary="用户正在连续追问 2# 反应釜的运行状态。",
    retrieved_memories=[
        MemoryItem(
            type="user_preference",
            content="用户偏好先看结论，再看证据。",
            scope="user",
            confidence=0.91,
        )
    ],
)
```

当前实现已经支持 memory 输入协议和空的 `memory_write_candidates` 输出字段，但还没有实现真实 Memory Store。后续如果要扩展，应优先增加 Memory Tool 或 Memory Adapter，而不是把存储和检索逻辑写进 `runtime.py`。

### RAG 设计理念

当前框架把 RAG 看作“可证据化的知识检索工具链”，而不是让 LLM 自己在 prompt 里隐式完成检索。RAG 应作为 Skill 执行图中的确定性 Tool 出现，检索结果进入结构化 state 和 `Evidence`，再由 Evidence Builder 汇总，最后交给 Composer 组织回答。

已有雏形是工业诊断 Skill 中的知识库案例检索：

```text
D4 Executor / Knowledge Search
kind = knowledge
tool_id = knowledge.case_search
no_llm = true
```

对应 demo 工具位于 `zhuge_agent/tools/demo/industrial.py` 的 `KnowledgeSearchTool`。它当前只返回案例夹具或固定演示命中结果，用来验证执行链和证据链闭环；生产化时应替换为真实知识库、案例库或向量检索 adapter。

推荐 RAG 链路：

```text
用户问题
-> MasterAgent / SubAgent 路由
-> Context Resolver 整理业务对象、用户、租户、数据域
-> Knowledge / Vector Search Tool 检索知识片段
-> Evidence Builder 汇总来源、命中片段、版本、相似度
-> Verifier 检查权限、来源和必要字段
-> Composer 基于结构化证据组织回答
-> Post-check 检查关键数字和动作边界
```

生产级 RAG adapter 建议补齐：

- 文档加载、切分、清洗和版本管理。
- embedding 生成和向量索引接入。
- 关键词 + 向量的 hybrid search。
- rerank、去重和片段压缩。
- 租户、角色、数据域和文档级权限过滤。
- 来源引用、页码、段落 ID、更新时间和可信度。
- 知识新鲜度检查和过期知识处理。
- 检索结果缓存和审计日志。

RAG 的核心边界是：检索、过滤、排序、权限和证据构建应由 Tool / Adapter 完成；LLM 只负责基于已返回的证据进行解释和表达。不要让 LLM 直接编造知识命中、引用来源、指标口径或历史案例。

## 运行 Demo

先配置本地 `config.ini` 中的 LLM 密钥。使用其他配置文件：

```bash
python3 zhuge_agent_demo.py --config /path/to/config.ini --stream tests/cases/query_tag_value_case.json
```

无需真实 API 的离线演示：

```bash
python3 zhuge_agent_demo.py --config tests/cases/fixture_config.ini --stream tests/cases/query_tag_value_case.json
```

流式模式会实时打印步骤进度，并对最终回答使用 token 级输出；LLM 切换为真实 API 后，DCS、数仓等工具仍使用案例或演示数据。若只想看完整最终回答，可加 `--no-typewriter` 关闭终端实时 token 打印。

默认运行工业诊断案例：

```bash
python3 zhuge_agent_demo.py
```

运行 DCS 查数案例：

```bash
python3 zhuge_agent_demo.py tests/cases/query_tag_value_case.json
```

运行普通问答案例，并实时打印主智能体/子智能体进度：

```bash
python3 zhuge_agent_demo.py --stream tests/cases/query_tag_value_case.json
```

运行经营指标分析案例：

```bash
python3 zhuge_agent_demo.py tests/cases/business_sales_case.json
```

运行 graph workflow 案例：

```bash
python3 zhuge_agent_demo.py --config tests/cases/fixture_config.ini --workflow --workflow-id business_metric_dcs_workflow tests/cases/business_dcs_workflow_case.json
```

省略 workflow ID 时，Runtime 会按请求文本从已加载 workflow 中选择最匹配的一条：

```bash
python3 zhuge_agent_demo.py --config tests/cases/fixture_config.ini --workflow tests/cases/business_dcs_workflow_case.json
```

运行并行子智能体案例，并实时打印子智能体进度：

```bash
python3 zhuge_agent_demo.py --parallel tests/cases/parallel_business_dcs_case.json
```

## 测试

安装开发依赖：

```bash
python3 -m pip install -r requirements-dev.txt
```

```bash
python3 -m unittest discover -s tests
```

Pylint 检查：

```bash
python3 -m pylint zhuge_agent zhuge_agent_demo.py tests
```

语法编译检查：

```bash
python3 -m compileall zhuge_agent zhuge_agent_demo.py tests
```

## 设计边界

当前 demo 重点验证架构，不直接连接真实 DCS、数仓或权限系统。生产化时应替换以下工具实现：

- `zhuge_agent/tools/demo/industrial.py` 中的 DCS、预测模型、规则和案例检索工具
- `zhuge_agent/tools/demo/business.py` 中的指标语义、查询计划、权限校验、数仓、可视化和洞察工具

真实接入时建议新增或替换适配器模块，例如：

- `zhuge_agent/tools/adapters/dcs.py`
- `zhuge_agent/tools/adapters/warehouse.py`
- `zhuge_agent/tools/adapters/metric_catalog.py`
- `zhuge_agent/tools/adapters/policy.py`

LLM 只负责语义理解、计划表达和结果表达。业务指标值、工业过程值、预测值、权限结论和可视化数据必须来自工具输出。
