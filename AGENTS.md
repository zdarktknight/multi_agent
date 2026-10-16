# AGENTS.md

本文件用于规范使用 Codex 或其他代码代理维护本项目时的行为。请先阅读 `README.md`，再开始修改代码。

## 项目定位

这是一个“诸葛 AI”智能体架构 demo。核心目标是验证：

```text
自然语言输入
-> MasterAgent 接收通用问题
-> MasterAgent 路由到一个或多个子智能体
-> 子智能体按自身 prompt 模板选择可见 Skill 并规划/表达
-> 工具链执行
-> 证据链
-> LLM 组织回答
-> 结果校验
```

不要把它退化成单个 Prompt demo，也不要把业务流程硬编码进一个 Agent 类里。

## 目录约定

```text
zhuge_agent/
  主运行时代码

zhuge_agent/utils/
  包内公共基础设施工具，例如配置、日志、Prompt 模板、文本处理等

zhuge_agent/agents/
  子智能体配置、专属 skill_ids 和 prompt_template 路径

zhuge_agent/workflows/
  Graph workflow 配置，用于跨子智能体 DAG、并行节点、人审检查点和汇总节点

zhuge_agent/skills/
  平台内置 Skills，随代码包发布

skills/
  外部扩展 Skills，占位保留给客户/项目定制

tests/
  测试和案例数据

draft/
  废弃草稿，除非用户明确要求，否则不要修改
```

## 开发原则

1. 优先扩展 Skill 和 Tool，不要在 `runtime.py` 里写业务特例。
2. 新子智能体应通过 `zhuge_agent/agents/*.json` 声明角色、专属 `skill_ids` 和 `prompt_template` 文件路径。
3. 新能力应优先通过 `zhuge_agent/skills/**/SKILL.md` 和 `skill.json` 声明执行图。
4. 新外部能力可以放到根目录 `skills/`，但内置能力应放到 `zhuge_agent/skills/`。
5. Runtime 应保持通用编排能力，不承载具体业务口径。
6. 测试案例放在 `tests/cases/`，不要放进 `zhuge_agent/` 包内。
7. 不要修改 `draft/`，它是历史资料。
8. 配置、日志、Prompt 模板解析、通用文本处理等跨模块能力放入 `zhuge_agent/utils/`，业务编排和领域工具不要放入 utils。
9. 跨智能体 pipeline / DAG 放入 `zhuge_agent/workflows/*.json`，不要把固定 pipeline 写死到 `MasterAgent` 或某个子智能体类里。

## Agent 规范

MasterAgent 是统一入口，负责理解用户的一般性问题，并判断是否需要委派一个或多个子智能体。普通寒暄、能力说明、无需业务工具的通用问答应由 MasterAgent 直接回答，不应进入子智能体链路。它不直接选择 Skill，不承载业务域专用 prompt，也不计算业务或工业数值。

每个子智能体用一个 JSON 文件声明：

```text
zhuge_agent/agents/
└── some_agent.json
```

子智能体配置必须包含：

- `id`
- `name`
- `role`
- `description`
- `skill_ids`
- `prompt_template`

示例：

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

子智能体的私有能力由 `skill_ids` 声明。业务域 prompt 放在子智能体引用的 Markdown / Jinja2 模板中，不要写进 Skill，也不要以内联字符串写进 JSON。

每个子智能体模板文件至少包含：

```text
<!-- skill_router_system -->
子智能体选择可见 Skill 的提示词。

<!-- planner_system -->
子智能体规划提示词。

<!-- composer_system -->
子智能体回答生成提示词。
```

模板可以使用 Jinja2 变量，例如 `agent`、`visible_skills`、`skill`、`context`、`state` 和 `evidence`。Runtime 构建子智能体时加载模板文件；渲染必须保持 lazy，只在 LLM 客户端消费 system prompt 时触发。

## Skill 规范

每个 Skill 必须是独立文件夹：

```text
skill_name/
├── SKILL.md
└── skill.json
```

`SKILL.md` 用于说明：

- 能力名称
- 适用问题
- 不适用边界
- 执行原则
- 是否允许 LLM 参与

`skill.json` 用于声明：

- `id`
- `name`
- `description`
- `mode`
- `result_type`
- `visibility`
- `allowed_agents`
- `intent_examples`
- `composer_template`
- `steps`

Skill 不声明所属子智能体。它只声明自身是否为公共能力或私有能力：

```json
{
  "visibility": "private",
  "allowed_agents": []
}
```

可见性规则：

- `visibility: "public"`：公共 Skill，对所有子智能体可见。
- `visibility: "private"`：私有 Skill，只对在 `zhuge_agent/agents/*.json` 的 `skill_ids` 中显式声明它的子智能体可见。
- `allowed_agents` 仅作为兼容字段保留，内置 Skill 不应依赖它表达归属。

每个 step 至少包含：

```json
{
  "id": "D1",
  "name": "Executor / Example",
  "kind": "data",
  "tool_id": "some.tool",
  "input_keys": ["context"],
  "output_key": "some_output",
  "no_llm": true
}
```

## Workflow 规范

Workflow 是跨子智能体的图编排层，适合固定 pipeline、复杂 DAG、人审节点和并行分支。它决定“哪些节点以什么依赖关系执行”，不负责计算业务数值，不替代 Skill / Tool / Verifier。

每个 Workflow 用一个 JSON 文件声明：

```text
zhuge_agent/workflows/
└── some_workflow.json
```

当前支持节点类型：

- `agent`：委派给一个子智能体；该节点内部仍必须完整经过 Skill Router、Planner、Executor、Verifier、Composer 和 Post-check。
- `human_review`：人审检查点；demo 可使用 `config.mode = "auto_approve"`，生产接入真实审批系统时应保持节点输出结构化。
- `join`：汇总节点；只汇总已验证的上游结果，不重新计算业务数值。

Workflow 配置必须包含：

- `id`
- `name`
- `description`
- `trigger_examples`
- `nodes`
- `edges`

示例：

```json
{
  "id": "business_metric_dcs_workflow",
  "name": "经营指标 + DCS 查数复合工作流",
  "nodes": [
    {"id": "business_metrics", "type": "agent", "name": "经营指标分析节点", "agent_id": "business_metric_agent"},
    {"id": "dcs_values", "type": "agent", "name": "DCS 当前值节点", "agent_id": "dcs_query_agent"},
    {"id": "human_review", "type": "human_review", "name": "人审检查点", "config": {"mode": "auto_approve"}},
    {"id": "final_join", "type": "join", "name": "汇总输出节点"}
  ],
  "edges": [
    {"from": "business_metrics", "to": "human_review"},
    {"from": "dcs_values", "to": "human_review"},
    {"from": "human_review", "to": "final_join", "condition": "passed"}
  ]
}
```

Workflow 必须是有向无环图。新增条件分支、人审状态、可视化编排或外部审批适配时，应优先扩展 Workflow node / edge 语义和测试，不要绕过 `Verifier` 或让 LLM 直接生成确定性结果。

## no_llm 规则

本项目最重要的设计边界是：能由确定性工具完成的，不交给 LLM 推测。

必须保持 `no_llm=true` 的步骤：

- 数据读取
- 指标计算
- 数仓查询
- 权限校验
- 口径校验
- 规则判断
- 模型预测结果读取
- 证据链收集
- Post-check

允许使用 LLM 的步骤：

- MasterAgent Router：识别用户意图并选择一个或多个子智能体
- SubAgent Skill Router：在公共 Skill 和自身私有 Skill 中选择具体能力
- Planner：解释执行计划
- Composer：把结构化结果组织成自然语言
- 特定 Skill 中明确声明允许 LLM 的语义解析步骤

LLM 不得直接生成业务指标值、工业测点值、预测数值、权限结论或图表数据。

## Tool 规范

新增工具应实现 `zhuge_agent/tools/base.py` 中的 `Tool` 接口，并注册到 `zhuge_agent/tools/registry.py` 的 `ToolRegistry.default()`。

当前 `zhuge_agent/tools/demo/industrial.py` 和 `zhuge_agent/tools/demo/business.py` 是离线 demo 工具实现，可以包含案例夹具和确定性 fallback。生产接入真实 DCS、数仓、指标平台、权限系统时，应新增清晰命名的 adapter 模块，避免继续把真实连接逻辑塞进 demo 模块。

工具输出建议包含：

```python
{
    "...": "...",
    "evidence": [
        Evidence(source="...", detail="...")
    ]
}
```

如果工具可能使用 LLM，`ToolSpec.no_llm` 必须设为 `False`，并且对应 Skill step 不得标记 `no_llm=true`。

## Runtime 规范

`runtime.py` 负责通用流程：

```text
A1 MasterAgent / Router
A2 SubAgent / Skill Router
B Context Resolver
SubAgent / C Planner
D Executor
E Evidence Builder
F Verifier
SubAgent / G Composer
H Post-check
```

修改 Runtime 前先确认：

1. 这个逻辑是否能放到 Skill。
2. 这个逻辑是否能放到 Tool。
3. 这个逻辑是否能放到某个子智能体配置的 prompt 模板。
4. 这个逻辑是否是跨所有业务能力通用的编排规则。

只有第 4 类才适合修改 Runtime。

并行编排规则：

- `answer(request, progress_callback=...)` 保持单子智能体链路；传入回调时，每完成一个步骤都应通过 `AgentProgressEvent` 回调进度。
- 对普通寒暄或无需工具的通用问答，`answer(request, ...)` 可以返回 `skill_id=direct_answer` 的 MasterAgent 直接回答，此时只应产生 A1 进度，不应触发 A2、Planner、Executor、Verifier 或 Post-check。
- `iter_answer(request)` 用于普通问答的流式消费，先 yield `AgentProgressEvent`，最后 yield `StructuredAnswer`。
- `answer_parallel(request, progress_callback=...)` 由 MasterAgent 一次选择多个子智能体，并由 Runtime 并发执行各子智能体链路；每完成一个步骤都应通过 `AgentProgressEvent` 回调进度。
- `iter_answer_parallel(request)` 用于流式消费进度，先 yield `AgentProgressEvent`，最后 yield `ParallelStructuredAnswer`。
- `answer_workflow(request, workflow_id=..., progress_callback=...)` 执行 graph workflow；同一 DAG 层级的节点可以并发执行，最后返回 `WorkflowStructuredAnswer`。
- `iter_answer_workflow(request, workflow_id=...)` 流式输出 workflow 节点进度、子智能体进度和最终回答 token，最后 yield `WorkflowStructuredAnswer`。
- 子智能体之间不要互相直接调用；跨子智能体并行、汇总和排序属于 Runtime / MasterAgent 编排职责。
- 每个并行子任务仍必须独立经过 Skill Router、Planner、Executor、Verifier、Composer 和 Post-check。
每个 workflow 中的 agent 节点也必须独立经过 Skill Router、Planner、Executor、Verifier、Composer 和 Post-check。

## 测试规范

任何新增 Skill 或 Tool 都要补测试。

最低要求：

1. 在 `tests/cases/` 增加一个案例输入。
2. 在 `tests/test_runtime.py` 或新的测试文件中验证 Skill 路由。
3. 验证关键输出字段存在。
4. 验证 `no_llm` 步骤没有被标记为 LLM 调用。
5. 验证最终回答中的关键数字可追溯到工具输出。

运行：

```bash
python3 -m unittest discover -s tests
python3 -m compileall zhuge_agent zhuge_agent_demo.py tests
```

## LLM API 规范

LLM 通过项目根目录的本地 `config.ini` 配置，不读取 LLM 环境变量：

```ini
[llm]
provider = openai
api_key =
base_url = https://api.openai.com/v1
model = gpt-4o-mini
timeout_seconds = 60
```

`provider = openai` 调用真实 API，配置无效时应明确报错，不自动降级为 fixture。`provider = fixture` 用于离线演示。CLI 用 `--config` 指定其他路径，Runtime 用 `config_path` 指定路径，显式注入 LLM 客户端时跳过文件加载。

本地 `config.ini` 已加入 `.gitignore`；只能在本地文件填写真实密钥，可提交的示例是 `config.example.ini`。不要把真实 API key 写入代码、测试、README、案例或日志。

测试应使用 `FixtureLLMClient`，保证离线、稳定、可重复。

## 代码风格

1. 使用 Python 标准库优先，除非用户明确要求引入依赖。
2. 保持类型标注。
3. 保持工具输出结构化，避免返回难解析的大段字符串。
4. 保持错误显式，例如 `SkillLoadError`、`LLMError`、`HardRuleViolation`。
5. 不要使用随机数生成测试关键结果。
6. 不要把测试用例写进运行时代码。

## 日志规范

1. 使用标准库 `logging.getLogger(__name__)`，统一格式和上下文由 `zhuge_agent/utils/logging_utils.py` 管理，不在各模块重复配置 Handler。
2. 日志格式固定为 `agent_id session_id %(asctime)s.%(msecs)03d %(levelname)s %(module)s@%(lineno)d - %(message)s`，时间格式为 `%Y-%m-%d %H:%M:%S`。
3. 关键函数使用 `@log_call("关键参数名", ...)` 记录输入、输出摘要、耗时和异常；普通事件使用模块 Logger。细粒度解析、模板和注册表查询使用 DEBUG，关键执行步骤使用 INFO，失败使用 ERROR。
4. 请求入口绑定主智能体和 session_id，子任务绑定配置 ID。新增线程需要用独立的 `copy_context()` 传递上下文，禁止把请求身份放到可变全局变量中。
5. 不记录真实密钥、Authorization、密码或连接凭证。密钥交给 `register_secret` 登记，结构化数据交给 `log_value` 生成脱敏摘要，禁止直接打印配置全文或客户端对象属性。
6. 不能为日志调用 `str(LazyPrompt)`、布尔判断或 render；日志不得改变延迟渲染和业务返回值。
7. `[logging]` 管理日志级别和可选文件路径。文件输出到 `logs/`，该目录不提交。不要用 Logger 代替 Demo 有意输出的业务结果。
8. 修改日志行为时验证并发会话隔离、流式线程上下文、异常恢复、密钥脱敏和重复配置不产生重复 Handler。

## 提交前检查

完成修改后至少运行：

```bash
python3 -m unittest discover -s tests
python3 -m pylint zhuge_agent zhuge_agent_demo.py tests
python3 -m compileall zhuge_agent zhuge_agent_demo.py tests
```

如果改动涉及 demo 行为，也运行对应案例：

```bash
python3 zhuge_agent_demo.py --config tests/cases/fixture_config.ini tests/cases/query_tag_value_case.json
python3 zhuge_agent_demo.py --config tests/cases/fixture_config.ini tests/cases/business_sales_case.json
python3 zhuge_agent_demo.py --config tests/cases/fixture_config.ini tests/cases/reactor_temp_case.json
```

## 禁止事项

- 不要修改 `draft/`，除非用户明确要求。
- 不要把业务口径硬编码到 `runtime.py`。
- 不要让 LLM 直接计算指标值或工业过程值。
- 不要让未校验查询计划直接访问数据源。
- 不要绕过 `Verifier` 或 `Post-check`。
- 不要提交密钥、真实连接串或客户敏感数据。
