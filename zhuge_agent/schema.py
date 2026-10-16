from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class StepKind(str, Enum):
    """执行步骤类型，用于描述 Skill 执行图和记录工具调用类别。"""

    LLM = "llm"
    SEMANTIC = "semantic"
    DATA = "data"
    QUERY = "query"
    WAREHOUSE = "warehouse"
    MODEL = "model"
    RULE = "rule"
    KNOWLEDGE = "knowledge"
    VISUALIZATION = "visualization"
    COMPOSER = "composer"
    VERIFIER = "verifier"


@dataclass(frozen=True)
class ToolSpec:
    """工具能力声明，包含标识、类型和是否禁止 LLM 的执行边界。"""

    id: str
    name: str
    kind: StepKind
    description: str
    no_llm: bool = True


@dataclass(frozen=True)
class SkillStep:
    """Skill 中的一个执行步骤，声明输入依赖、输出键、工具 ID 和 no_llm 约束。"""

    id: str
    name: str
    kind: StepKind
    tool_id: str | None
    input_keys: tuple[str, ...]
    output_key: str
    no_llm: bool


@dataclass(frozen=True)
class AgentSpec:
    """子智能体角色配置，包含私有 Skill ID 和可选的提示词模板路径。"""

    id: str
    name: str
    role: str
    description: str
    skill_ids: tuple[str, ...] = ()
    prompt_template_path: Path | None = None


@dataclass(frozen=True)
class Skill:
    """完整能力定义，组合用途说明、可见性、执行步骤及回答组织要求。"""

    id: str
    name: str
    description: str
    mode: str
    intent_examples: tuple[str, ...]
    steps: tuple[SkillStep, ...]
    composer_template: str
    result_type: str
    visibility: str = "private"
    allowed_agents: tuple[str, ...] = ()
    instructions: str = ""
    root_path: Path | None = None


@dataclass
class TraceEvent:
    """单个步骤的执行记录，供校验、调试和链路展示使用。"""

    step: str
    actor: str
    status: str
    message: str
    used_llm: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Evidence:
    """工具结果的来源与说明，用于将回答关联到数据、规则或模型输出。"""

    source: str
    detail: str


@dataclass
class StepResult:
    """工具执行结果，封装业务数据、证据、输出键和 LLM 使用标记。"""

    step_id: str
    actor: str
    kind: StepKind
    output_key: str
    data: dict[str, Any]
    evidence: list[Evidence]
    used_llm: bool


@dataclass(frozen=True)
class MemoryItem:
    """后端或 Memory Service 检索出的记忆片段，供 Runtime 作为上下文消费。

    scope 表示记忆作用域，例如 session、user、tenant 或 domain；Runtime 不负责鉴权，
    只消费后端已经筛选、脱敏和授权后的内容。
    """

    type: str
    content: str
    scope: str
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryWriteCandidate:
    """智能体运行后建议写入长期记忆的候选项，是否持久化由后端裁决。

    该结构只表达建议，不代表 Runtime 已经写入任何数据库、向量库或会话存储。
    """

    type: str
    content: str
    scope: str
    confidence: float
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StructuredAnswer:
    """单个子智能体的最终结果，包含文案、指标、建议、证据、校验及执行进度。"""

    request: str
    agent_id: str
    agent_name: str
    intent: str
    skill_id: str
    answer: str
    metrics: dict[str, Any]
    evidence: list[Evidence]
    actions: list[str]
    trace: list[TraceEvent]
    verification: dict[str, Any]
    progress: list[AgentProgressEvent] = field(default_factory=list)
    memory_write_candidates: list[MemoryWriteCandidate] = field(default_factory=list)


@dataclass
class AgentProgressEvent:
    """适合回调或流式传递的进度事件，携带子智能体身份及步骤状态。"""

    request: str
    agent_id: str | None
    agent_name: str | None
    step: str
    actor: str
    status: str
    message: str
    used_llm: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnswerTokenEvent:
    """最终回答的增量文本事件，仅用于流式展示，不参与工具执行和校验。

    delta 是本次新增文本片段；sequence 是同一个回答内从 1 开始递增的片段序号。
    """

    request: str
    agent_id: str | None
    agent_name: str | None
    delta: str
    sequence: int


@dataclass(frozen=True)
class AgentRunRequest:
    """后端接入 Runtime 的结构化请求，不负责鉴权，只承载已解析的调用上下文。

    mode 用于选择运行入口：auto/single 使用主智能体单链路，parallel 使用并行委派，
    workflow 使用 graph workflow；context 可由后端放入租户、角色、数据域或业务对象。
    长对话上下文由后端筛选后放入 recent_messages、conversation_summary 和 retrieved_memories。
    """

    request: str
    mode: str = "auto"
    workflow_id: str | None = None
    max_workers: int | None = None
    session_id: str | None = None
    user_id: str | None = None
    tenant_id: str | None = None
    role: str | None = None
    data_scope: dict[str, Any] = field(default_factory=dict)
    recent_messages: list[dict[str, Any]] = field(default_factory=list)
    conversation_summary: str | None = None
    retrieved_memories: list[MemoryItem] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeErrorInfo:
    """面向后端事件流的错误协议，避免 API 层直接解析 Python 异常对象。"""

    code: str
    message: str
    retryable: bool = False
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeEvent:
    """统一运行事件包装，便于后端直接映射为 SSE、WebSocket 或消息队列事件。"""

    type: str
    payload: Any
    sequence: int
    session_id: str | None = None
    run_mode: str = "auto"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParallelStructuredAnswer:
    """并行子任务的汇总结果，保留每个子回答、合并文案、总体校验和进度。"""

    request: str
    answer: str
    sub_answers: list[StructuredAnswer]
    trace: list[TraceEvent]
    verification: dict[str, Any]
    progress: list[AgentProgressEvent] = field(default_factory=list)
    memory_write_candidates: list[MemoryWriteCandidate] = field(default_factory=list)


@dataclass(frozen=True)
class WorkflowNode:
    """Graph workflow 中的一个节点，可表示子智能体、人审检查点或汇总节点。"""

    id: str
    type: str
    name: str
    agent_id: str | None = None
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkflowEdge:
    """Graph workflow 中的有向边，condition 用于后续扩展条件分支。"""

    from_node: str
    to_node: str
    condition: str = "always"


@dataclass(frozen=True)
class WorkflowSpec:
    """可配置工作流定义，描述触发示例、节点和边组成的有向无环图。"""

    id: str
    name: str
    description: str
    trigger_examples: tuple[str, ...]
    nodes: tuple[WorkflowNode, ...]
    edges: tuple[WorkflowEdge, ...]


@dataclass
class WorkflowStructuredAnswer:
    """Graph workflow 的最终结果，包含每个 agent 节点回答、汇总文案、校验和进度。"""

    request: str
    workflow_id: str
    workflow_name: str
    answer: str
    sub_answers: list[StructuredAnswer]
    node_outputs: dict[str, Any]
    trace: list[TraceEvent]
    verification: dict[str, Any]
    progress: list[AgentProgressEvent] = field(default_factory=list)
    memory_write_candidates: list[MemoryWriteCandidate] = field(default_factory=list)
