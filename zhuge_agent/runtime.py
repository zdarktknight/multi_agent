from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from dataclasses import asdict, dataclass
from pathlib import Path
from queue import Queue
from threading import Lock, Thread
from typing import Any, Callable, Iterator

from .agent_loader import AgentRegistry
from .llm import LLMClient, LLMError, llm_client_from_config
from .schema import (
    AgentRunRequest,
    AgentSpec,
    AgentProgressEvent,
    AnswerTokenEvent,
    Evidence,
    ParallelStructuredAnswer,
    RuntimeErrorInfo,
    RuntimeEvent,
    Skill,
    StepKind,
    StepResult,
    StructuredAnswer,
    TraceEvent,
    WorkflowNode,
    WorkflowSpec,
    WorkflowStructuredAnswer,
)
from .skill_loader import SkillRegistry
from .tools import ToolRegistry
from .utils.config import load_logging_config
from .utils.logging_utils import configure_logging, ensure_logging_configured, log_call, log_value
from .utils.prompt_templates import AgentPromptTemplate
from .workflow_loader import WorkflowRegistry


class HardRuleViolation(RuntimeError):
    """确定性校验或后验检查失败时抛出，避免未验证结果继续对外返回。"""


ProgressCallback = Callable[[AgentProgressEvent], None]
TokenCallback = Callable[[AnswerTokenEvent], None]
logger = logging.getLogger(__name__)


@dataclass
class _AgentExecutionContext:
    """子智能体链路的执行回调和外部上下文，避免内部函数参数继续膨胀。"""

    progress_callback: ProgressCallback | None = None
    token_callback: TokenCallback | None = None
    progress_events: list[AgentProgressEvent] | None = None
    context_override: dict[str, Any] | None = None


class MasterAgent:
    """接收通用用户问题并委派子智能体，支持单任务路由和并行任务选择。"""

    name = "MasterAgent"

    def __init__(self, llm: LLMClient) -> None:
        """注入用于识别意图和选择子智能体的 LLM 客户端。"""
        self.llm = llm

    @log_call("request")
    def plan_execution(
        self,
        request: str,
        agent_registry: AgentRegistry,
        workflow_registry: WorkflowRegistry,
    ) -> tuple[dict[str, Any], TraceEvent]:
        """让 MasterAgent 判断整体执行模式：直接回答、单智能体、并行智能体或 workflow。

        本步骤只选择编排方式，不选择具体 Skill，也不计算业务或工业数值；串行、有依赖或需要人审
        的任务应优先选择已配置的 graph workflow。
        """
        agents = [
            {
                "agent_id": agent.id,
                "name": agent.name,
                "role": agent.role,
                "description": agent.description,
                "private_skill_ids": list(agent.skill_ids),
            }
            for agent in agent_registry.all()
        ]
        workflows = [
            {
                "workflow_id": workflow.id,
                "name": workflow.name,
                "description": workflow.description,
                "trigger_examples": list(workflow.trigger_examples),
                "node_count": len(workflow.nodes),
            }
            for workflow in workflow_registry.all()
        ]
        response = self.llm.chat_json(
            system=(
                "You are MasterAgent in Zhuge AI. Decide the execution mode for the user's request. "
                "Use direct for greetings or general questions that need no tools. Use single when one "
                "specialist sub-agent can finish the task. Use parallel when multiple independent "
                "specialist sub-agents can work at the same time. Use workflow when the task has serial "
                "dependencies, requires a configured pipeline, needs human review, or matches an available "
                "graph workflow. Do not select concrete Skills and do not calculate business or industrial "
                "numbers. Return JSON only."
            ),
            user=_json_dumps(
                {
                    "request": request,
                    "available_agents": agents,
                    "available_workflows": workflows,
                }
            ),
            schema_hint={
                "execution_mode": "direct|single|parallel|workflow",
                "agent_ids": ["string"],
                "workflow_id": "string",
                "intent": "string",
                "confidence": 0.0,
                "reason": "string",
                "direct_answer": "string",
            },
        )
        plan = self._normalize_execution_plan(response, request, agent_registry, workflow_registry)
        return plan, TraceEvent(
            step="A0",
            actor=f"{self.name} / Execution Planner",
            status="done",
            message=f"选择执行模式「{plan['execution_mode']}」：{plan.get('reason', '')}",
            used_llm=True,
            metadata=plan,
        )

    @log_call("request")
    def route(self, request: str, registry: AgentRegistry) -> tuple[AgentSpec | None, dict[str, Any], TraceEvent]:
        """调用 LLM 判断直接回答或选择子智能体，返回规范化意图和 A1 路由记录。

        普通寒暄或不需要业务工具的问题直接由 MasterAgent 回答；需要专业能力时才委派子智能体。
        LLM 返回的 ID 未注册且不是直接回答时改用 registry 的文本匹配路由。
        """
        agents = [
            {
                "agent_id": agent.id,
                "name": agent.name,
                "role": agent.role,
                "description": agent.description,
                "private_skill_ids": list(agent.skill_ids),
            }
            for agent in registry.all()
        ]
        intent = self.llm.chat_json(
            system=(
                "You are MasterAgent in Zhuge AI. Receive the user's general question. If the request is "
                "small talk, a greeting, a capability question, or a general answer that does not require "
                "business metrics, DCS data, diagnosis, tools, evidence, or a specialist workflow, set "
                "route_type to 'direct', leave agent_id empty, and provide direct_answer. Only select a "
                "specialist sub-agent when the request needs that specialist workflow. Do not select a "
                "concrete Skill and do not calculate business or industrial numbers. Return JSON only."
            ),
            user=_json_dumps({"request": request, "available_agents": agents}),
            schema_hint={
                "route_type": "direct|delegate",
                "agent_id": "string",
                "intent": "string",
                "confidence": 0.0,
                "reason": "string",
                "direct_answer": "string",
            },
        )
        if intent.get("route_type") == "direct" or (not intent.get("agent_id") and intent.get("direct_answer")):
            normalized = {
                "agent_id": None,
                "intent": intent.get("intent") or "通用问答",
                "confidence": float(intent.get("confidence", 0.0)),
                "reason": intent.get("reason", ""),
                "route_type": "direct",
                "direct_answer": intent.get("direct_answer") or "你好，我在。有什么想一起看看的吗？",
            }
            return None, normalized, TraceEvent(
                step="A1",
                actor=f"{self.name} / Router",
                status="done",
                message=f"识别意图为「{normalized['intent']}」，由 MasterAgent 直接回答。",
                used_llm=True,
                metadata=normalized,
            )
        agent_id = intent.get("agent_id")
        agent = registry.get(agent_id) if registry.has(agent_id) else registry.route(request)
        intent = {
            "agent_id": agent.id,
            "intent": intent.get("intent") or agent.name,
            "confidence": float(intent.get("confidence", 0.0)),
            "reason": intent.get("reason", ""),
            "route_type": "delegate",
        }
        return agent, intent, TraceEvent(
            step="A1",
            actor=f"{self.name} / Router",
            status="done",
            message=f"识别意图为「{intent['intent']}」，委派给「{agent.name}」。",
            used_llm=True,
            metadata={**intent, "sub_agent_name": agent.name},
        )

    @log_call("request")
    def route_many(self, request: str, registry: AgentRegistry) -> tuple[list[AgentSpec], dict[str, Any], TraceEvent]:
        """让 LLM 为复合问题选择多个子智能体，或声明无需委派并由 MasterAgent 直接回答。

        过滤无效及重复 ID；没有有效候选时使用备用路由选择一个智能体，实际并发由 Runtime 执行。
        """
        agents = [
            {
                "agent_id": agent.id,
                "name": agent.name,
                "role": agent.role,
                "description": agent.description,
                "private_skill_ids": list(agent.skill_ids),
            }
            for agent in registry.all()
        ]
        response = self.llm.chat_json(
            system=(
                "You are MasterAgent in Zhuge AI. Receive the user's general question, decompose it into "
                "independent specialist sub-agent delegations when useful, and return every sub-agent that "
                "should work in parallel. If the request is small talk, a greeting, or a general answer "
                "that does not need specialist tools, set route_type to 'direct', return no agent_ids, and "
                "provide direct_answer. Do not select concrete Skills and do not calculate business or "
                "industrial numbers. Return JSON only."
            ),
            user=_json_dumps({"request": request, "available_agents": agents}),
            schema_hint={
                "route_type": "direct|delegate",
                "agent_ids": ["string"],
                "intent": "string",
                "confidence": 0.0,
                "reason": "string",
                "direct_answer": "string",
            },
        )
        if response.get("route_type") == "direct" or (
            not response.get("agent_ids") and response.get("direct_answer")
        ):
            intent = {
                "agent_ids": [],
                "intent": response.get("intent") or "通用问答",
                "confidence": float(response.get("confidence", 0.0)),
                "reason": response.get("reason", ""),
                "route_type": "direct",
                "direct_answer": response.get("direct_answer") or "你好，我在。有什么想一起看看的吗？",
            }
            return [], intent, TraceEvent(
                step="A1",
                actor=f"{self.name} / Parallel Router",
                status="done",
                message=f"识别意图为「{intent['intent']}」，由 MasterAgent 直接回答。",
                used_llm=True,
                metadata=intent,
            )
        selected = self._selected_agents(response.get("agent_ids", []), registry)
        if not selected:
            selected = [registry.route(request)]
        intent = {
            "agent_ids": [agent.id for agent in selected],
            "intent": response.get("intent") or "并行智能体委派",
            "confidence": float(response.get("confidence", 0.0)),
            "reason": response.get("reason", ""),
            "route_type": "delegate",
        }
        names = "、".join(agent.name for agent in selected)
        return selected, intent, TraceEvent(
            step="A1",
            actor=f"{self.name} / Parallel Router",
            status="done",
            message=f"识别意图为「{intent['intent']}」，并行委派给 {len(selected)} 个子智能体：{names}。",
            used_llm=True,
            metadata={**intent, "sub_agent_names": [agent.name for agent in selected]},
        )

    def _selected_agents(self, agent_ids: Any, registry: AgentRegistry) -> list[AgentSpec]:
        """按 LLM 列表顺序保留已注册且未重复的智能体，非列表输入返回空列表。"""
        selected = []
        seen = set()
        for agent_id in agent_ids if isinstance(agent_ids, list) else []:
            if registry.has(agent_id) and agent_id not in seen:
                selected.append(registry.get(agent_id))
                seen.add(agent_id)
        return selected

    def _normalize_execution_plan(
        self,
        response: dict[str, Any],
        request: str,
        agent_registry: AgentRegistry,
        workflow_registry: WorkflowRegistry,
    ) -> dict[str, Any]:
        """校正 LLM 返回的执行模式和目标 ID，保证 Runtime 后续入口可安全消费。"""
        mode = str(response.get("execution_mode") or "").strip().lower()
        selected_agents = self._selected_agents(response.get("agent_ids", []), agent_registry)
        workflow_id = str(response.get("workflow_id") or "").strip()
        if mode == "workflow" and not workflow_registry.has(workflow_id):
            workflow_id = workflow_registry.route(request).id if workflow_registry.all() else ""
            mode = "workflow" if workflow_id else "parallel"
        if mode not in {"direct", "single", "parallel", "workflow"}:
            mode = "parallel" if len(selected_agents) > 1 else "single" if selected_agents else "direct"
        if mode == "direct":
            selected_agents = []
        elif mode == "single":
            selected_agents = selected_agents[:1] or [agent_registry.route(request)]
        elif mode == "parallel":
            selected_agents = selected_agents or [agent_registry.route(request)]
            if len(selected_agents) == 1:
                mode = "single"
        return {
            "execution_mode": mode,
            "agent_ids": [agent.id for agent in selected_agents],
            "workflow_id": workflow_id if mode == "workflow" else "",
            "intent": response.get("intent") or "自动编排",
            "confidence": float(response.get("confidence", 0.0)),
            "reason": response.get("reason", ""),
            "direct_answer": response.get("direct_answer") or "你好，我在。有什么想一起看看的吗？",
        }


class SubAgent:
    """封装子智能体角色、专属提示词模板和可见 Skill 的选择能力。"""

    def __init__(self, spec: AgentSpec, llm: LLMClient) -> None:
        """绑定角色配置与 LLM，读取可选模板文件的段落，保留提示词延迟渲染。"""
        self.spec = spec
        self.llm = llm
        self.prompt_template = AgentPromptTemplate(spec.prompt_template_path) if spec.prompt_template_path else None

    @property
    def name(self) -> str:
        """返回配置中的子智能体显示名称，供执行记录和最终结果使用。"""
        return self.spec.name

    @log_call("key", "variables", level=logging.DEBUG, context="agent")
    def prompt(self, key: str, fallback: str, **variables: Any) -> object:
        """注入 agent 变量并返回指定段落的 LazyPrompt；没有模板时直接返回 fallback 文本。"""
        values = {"agent": self.spec, **variables}
        if not self.prompt_template:
            return fallback
        return self.prompt_template.lazy_render(key, fallback, **values)

    @log_call("request", context="agent")
    def select_skill(self, request: str, registry: SkillRegistry) -> tuple[Skill, TraceEvent]:
        """让 LLM 在公共和自身私有 Skill 中选择能力，返回 Skill 与 A2 执行记录。

        若返回 ID 不在可见集合中，则使用 registry 对可见候选进行文本匹配。
        """
        visible_skills = registry.visible_to(self.spec.id, self.spec.skill_ids)
        response = self.llm.chat_json(
            system=self.prompt(
                "skill_router_system",
                (
                    "You are a Zhuge AI specialist sub-agent. Select the best visible Skill for the user "
                    "request. Public skills are available to you, and private skills are available only "
                    "when assigned to your agent. Return JSON only."
                ),
                visible_skills=visible_skills,
            ),
            user=_json_dumps(
                {
                    "request": request,
                    "sub_agent": {
                        "id": self.spec.id,
                        "name": self.spec.name,
                        "role": self.spec.role,
                    },
                    "visible_skills": [
                        {
                            "skill_id": skill.id,
                            "name": skill.name,
                            "description": skill.description,
                            "mode": skill.mode,
                            "visibility": skill.visibility,
                            "intent_examples": list(skill.intent_examples),
                        }
                        for skill in visible_skills
                    ],
                }
            ),
            schema_hint={"skill_id": "string", "intent": "string", "confidence": 0.0, "reason": "string"},
        )
        skill_id = response.get("skill_id")
        visible_ids = {skill.id for skill in visible_skills}
        skill = registry.get(skill_id) if skill_id in visible_ids else registry.route(request, visible_skills)
        return skill, TraceEvent(
            step="A2",
            actor=f"{self.name} / Skill Router",
            status="done",
            message=f"在可见 Skills 中选择「{skill.name}」。",
            used_llm=True,
            metadata={
                "sub_agent_id": self.spec.id,
                "skill_id": skill.id,
                "visible_skill_ids": sorted(visible_ids),
                "reason": response.get("reason", ""),
            },
        )


class ContextResolverAgent:
    """为执行链合并请求、案例上下文和演示默认参数。"""

    name = "B Context Resolver"

    def __init__(self, case_data: dict[str, Any] | None = None) -> None:
        """保存可选案例上下文，供后续执行步骤共享装置、窗口和访问主体信息。"""
        self.case_data = case_data or {}

    @log_call("request", "context_override")
    def run(self, request: str, context_override: dict[str, Any] | None = None) -> tuple[dict[str, Any], TraceEvent]:
        """构造包含请求、装置、小时窗口、租户、角色及默认阈值的上下文，返回 B 步骤记录。

        context_override 是后端按用户、租户、数据域预处理后的本次运行上下文，优先级高于案例夹具。
        """
        provided = {**(self.case_data.get("context") or {}), **(context_override or {})}
        asset = provided.get("asset") or ("2# 反应釜" if "2#" in request or "反应釜" in request else "默认装置")
        context = dict(provided)
        context.update({
            "request": request,
            "asset": asset,
            "hours": int(provided.get("hours", 72)),
            "tenant": provided.get("tenant", "中粮（太仓）"),
            "role": provided.get("role", "业务用户"),
            "defaults": {
                **(provided.get("defaults") or {}),
                "temperature_threshold": float(provided.get("temperature_threshold", 130.0)),
            },
        })
        return context, TraceEvent(
            step="B",
            actor=self.name,
            status="done",
            message=f"合并会话上下文和 Agent 默认值：对象={asset}，窗口={context['hours']}h。",
            used_llm=False,
            metadata=context,
        )


class PlannerAgent:
    """通过子智能体模板解释 Skill 执行图，保留确定性步骤的约束。"""

    @log_call("skill", "context")
    def run(self, sub_agent: SubAgent, skill: Skill, context: dict[str, Any]) -> TraceEvent:
        """向 LLM 提交 Skill 步骤和上下文，返回计划说明及 no_llm 步骤信息的 C 记录。

        实际执行顺序仍由 Skill 配置决定，本函数不让 LLM 生成或替换业务计算结果。
        """
        system_prompt = sub_agent.prompt(
            "planner_system",
            (
                "You are C Planner in Zhuge AI. Expand the selected Skill.graph into an execution plan. "
                "Respect no_llm steps: data/model/rule/knowledge/verifier steps must stay deterministic. "
                "Do not calculate industrial numbers. Return JSON only."
            ),
            skill=skill,
            context=context,
        )
        plan = sub_agent.llm.chat_json(
            system=system_prompt,
            user=_json_dumps({
                "skill": {
                    "id": skill.id,
                    "name": skill.name,
                    "steps": [
                        {
                            "id": s.id,
                            "name": s.name,
                            "kind": s.kind.value,
                            "tool_id": s.tool_id,
                            "no_llm": s.no_llm,
                        }
                        for s in skill.steps
                    ],
                },
                "context": context,
            }),
            schema_hint={"steps": ["string"], "no_llm_steps": ["string"], "reason": "string"},
        )
        fallback_no_llm = [s.name for s in skill.steps if s.no_llm]
        steps = plan.get("steps") or [s.name for s in skill.steps]
        no_llm_steps = plan.get("no_llm_steps") or fallback_no_llm
        return TraceEvent(
            step="C",
            actor=f"{sub_agent.name} / C Planner",
            status="done",
            message=f"展开执行计划：{len(steps)} 个步骤，其中 {len(no_llm_steps)} 个 no_llm 步骤。",
            used_llm=True,
            metadata={
                "sub_agent_id": sub_agent.spec.id,
                "steps": steps,
                "no_llm_steps": no_llm_steps,
                "reason": plan.get("reason", ""),
            },
        )


class ExecutorAgent:
    """统一调用已注册工具，并在执行前检查步骤的 no_llm 边界。"""

    name = "D Executor"

    def __init__(self, tools: ToolRegistry) -> None:
        """注入工具注册表，避免执行器依赖具体业务工具类。"""
        self.tools = tools

    @log_call("step_id", "tool_id", "inputs", "no_llm")
    def run(self, step_id: str, tool_id: str, inputs: dict[str, Any], no_llm: bool) -> StepResult:
        """校验步骤与工具的 LLM 约束后调用工具，封装业务结果、证据和工具类型。

        禁止 LLM 的步骤绑定允许 LLM 的工具时抛出 HardRuleViolation。
        used_llm 优先读取工具返回的 _used_llm，否则根据工具声明标记；
        output_key 由 Runtime 按 Skill 配置随后填写。
        """
        tool = self.tools.get(tool_id)
        if no_llm and not tool.spec.no_llm:
            raise HardRuleViolation(f"{step_id} requires no_llm, but tool {tool_id} may use LLM")
        result = tool.invoke(inputs)
        evidence = result.pop("evidence", [])
        used_llm = bool(result.pop("_used_llm", not tool.spec.no_llm))
        return StepResult(
            step_id=step_id,
            actor=f"{self.name} / {tool.spec.name}",
            kind=tool.spec.kind,
            output_key="",
            data=result,
            evidence=evidence,
            used_llm=used_llm,
        )


class EvidenceBuilderAgent:
    """收集各工具产生的证据，形成回答和校验共享的来源链。"""

    name = "E Evidence Builder"

    @log_call("results")
    def collect(self, results: list[StepResult]) -> tuple[list[Evidence], TraceEvent]:
        """按步骤顺序汇总 StepResult.evidence，返回证据列表及 E 步骤记录，不调用 LLM。"""
        evidence: list[Evidence] = []
        for result in results:
            evidence.extend(result.evidence)
        return evidence, TraceEvent(
            step="E",
            actor=self.name,
            status="done",
            message=f"组装证据链：{len(evidence)} 项来源、模型、规则和历史案例。",
            used_llm=False,
            metadata={"evidence_count": len(evidence)},
        )


class VerifierAgent:
    """在生成文案前检查输出完整性、查询策略结果及 Skill 的 no_llm 类型声明。"""

    name = "F Verifier"

    @log_call("state", "skill")
    def verify(self, state: dict[str, Any], skill: Skill) -> tuple[dict[str, Any], TraceEvent]:
        """按结果类型检查缺失输出及 no_llm 与步骤类型的冲突，返回校验字典及 F 记录。

        经营分析还需检查 policy_check；LLM 执行记录仅汇总到结果中，由 Runtime 决定是否中止。
        """
        violations = []
        for step in skill.steps:
            if step.no_llm and step.kind == StepKind.LLM:
                violations.append(f"{step.id}:{step.name} 标记 no_llm 但 kind=llm")
        required_by_type = {
            "diagnosis": ("dcs_data", "forecast", "rule_check", "case_evidence"),
            "tag_query": ("current_values",),
            "business_metric_analysis": (
                "semantic_parse",
                "query_plan",
                "policy_check",
                "warehouse_result",
                "visualization",
                "business_insights",
            ),
        }
        required = required_by_type.get(skill.result_type, ())
        missing = [key for key in required if key not in state]
        policy_failed = (
            skill.result_type == "business_metric_analysis"
            and not state.get("policy_check", {}).get("passed", False)
        )
        verification = {
            "passed": not violations and not missing and not policy_failed,
            "hard_rule_violations": violations,
            "missing_outputs": missing,
            "policy_failed": policy_failed,
            "numeric_outputs_from_llm": False,
            "used_llm_steps": [e.step for e in state["trace"] if e.used_llm],
        }
        return verification, TraceEvent(
            step="F",
            actor=self.name,
            status="done" if verification["passed"] else "failed",
            message="结构校验通过，数值结果均来自 data/model/rule/knowledge 工具。"
            if verification["passed"]
            else "结构校验失败。",
            used_llm=False,
            metadata=verification,
        )


class ComposerAgent:
    """使用子智能体模板，将工具结果和证据组织为中文回答。"""

    @log_call("request", "state", "evidence")
    def compose(
        self,
        request: str,
        state: dict[str, Any],
        evidence: list[Evidence],
        token_callback: TokenCallback | None = None,
    ) -> tuple[str, TraceEvent]:
        """按结果类型准备结构化数据，调用子智能体 LLM 返回回答文本和 G 步骤记录。

        提示词限制数字必须来自上游结果；空回答抛出 LLMError，关键数字再由 Post-check 检查。
        """
        skill = state["skill"]
        sub_agent = state["sub_agent"]
        payload: dict[str, Any] = {
            "request": request,
            "result_type": skill.result_type,
            "context": state["context"],
            "evidence": [asdict(item) for item in evidence],
            "constraints": [
                "Do not invent numbers.",
                "Every number in answer must come from tool outputs or evidence.",
                "Keep action boundary as human confirmation; never claim direct DCS write.",
            ],
        }
        if skill.result_type == "tag_query":
            payload["current_values"] = state["current_values"]
        elif skill.result_type == "business_metric_analysis":
            payload["semantic_parse"] = state["semantic_parse"]
            payload["query_plan"] = state["query_plan"]
            payload["policy_check"] = state["policy_check"]
            payload["warehouse_result"] = state["warehouse_result"]
            payload["visualization"] = state["visualization"]
            payload["business_insights"] = state["business_insights"]
        else:
            forecast = state["forecast"]
            rule = state["rule_check"]
            payload["metrics"] = {
                "forecast_2h": forecast["forecast_2h"],
                "threshold": forecast["threshold"],
                "over_limit_probability": forecast["over_limit_probability"],
                "over_limit_probability_text": f"{forecast['over_limit_probability']:.0%}",
                "cooling_drop_rate": rule["cooling_drop_rate"],
                "cooling_drop_rate_text": f"{rule['cooling_drop_rate']:.1%}",
                "feed_temperature_delta": rule["feed_delta"],
            }
            payload["rule_check"] = rule
            payload["case_evidence"] = state["case_evidence"]
        system_prompt = sub_agent.prompt(
            "composer_system",
            (
                "You are G Composer in Zhuge AI. Convert structured industrial results into a concise "
                "Chinese answer. You may write prose, but you must not create new numeric facts. Return JSON only."
            ),
            skill=skill,
            state=state,
            evidence=evidence,
        )
        # 非流式入口继续要求 Composer 返回 JSON，便于 API 调用方拿到稳定结构。
        if token_callback is None:
            response = sub_agent.llm.chat_json(
                system=system_prompt,
                user=_json_dumps(payload),
                schema_hint={"answer": "string"},
                temperature=0.2,
            )
            answer = str(response.get("answer", "")).strip()
        else:
            # 流式入口只让最终自然语言文案走 token 流；上游工具、校验和数值仍保持确定性。
            answer = self._compose_streaming_answer(
                request=request,
                sub_agent=sub_agent,
                system_prompt=system_prompt,
                payload=payload,
                token_callback=token_callback,
            )
        if not answer:
            raise LLMError("Composer returned empty answer.")
        return answer, TraceEvent(
            step="G",
            actor=f"{sub_agent.name} / G Composer",
            status="done",
            message="调用 LLM 将结构化结果组织成业务可读文案，数值仍由上游工具提供。",
            used_llm=True,
            metadata={
                "sub_agent_id": sub_agent.spec.id,
                "template": skill.composer_template,
                "evidence_count": len(evidence),
                "streamed": token_callback is not None,
            },
        )

    def _compose_streaming_answer(
        self,
        *,
        request: str,
        sub_agent: SubAgent,
        system_prompt: object,
        payload: dict[str, Any],
        token_callback: TokenCallback,
    ) -> str:
        """调用自然语言流式接口生成 Composer 文案，并同步上报 token 事件。

        返回值仍是完整 answer，供 Post-check 校验关键数字是否来自工具输出。
        """
        user = (
            f"{_json_dumps(payload)}\n\n"
            "请直接输出最终中文回答，不要输出 JSON、Markdown 代码块或额外字段。"
            "不得编造新的业务数字；所有数字必须来自上面的结构化结果。"
        )
        chunks: list[str] = []
        for sequence, delta in enumerate(
            sub_agent.llm.stream_text(system=system_prompt, user=user, temperature=0.2),
            start=1,
        ):
            chunks.append(delta)
            # token_callback 只负责展示增量文本，不改变 state，也不绕过后续 Post-check。
            token_callback(
                AnswerTokenEvent(
                    request=request,
                    agent_id=sub_agent.spec.id,
                    agent_name=sub_agent.name,
                    delta=delta,
                    sequence=sequence,
                )
            )
        return "".join(chunks).strip()


class PostCheckAgent:
    """生成回答后检查工具提供的关键数字是否出现在文案中。"""

    name = "H Post-check"

    @log_call("answer", "state")
    def check(self, answer: str, state: dict[str, Any]) -> tuple[dict[str, Any], TraceEvent]:
        """按结果类型格式化上游关键数字，检查文本包含关系，返回后验结果和 H 记录。

        这是关键数字覆盖检查，不会扫描回答中的全部数字或验证所有自然语言结论。
        """
        if state["skill"].result_type == "tag_query":
            numbers = {
                item["name"]: str(item["value"])
                for item in state["current_values"].get("values", [])
            }
        elif state["skill"].result_type == "business_metric_analysis":
            totals = state["warehouse_result"]["totals"]
            rows = state["warehouse_result"]["rows"]
            numbers = {
                "total_sales": str(totals["sales_amount"]),
                "total_mom": f"{totals['mom_rate']:.1%}",
                **{row["city"]: str(row["sales_amount"]) for row in rows},
            }
        else:
            numbers = {
                "forecast_2h": str(state["forecast"]["forecast_2h"]),
                "cooling_drop_rate": f"{state['rule_check']['cooling_drop_rate']:.1%}",
                "over_limit_probability": f"{state['forecast']['over_limit_probability']:.0%}",
            }
        missing = [name for name, value in numbers.items() if value not in answer]
        post_check = {
            "passed": not missing,
            "checked_numbers": numbers,
            "missing_numbers": missing,
            "action_boundary": "仅生成分析结果，不直接写入业务系统或控制系统。",
        }
        return post_check, TraceEvent(
            step="H",
            actor=self.name,
            status="done" if post_check["passed"] else "failed",
            message="文案中的关键数字均可回溯到工具输出；动作边界保持为人工确认。"
            if post_check["passed"]
            else "文案缺少可回溯关键数字，需要重新生成或人工复核。",
            used_llm=False,
            metadata=post_check,
        )


class ZhugeAgentRuntime:
    """编排主智能体、子智能体、Skill 和工具，提供单任务、并行及进度流入口。"""

    def __init__(
        self,
        *,
        llm_client: LLMClient | None = None,
        case_data: dict[str, Any] | None = None,
        tools: ToolRegistry | None = None,
        agent_registry: AgentRegistry | None = None,
        skill_registry: SkillRegistry | None = None,
        workflow_registry: WorkflowRegistry | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        """装配注册表和执行链组件，可注入 LLM、案例数据、工具和各类注册表。

        未注入 LLM 时读取 config_path 或根目录 config.ini，真实或离线模式由配置显式选择。
        显式注入 llm_client 时跳过配置文件读取；注册表注入用于后端管理层接管配置来源。
        """
        if llm_client is None:
            log_config = load_logging_config(config_path)
            configure_logging(log_config.level, log_file=log_config.file)
        else:
            ensure_logging_configured()
        logger.info("初始化运行时 config_path=%s client=%s", config_path, type(llm_client).__name__)
        self.llm = llm_client if llm_client is not None else llm_client_from_config(config_path)
        self.agents = agent_registry or AgentRegistry()
        self.skills = skill_registry or SkillRegistry()
        self.workflows = workflow_registry or WorkflowRegistry()
        self.tools = tools or ToolRegistry.default(case_data)
        self.master = MasterAgent(self.llm)
        self.context_resolver = ContextResolverAgent(case_data)
        self.planner = PlannerAgent()
        self.executor = ExecutorAgent(self.tools)
        self.evidence_builder = EvidenceBuilderAgent()
        self.verifier = VerifierAgent()
        self.composer = ComposerAgent()
        self.post_check = PostCheckAgent()
        logger.info("运行时就绪 llm=%s agents=%s skills=%s workflows=%s", type(self.llm).__name__,
                    [agent.id for agent in self.agents.all()], [skill.id for skill in self.skills.all()],
                    [workflow.id for workflow in self.workflows.all()])

    @log_call("request", "session_id", context="request")
    def answer(
        self,
        request: str,
        progress_callback: ProgressCallback | None = None,
        token_callback: TokenCallback | None = None,
        *,
        session_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> StructuredAnswer:
        """由主智能体选择一个子智能体，完成执行和校验后返回 StructuredAnswer。

        request 是自然语言问题；progress_callback 在每个步骤记录后接收进度事件，
        所有已发出的事件也保存在结果的 progress 中，执行或校验异常直接向调用方传播。
        session_id 可由调用方传入；未提供时复用上层会话或自动生成，仅用于日志关联。
        """
        # log_call 已消费并绑定会话参数，执行链不需要另存全局会话状态。
        del session_id
        agent, intent, event = self.master.route(request, self.agents)
        progress: list[AgentProgressEvent] = []

        def emit(progress_event: AgentProgressEvent) -> None:
            """累积单任务进度，并同步通知调用方提供的回调。"""
            progress.append(progress_event)
            if progress_callback:
                progress_callback(progress_event)

        if agent is None:
            return self._answer_direct(
                request,
                intent=intent,
                route_event=event,
                progress_callback=emit,
                token_callback=token_callback,
                progress_events=progress,
            )

        return self._answer_with_agent(
            request,
            agent=agent,
            intent=intent,
            route_event=event,
            execution_context=_AgentExecutionContext(
                progress_callback=emit,
                token_callback=token_callback,
                progress_events=progress,
                context_override=context,
            ),
        )

    @log_call("request", "max_workers", "session_id", context="request")
    def answer_parallel(
        self,
        request: str,
        max_workers: int | None = None,
        progress_callback: ProgressCallback | None = None,
        token_callback: TokenCallback | None = None,
        *,
        session_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> ParallelStructuredAnswer:
        """由主智能体一次选择多个子智能体，通过线程池执行各自完整链路并汇总结果。

        max_workers 控制线程数，默认等于所选智能体数量；返回顺序保持路由顺序。
        progress_callback 可能由多个工作线程调用，调用方需要保证回调自身的线程安全。
        任一子任务异常都会向调用方传播，本接口不返回部分成功结果。
        session_id 绑定整次并行请求，各子线程自动继承并保持独立智能体身份。
        """
        del session_id
        agents, intent, event = self.master.route_many(request, self.agents)
        if not agents and intent.get("route_type") == "direct":
            progress: list[AgentProgressEvent] = []

            def emit_direct(progress_event: AgentProgressEvent) -> None:
                """累积直接回答进度，并同步通知调用方。"""
                progress.append(progress_event)
                if progress_callback:
                    progress_callback(progress_event)

            direct_answer = self._answer_direct(
                request,
                intent=intent,
                route_event=event,
                progress_callback=emit_direct,
                token_callback=token_callback,
                progress_events=progress,
            )
            return ParallelStructuredAnswer(
                request=request,
                answer=direct_answer.answer,
                sub_answers=[],
                trace=direct_answer.trace,
                verification={"passed": True, "sub_agent_count": 0, "agent_ids": [], "post_checks_passed": True},
                progress=direct_answer.progress,
            )
        workers = max_workers or len(agents)
        sub_answers: list[StructuredAnswer] = []
        progress: list[AgentProgressEvent] = []
        progress_lock = Lock()

        def emit(progress_event: AgentProgressEvent) -> None:
            """用锁保护共享进度列表，随后在当前工作线程中调用外部回调。"""
            with progress_lock:
                progress.append(progress_event)
            if progress_callback:
                progress_callback(progress_event)

        emit(self._progress_event(request, event, None))

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    copy_context().run,
                    self._answer_with_agent,
                    request,
                    agent=agent,
                    intent=self._intent_for_agent(intent, agent),
                    route_event=self._parallel_route_event(event, agent, index),
                    execution_context=_AgentExecutionContext(
                        progress_callback=emit,
                        token_callback=token_callback,
                        progress_events=[],
                        context_override=context,
                    ),
                ): index
                for index, agent in enumerate(agents, start=1)
            }
            completed = []
            for future in as_completed(futures):
                completed.append((futures[future], future.result()))
        sub_answers = [answer for _, answer in sorted(completed, key=lambda item: item[0])]
        trace = [event]
        for answer in sub_answers:
            trace.extend(answer.trace)

        verification = {
            "passed": all(answer.verification.get("passed") for answer in sub_answers),
            "sub_agent_count": len(sub_answers),
            "agent_ids": [answer.agent_id for answer in sub_answers],
            "post_checks_passed": all(
                answer.verification.get("post_check", {}).get("passed") for answer in sub_answers
            ),
        }
        return ParallelStructuredAnswer(
            request=request,
            answer=self._compose_parallel_answer(sub_answers),
            sub_answers=sub_answers,
            trace=trace,
            verification=verification,
            progress=progress,
        )

    def answer_parallel_as_dict(
        self,
        request: str,
        max_workers: int | None = None,
        *,
        session_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """执行并行问答并递归转换数据类为字典，便于 JSON 序列化或 API 输出。"""
        return asdict(self.answer_parallel(
            request,
            max_workers=max_workers,
            session_id=session_id,
            context=context,
        ))

    @log_call("request", "max_workers", "session_id", context="request")
    def answer_auto(
        self,
        request: str,
        max_workers: int | None = None,
        progress_callback: ProgressCallback | None = None,
        token_callback: TokenCallback | None = None,
        *,
        session_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> StructuredAnswer | ParallelStructuredAnswer | WorkflowStructuredAnswer:
        """由 MasterAgent 先判断 direct/single/parallel/workflow，再委派到对应运行入口。

        auto 是面向产品层的默认入口；显式指定 single/parallel/workflow 时仍可绕过本规划步骤。
        """
        plan, event = self.master.plan_execution(request, self.agents, self.workflows)
        logger.info("自动编排选择 mode=%s metadata=%s", plan["execution_mode"], log_value(event.metadata))
        if plan["execution_mode"] in {"direct", "single"}:
            return self.answer(
                request,
                progress_callback=progress_callback,
                token_callback=token_callback,
                session_id=session_id,
                context=context,
            )
        if plan["execution_mode"] == "parallel":
            return self.answer_parallel(
                request,
                max_workers=max_workers,
                progress_callback=progress_callback,
                token_callback=token_callback,
                session_id=session_id,
                context=context,
            )
        return self.answer_workflow(
            request,
            workflow_id=plan.get("workflow_id") or None,
            max_workers=max_workers,
            progress_callback=progress_callback,
            token_callback=token_callback,
            session_id=session_id,
            context=context,
        )

    @log_call("request", "workflow_id", "max_workers", "session_id", context="request")
    def answer_workflow(
        self,
        request: str,
        *,
        workflow_id: str | None = None,
        max_workers: int | None = None,
        progress_callback: ProgressCallback | None = None,
        token_callback: TokenCallback | None = None,
        session_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> WorkflowStructuredAnswer:
        """执行 graph-based workflow，支持 agent、human_review 和 join 节点。

        workflow_id 为空时按请求文本匹配最合适工作流；agent 节点仍独立经过 Skill Router、
        Planner、Executor、Verifier、Composer 和 Post-check。session_id 仅由日志上下文消费。
        """
        del session_id
        workflow = self.workflows.get(workflow_id) if workflow_id else self.workflows.route(request)
        progress: list[AgentProgressEvent] = []
        progress_lock = Lock()

        def emit(progress_event: AgentProgressEvent) -> None:
            """线程安全记录 workflow 及子智能体进度，并同步转发给调用方。"""
            with progress_lock:
                progress.append(progress_event)
            if progress_callback:
                progress_callback(progress_event)

        trace: list[TraceEvent] = []
        node_outputs, sub_answers = self._execute_workflow_graph(
            request=request,
            workflow=workflow,
            max_workers=max_workers,
            progress_callback=emit,
            token_callback=token_callback,
            trace=trace,
            context_override=context,
        )
        verification = {
            "passed": all(answer.verification.get("passed") for answer in sub_answers)
            and all(output.get("passed", True) for output in node_outputs.values() if isinstance(output, dict)),
            "workflow_id": workflow.id,
            "node_count": len(workflow.nodes),
            "agent_node_count": len(sub_answers),
            "agent_ids": [answer.agent_id for answer in sub_answers],
            "human_review_passed": all(
                output.get("passed", True)
                for output in node_outputs.values()
                if isinstance(output, dict) and output.get("node_type") == "human_review"
            ),
        }
        return WorkflowStructuredAnswer(
            request=request,
            workflow_id=workflow.id,
            workflow_name=workflow.name,
            answer=self._compose_workflow_answer(workflow, sub_answers, node_outputs),
            sub_answers=sub_answers,
            node_outputs=node_outputs,
            trace=trace,
            verification=verification,
            progress=progress,
        )

    def answer_workflow_as_dict(
        self,
        request: str,
        workflow_id: str | None = None,
        max_workers: int | None = None,
        *,
        session_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """执行 graph workflow 并递归转换为字典，便于 API 或 CLI 输出。"""
        return asdict(
            self.answer_workflow(
                request,
                workflow_id=workflow_id,
                max_workers=max_workers,
                session_id=session_id,
                context=context,
            )
        )

    @log_call("run_request", context="request")
    def answer_request(
        self,
        run_request: AgentRunRequest,
        progress_callback: ProgressCallback | None = None,
        token_callback: TokenCallback | None = None,
    ) -> StructuredAnswer | ParallelStructuredAnswer | WorkflowStructuredAnswer:
        """执行后端传入的结构化请求，并根据 mode 选择单链路、并行或 workflow 入口。

        Runtime 只消费后端已经准备好的上下文，不在这里做用户鉴权或租户权限判断。
        """
        context = self._context_from_run_request(run_request)
        if run_request.mode == "auto":
            return self.answer_auto(
                run_request.request,
                max_workers=run_request.max_workers,
                progress_callback=progress_callback,
                token_callback=token_callback,
                session_id=run_request.session_id,
                context=context,
            )
        if run_request.mode == "single":
            return self.answer(
                run_request.request,
                progress_callback=progress_callback,
                token_callback=token_callback,
                session_id=run_request.session_id,
                context=context,
            )
        if run_request.mode == "parallel":
            return self.answer_parallel(
                run_request.request,
                max_workers=run_request.max_workers,
                progress_callback=progress_callback,
                token_callback=token_callback,
                session_id=run_request.session_id,
                context=context,
            )
        if run_request.mode == "workflow":
            return self.answer_workflow(
                run_request.request,
                workflow_id=run_request.workflow_id,
                max_workers=run_request.max_workers,
                progress_callback=progress_callback,
                token_callback=token_callback,
                session_id=run_request.session_id,
                context=context,
            )
        raise ValueError("AgentRunRequest.mode 必须为 auto、single、parallel 或 workflow。")

    def iter_run_events(self, run_request: AgentRunRequest) -> Iterator[RuntimeEvent]:
        """以统一事件协议流式执行结构化请求，供后端直接映射 SSE 或 WebSocket。

        本入口不抛出业务异常；异常会转换为 type=error 的 RuntimeEvent，便于 API 层稳定收口。
        """
        queue: Queue[RuntimeEvent | object] = Queue()
        sentinel = object()
        sequence = 0
        sequence_lock = Lock()

        def next_event(event_type: str, payload: Any) -> RuntimeEvent:
            """生成单调递增序号的统一事件，保持同一 run 内事件顺序可追踪。"""
            nonlocal sequence
            with sequence_lock:
                sequence += 1
                current = sequence
            return RuntimeEvent(
                type=event_type,
                payload=payload,
                sequence=current,
                session_id=run_request.session_id,
                run_mode=run_request.mode,
                metadata=run_request.metadata,
            )

        def emit_progress(progress_event: AgentProgressEvent) -> None:
            """把内部进度事件包装成 runtime progress 事件。"""
            queue.put(next_event("progress", progress_event))

        def emit_token(token_event: AnswerTokenEvent) -> None:
            """把内部 token 事件包装成 runtime token 事件。"""
            queue.put(next_event("token", token_event))

        def run() -> None:
            """后台执行结构化请求，结束时输出 final 或 error 事件。"""
            try:
                result = self.answer_request(
                    run_request,
                    progress_callback=emit_progress,
                    token_callback=emit_token,
                )
                queue.put(next_event("final", result))
            except Exception as exc:  # pylint: disable=broad-exception-caught
                queue.put(next_event("error", self._error_info(exc)))
            finally:
                queue.put(sentinel)

        inherited_context = copy_context()
        Thread(target=inherited_context.run, args=(run,), daemon=True).start()
        while True:
            item = queue.get()
            if item is sentinel:
                break
            yield item

    def iter_answer(
        self, request: str, *, session_id: str | None = None
    ) -> Iterator[AgentProgressEvent | AnswerTokenEvent | StructuredAnswer]:
        """在后台执行单任务，通过队列依次 yield 进度和最终回答 token，最后 yield 完整结果。

        Router/Planner 仍是结构化完整调用；只有直接回答或 Composer 文案会产生 token 事件。
        后台异常在消费迭代器时重新抛出。
        """
        queue: Queue[AgentProgressEvent | AnswerTokenEvent | StructuredAnswer | Exception | object] = Queue()
        sentinel = object()

        def emit(progress_event: AgentProgressEvent) -> None:
            """把单任务回调事件送入队列，供调用线程逐条消费。"""
            queue.put(progress_event)

        def emit_token(token_event: AnswerTokenEvent) -> None:
            """把最终回答 token 事件送入队列，供调用线程实时展示。"""
            queue.put(token_event)

        def run() -> None:
            """后台执行问答并传递结果或异常，最终始终发送结束标记。"""
            try:
                queue.put(self.answer(
                    request,
                    progress_callback=emit,
                    token_callback=emit_token,
                    session_id=session_id,
                ))
            except Exception as exc:  # pylint: disable=broad-exception-caught
                queue.put(exc)
            finally:
                queue.put(sentinel)

        inherited_context = copy_context()
        Thread(target=inherited_context.run, args=(run,), daemon=True).start()
        while True:
            item = queue.get()
            if item is sentinel:
                break
            if isinstance(item, Exception):
                raise item
            yield item

    def iter_answer_parallel(
        self,
        request: str,
        max_workers: int | None = None,
        *,
        session_id: str | None = None,
    ) -> Iterator[AgentProgressEvent | AnswerTokenEvent | ParallelStructuredAnswer]:
        """逐条 yield 多个子任务的进度和最终回答 token，全部完成后 yield 汇总结果。

        进度顺序取决于并发执行，最终子回答保持路由顺序；后台异常传播给迭代器消费者。
        """
        queue: Queue[AgentProgressEvent | AnswerTokenEvent | ParallelStructuredAnswer | Exception | object] = Queue()
        sentinel = object()

        def emit(progress_event: AgentProgressEvent) -> None:
            """将各工作线程上报的进度写入线程安全队列。"""
            queue.put(progress_event)

        def emit_token(token_event: AnswerTokenEvent) -> None:
            """将各工作线程上报的最终回答 token 写入线程安全队列。"""
            queue.put(token_event)

        def run() -> None:
            """后台执行并行问答，向队列传递最终结果或异常并发送结束标记。"""
            try:
                queue.put(self.answer_parallel(
                    request,
                    max_workers=max_workers,
                    progress_callback=emit,
                    token_callback=emit_token,
                    session_id=session_id,
                ))
            except Exception as exc:  # pylint: disable=broad-exception-caught
                queue.put(exc)
            finally:
                queue.put(sentinel)

        inherited_context = copy_context()
        Thread(target=inherited_context.run, args=(run,), daemon=True).start()
        while True:
            item = queue.get()
            if item is sentinel:
                break
            if isinstance(item, Exception):
                raise item
            yield item

    def iter_answer_workflow(
        self,
        request: str,
        workflow_id: str | None = None,
        max_workers: int | None = None,
        *,
        session_id: str | None = None,
    ) -> Iterator[AgentProgressEvent | AnswerTokenEvent | WorkflowStructuredAnswer]:
        """逐条 yield workflow 图节点进度、子智能体 token，全部完成后 yield 汇总结果。"""
        queue: Queue[AgentProgressEvent | AnswerTokenEvent | WorkflowStructuredAnswer | Exception | object] = Queue()
        sentinel = object()

        def emit(progress_event: AgentProgressEvent) -> None:
            """将 workflow 或子智能体进度写入线程安全队列。"""
            queue.put(progress_event)

        def emit_token(token_event: AnswerTokenEvent) -> None:
            """将 agent 节点产生的最终回答 token 写入线程安全队列。"""
            queue.put(token_event)

        def run() -> None:
            """后台执行 workflow，向队列传递最终结果或异常并发送结束标记。"""
            try:
                queue.put(self.answer_workflow(
                    request,
                    workflow_id=workflow_id,
                    max_workers=max_workers,
                    progress_callback=emit,
                    token_callback=emit_token,
                    session_id=session_id,
                ))
            except Exception as exc:  # pylint: disable=broad-exception-caught
                queue.put(exc)
            finally:
                queue.put(sentinel)

        inherited_context = copy_context()
        Thread(target=inherited_context.run, args=(run,), daemon=True).start()
        while True:
            item = queue.get()
            if item is sentinel:
                break
            if isinstance(item, Exception):
                raise item
            yield item

    @log_call("request", "intent")
    def _answer_direct(
        self,
        request: str,
        *,
        intent: dict[str, Any],
        route_event: TraceEvent,
        progress_callback: ProgressCallback | None = None,
        token_callback: TokenCallback | None = None,
        progress_events: list[AgentProgressEvent] | None = None,
        context_override: dict[str, Any] | None = None,
    ) -> StructuredAnswer:
        """由 MasterAgent 直接回答普通问候或通用问题，不进入任何子智能体或 Skill 链路。"""
        # 直接回答不进入 ContextResolver，但保留参数以便入口层可统一传递结构化请求。
        del context_override
        state: dict[str, Any] = {"request": request, "trace": []}
        self._record_event(state, route_event, request=request, agent=None, progress_callback=progress_callback)
        answer = self._direct_answer_text(request, intent, token_callback)
        verification = {
            "passed": True,
            "direct_answer": True,
            "post_check": {"passed": True, "direct_answer": True},
        }
        return StructuredAnswer(
            request=request,
            agent_id="MasterAgent",
            agent_name="MasterAgent",
            intent=str(intent.get("intent") or "通用问答"),
            skill_id="direct_answer",
            answer=answer,
            metrics={},
            evidence=[],
            actions=[],
            trace=state["trace"],
            verification=verification,
            progress=progress_events or [],
        )

    def _direct_answer_text(
        self,
        request: str,
        intent: dict[str, Any],
        token_callback: TokenCallback | None,
    ) -> str:
        """生成 MasterAgent 直接回答；流式入口使用 token callback 输出增量文本。

        非流式调用复用 Router 返回的 direct_answer，避免普通 answer() 额外触发一次 LLM。
        """
        fallback = str(intent.get("direct_answer") or "你好，我在。有什么想一起看看的吗？")
        if token_callback is None:
            return fallback
        system = (
            "You are MasterAgent in Zhuge AI. Answer the user's general small-talk or general question "
            "directly in concise Chinese. Do not call tools, do not mention sub-agents, and do not invent "
            "business, industrial, or numeric facts."
        )
        user = (
            f"用户问题：{request}\n"
            f"路由意图：{intent.get('intent')}\n"
            f"路由原因：{intent.get('reason')}\n"
            "请直接输出最终回答文本，不要输出 JSON。"
        )
        chunks: list[str] = []
        for sequence, delta in enumerate(self.llm.stream_text(system=system, user=user, temperature=0.3), start=1):
            chunks.append(delta)
            # 直接回答没有子智能体身份，因此 token 事件归属 MasterAgent。
            token_callback(
                AnswerTokenEvent(
                    request=request,
                    agent_id="MasterAgent",
                    agent_name="MasterAgent",
                    delta=delta,
                    sequence=sequence,
                )
            )
        answer = "".join(chunks).strip()
        return answer or fallback

    @log_call("request", "agent", "intent", context="agent")
    def _answer_with_agent(
        self,
        request: str,
        *,
        agent: AgentSpec,
        intent: dict[str, Any],
        route_event: TraceEvent,
        execution_context: _AgentExecutionContext | None = None,
    ) -> StructuredAnswer:
        """执行一个已委派子智能体的完整链路，供单任务和并行入口复用。

        依次完成 Skill 选择、上下文、计划、工具执行、证据汇总、校验、回答及后验检查。
        agent、intent 和 route_event 来自主路由；execution_context 承载回调、进度列表和后端上下文。
        结构或后验检查失败时抛出 HardRuleViolation，不返回未通过校验的回答。
        """
        execution_context = execution_context or _AgentExecutionContext()
        state: dict[str, Any] = {"request": request, "trace": []}
        step_results: list[StepResult] = []

        sub_agent = SubAgent(agent, self.llm)
        state["sub_agent"] = sub_agent
        state["intent"] = intent
        self._record_event(
            state,
            route_event,
            request=request,
            agent=agent,
            progress_callback=execution_context.progress_callback,
        )

        skill, event = sub_agent.select_skill(request, self.skills)
        state["skill"] = skill
        self._record_event(
            state,
            event,
            request=request,
            agent=agent,
            progress_callback=execution_context.progress_callback,
        )

        context, event = self.context_resolver.run(
            request,
            context_override=execution_context.context_override,
        )
        state["context"] = context
        self._record_event(
            state,
            event,
            request=request,
            agent=agent,
            progress_callback=execution_context.progress_callback,
        )

        self._record_event(
            state,
            self.planner.run(sub_agent, skill, context),
            request=request,
            agent=agent,
            progress_callback=execution_context.progress_callback,
        )

        for step in skill.steps:
            if not step.tool_id:
                continue
            inputs = {key: state[key] for key in step.input_keys if key in state}
            result = self.executor.run(step.id, step.tool_id, inputs, step.no_llm)
            result.output_key = step.output_key
            state[step.output_key] = result.data
            step_results.append(result)
            self._record_event(
                state,
                TraceEvent(
                    step=step.id,
                    actor=result.actor,
                    status="done",
                    message=f"{step.name} 完成，输出 {step.output_key}。",
                    used_llm=result.used_llm,
                    metadata={"kind": result.kind.value},
                ),
                request=request,
                agent=agent,
                progress_callback=execution_context.progress_callback,
            )

        evidence, event = self.evidence_builder.collect(step_results)
        state["evidence"] = evidence
        self._record_event(
            state,
            event,
            request=request,
            agent=agent,
            progress_callback=execution_context.progress_callback,
        )

        verification, event = self.verifier.verify(state, skill)
        state["verification"] = verification
        self._record_event(
            state,
            event,
            request=request,
            agent=agent,
            progress_callback=execution_context.progress_callback,
        )
        if not verification["passed"]:
            raise HardRuleViolation(str(verification))

        answer, event = self.composer.compose(
            request,
            state,
            evidence,
            token_callback=execution_context.token_callback,
        )
        state["answer"] = answer
        self._record_event(
            state,
            event,
            request=request,
            agent=agent,
            progress_callback=execution_context.progress_callback,
        )

        post_check, event = self.post_check.check(answer, state)
        state["post_check"] = post_check
        self._record_event(
            state,
            event,
            request=request,
            agent=agent,
            progress_callback=execution_context.progress_callback,
        )
        if not post_check["passed"]:
            raise HardRuleViolation(str(post_check))

        metrics = self._metrics_for(state)
        actions = state.get("rule_check", {}).get("actions", [])
        return StructuredAnswer(
            request=request,
            agent_id=agent.id,
            agent_name=agent.name,
            intent=intent["intent"],
            skill_id=skill.id,
            answer=answer,
            metrics=metrics,
            evidence=evidence,
            actions=actions,
            trace=state["trace"],
            verification={**verification, "post_check": post_check},
            progress=execution_context.progress_events or [],
        )

    def answer_as_dict(
        self,
        request: str,
        *,
        session_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """执行单任务问答并转换为字典，保留指标、证据、执行记录、校验和进度字段。"""
        return asdict(self.answer(request, session_id=session_id, context=context))

    def _execute_workflow_graph(
        self,
        *,
        request: str,
        workflow: WorkflowSpec,
        max_workers: int | None,
        progress_callback: ProgressCallback,
        token_callback: TokenCallback | None,
        trace: list[TraceEvent],
        context_override: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], list[StructuredAnswer]]:
        """按 DAG 入度逐层执行 workflow 节点，支持同一层节点并发运行。"""
        node_by_id = {node.id: node for node in workflow.nodes}
        outgoing = {node.id: [] for node in workflow.nodes}
        indegree = {node.id: 0 for node in workflow.nodes}
        for edge in workflow.edges:
            outgoing[edge.from_node].append(edge)
            indegree[edge.to_node] += 1

        self._record_workflow_event(
            trace,
            TraceEvent(
                step="W0",
                actor="WorkflowRuntime / Graph",
                status="done",
                message=f"启动工作流「{workflow.name}」，节点数 {len(workflow.nodes)}。",
                used_llm=False,
                metadata={"workflow_id": workflow.id, "node_count": len(workflow.nodes)},
            ),
            request=request,
            workflow=workflow,
            progress_callback=progress_callback,
        )
        ready = [node for node in workflow.nodes if indegree[node.id] == 0]
        completed: set[str] = set()
        node_outputs: dict[str, Any] = {}
        sub_answers_by_node: dict[str, StructuredAnswer] = {}
        workers = max_workers or max(1, len(workflow.nodes))

        with ThreadPoolExecutor(max_workers=workers) as executor:
            while ready:
                futures = {
                    executor.submit(
                        copy_context().run,
                        self._run_workflow_node,
                        request=request,
                        workflow=workflow,
                        node=node,
                        progress_callback=progress_callback,
                        token_callback=token_callback,
                        node_outputs=node_outputs,
                        context_override=context_override,
                    ): node
                    for node in ready
                }
                ready = []
                for future in as_completed(futures):
                    node = futures[future]
                    output, node_trace, answer = future.result()
                    completed.add(node.id)
                    node_outputs[node.id] = output
                    if answer is not None:
                        sub_answers_by_node[node.id] = answer
                    trace.extend(node_trace)
                    for edge in outgoing[node.id]:
                        if not self._workflow_edge_passed(edge.condition, output):
                            raise HardRuleViolation(
                                f"Workflow {workflow.id} edge {edge.from_node}->{edge.to_node} condition failed."
                            )
                        indegree[edge.to_node] -= 1
                        if indegree[edge.to_node] == 0:
                            ready.append(node_by_id[edge.to_node])

        if len(completed) != len(workflow.nodes):
            missing = sorted(set(node_by_id) - completed)
            raise HardRuleViolation(f"Workflow {workflow.id} did not complete nodes: {missing}")
        ordered_answers = [
            sub_answers_by_node[node.id]
            for node in workflow.nodes
            if node.id in sub_answers_by_node
        ]
        return node_outputs, ordered_answers

    def _run_workflow_node(
        self,
        *,
        request: str,
        workflow: WorkflowSpec,
        node: WorkflowNode,
        progress_callback: ProgressCallback,
        token_callback: TokenCallback | None,
        node_outputs: dict[str, Any],
        context_override: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], list[TraceEvent], StructuredAnswer | None]:
        """执行单个 workflow 节点，返回节点输出、节点 trace 和可选子智能体回答。"""
        if node.type == "agent":
            return self._run_workflow_agent_node(
                request=request,
                workflow=workflow,
                node=node,
                progress_callback=progress_callback,
                token_callback=token_callback,
                context_override=context_override,
            )
        if node.type == "human_review":
            return self._run_workflow_human_review_node(
                request=request,
                workflow=workflow,
                node=node,
                progress_callback=progress_callback,
                node_outputs=node_outputs,
            )
        if node.type == "join":
            return self._run_workflow_join_node(
                request=request,
                workflow=workflow,
                node=node,
                progress_callback=progress_callback,
                node_outputs=node_outputs,
            )
        raise HardRuleViolation(f"Unsupported workflow node type: {node.type}")

    def _run_workflow_agent_node(
        self,
        *,
        request: str,
        workflow: WorkflowSpec,
        node: WorkflowNode,
        progress_callback: ProgressCallback,
        token_callback: TokenCallback | None,
        context_override: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], list[TraceEvent], StructuredAnswer]:
        """执行 agent 节点，复用现有子智能体链路以保留 Skill 可见性和 no_llm 校验。"""
        agent = self.agents.get(str(node.agent_id))
        route_event = TraceEvent(
            step=f"W:{node.id}",
            actor="WorkflowRuntime / Agent Node",
            status="done",
            message=f"工作流节点「{node.name}」委派给「{agent.name}」。",
            used_llm=False,
            metadata={"workflow_id": workflow.id, "node_id": node.id, "agent_id": agent.id},
        )
        answer = self._answer_with_agent(
            request,
            agent=agent,
            intent={
                "agent_id": agent.id,
                "intent": node.name,
                "confidence": 1.0,
                "reason": f"由工作流 {workflow.id} 显式编排。",
            },
            route_event=route_event,
            execution_context=_AgentExecutionContext(
                progress_callback=progress_callback,
                token_callback=token_callback,
                progress_events=[],
                context_override=context_override,
            ),
        )
        return {
            "node_type": node.type,
            "agent_id": agent.id,
            "skill_id": answer.skill_id,
            "passed": answer.verification.get("passed", False),
        }, answer.trace, answer

    def _run_workflow_human_review_node(
        self,
        *,
        request: str,
        workflow: WorkflowSpec,
        node: WorkflowNode,
        progress_callback: ProgressCallback,
        node_outputs: dict[str, Any],
    ) -> tuple[dict[str, Any], list[TraceEvent], None]:
        """执行人审检查点；demo 默认 auto_approve，后续可替换为真实审批系统回调。"""
        mode = node.config.get("mode", "auto_approve")
        passed = mode == "auto_approve"
        output = {
            "node_type": node.type,
            "passed": passed,
            "mode": mode,
            "review_focus": node.config.get("review_focus", []),
            "reviewed_node_ids": sorted(node_outputs),
        }
        event = TraceEvent(
            step=f"W:{node.id}",
            actor="WorkflowRuntime / Human Review",
            status="done" if passed else "failed",
            message=f"人审检查点「{node.name}」{'自动通过' if passed else '等待人工处理'}。",
            used_llm=False,
            metadata={"workflow_id": workflow.id, "node_id": node.id, **output},
        )
        self._emit_workflow_progress(request, workflow, event, progress_callback)
        return output, [event], None

    def _run_workflow_join_node(
        self,
        *,
        request: str,
        workflow: WorkflowSpec,
        node: WorkflowNode,
        progress_callback: ProgressCallback,
        node_outputs: dict[str, Any],
    ) -> tuple[dict[str, Any], list[TraceEvent], None]:
        """执行汇总节点，记录可用上游节点数量；最终文案由 WorkflowStructuredAnswer 组装。"""
        output = {
            "node_type": node.type,
            "passed": True,
            "joined_node_ids": sorted(node_outputs),
        }
        event = TraceEvent(
            step=f"W:{node.id}",
            actor="WorkflowRuntime / Join",
            status="done",
            message=f"汇总节点「{node.name}」已收集 {len(node_outputs)} 个上游节点输出。",
            used_llm=False,
            metadata={"workflow_id": workflow.id, "node_id": node.id, **output},
        )
        self._emit_workflow_progress(request, workflow, event, progress_callback)
        return output, [event], None

    def _intent_for_agent(self, intent: dict[str, Any], agent: AgentSpec) -> dict[str, Any]:
        """从并行主意图构建单个子任务的意图字典，绑定对应智能体 ID。"""
        return {
            "agent_id": agent.id,
            "intent": intent.get("intent") or agent.name,
            "confidence": float(intent.get("confidence", 0.0)),
            "reason": intent.get("reason", ""),
        }

    def _parallel_route_event(self, event: TraceEvent, agent: AgentSpec, index: int) -> TraceEvent:
        """为子任务创建带 A1 序号的委派记录，保留其与主路由步骤的关联。"""
        return TraceEvent(
            step=f"A1.{index}",
            actor="MasterAgent / Parallel Router",
            status="done",
            message=f"并行子任务 {index} 委派给「{agent.name}」。",
            used_llm=event.used_llm,
            metadata={
                "agent_id": agent.id,
                "sub_agent_name": agent.name,
                "parallel_parent_step": event.step,
            },
        )

    def _compose_parallel_answer(self, sub_answers: list[StructuredAnswer]) -> str:
        """按子回答顺序添加智能体名称并拼接文案，复用已校验内容而不再次调用 LLM。"""
        return "\n\n".join(f"【{answer.agent_name}】{answer.answer}" for answer in sub_answers)

    def _compose_workflow_answer(
        self,
        workflow: WorkflowSpec,
        sub_answers: list[StructuredAnswer],
        node_outputs: dict[str, Any],
    ) -> str:
        """按 workflow 节点顺序拼接已校验子回答，并附加人审检查点状态。"""
        answer_text = "\n\n".join(f"【{answer.agent_name}】{answer.answer}" for answer in sub_answers)
        review_outputs = [
            output
            for output in node_outputs.values()
            if isinstance(output, dict) and output.get("node_type") == "human_review"
        ]
        if review_outputs:
            review_state = "；".join(
                f"{item.get('mode')}:{'通过' if item.get('passed') else '未通过'}"
                for item in review_outputs
            )
            return f"{answer_text}\n\n【{workflow.name}】人审检查点：{review_state}。"
        return answer_text

    def _record_workflow_event(
        self,
        trace: list[TraceEvent],
        event: TraceEvent,
        *,
        request: str,
        workflow: WorkflowSpec,
        progress_callback: ProgressCallback,
    ) -> None:
        """记录 workflow 级事件，并转换为前端进度事件。"""
        trace.append(event)
        self._emit_workflow_progress(request, workflow, event, progress_callback)

    def _emit_workflow_progress(
        self,
        request: str,
        workflow: WorkflowSpec,
        event: TraceEvent,
        progress_callback: ProgressCallback,
    ) -> None:
        """把 workflow 级 TraceEvent 转为 AgentProgressEvent，身份字段绑定 workflow。"""
        logger.log(logging.INFO if event.status == "done" else logging.WARNING,
                   "Workflow 进度 step=%s actor=%s status=%s metadata=%s", event.step, event.actor,
                   event.status, log_value(event.metadata))
        progress_callback(
            AgentProgressEvent(
                request=request,
                agent_id=workflow.id,
                agent_name=workflow.name,
                step=event.step,
                actor=event.actor,
                status=event.status,
                message=event.message,
                used_llm=event.used_llm,
                metadata=event.metadata,
            )
        )

    def _workflow_edge_passed(self, condition: str, output: dict[str, Any]) -> bool:
        """检查 workflow 边条件；当前支持 always、passed 和 failed 三种确定性条件。"""
        if condition in {"", "always"}:
            return True
        if condition == "passed":
            return bool(output.get("passed"))
        if condition == "failed":
            return not bool(output.get("passed"))
        raise HardRuleViolation(f"Unsupported workflow edge condition: {condition}")

    def _record_event(
        self,
        state: dict[str, Any],
        event: TraceEvent,
        *,
        request: str,
        agent: AgentSpec | None,
        progress_callback: ProgressCallback | None,
    ) -> None:
        """把步骤记录加入当前任务 trace，并在有回调时转换和上报进度事件。"""
        state["trace"].append(event)
        logger.log(logging.INFO if event.status == "done" else logging.WARNING,
                   "步骤进度 step=%s actor=%s status=%s metadata=%s", event.step, event.actor,
                   event.status, log_value(event.metadata))
        if progress_callback:
            progress_callback(self._progress_event(request, event, agent))

    def _progress_event(
        self,
        request: str,
        event: TraceEvent,
        agent: AgentSpec | None,
    ) -> AgentProgressEvent:
        """将执行记录转换为前端进度事件；主路由没有具体子智能体时保留空身份字段。"""
        return AgentProgressEvent(
            request=request,
            agent_id=agent.id if agent else None,
            agent_name=agent.name if agent else None,
            step=event.step,
            actor=event.actor,
            status=event.status,
            message=event.message,
            used_llm=event.used_llm,
            metadata=event.metadata,
        )

    def _context_from_run_request(self, run_request: AgentRunRequest) -> dict[str, Any]:
        """从结构化运行请求提取 Runtime 可消费的上下文，不执行鉴权和权限推断。

        API/后端层可以把租户、用户、角色和数据域放进 AgentRunRequest；Runtime 只把这些字段
        透传给 ContextResolver 和后续工具，真正的权限校验仍应由确定性 Tool 或后端系统完成。
        """
        context = dict(run_request.context)
        if run_request.tenant_id is not None:
            context["tenant_id"] = run_request.tenant_id
        if run_request.user_id is not None:
            context["user_id"] = run_request.user_id
        if run_request.role is not None:
            context["role"] = run_request.role
        if run_request.data_scope:
            context["data_scope"] = dict(run_request.data_scope)
        conversation = dict(context.get("conversation") or {})
        if run_request.recent_messages:
            conversation["recent_messages"] = list(run_request.recent_messages)
        if run_request.conversation_summary is not None:
            conversation["summary"] = run_request.conversation_summary
        if run_request.retrieved_memories:
            conversation["retrieved_memories"] = [
                asdict(memory) for memory in run_request.retrieved_memories
            ]
        if conversation:
            context["conversation"] = conversation
        return context

    def _error_info(self, exc: Exception) -> RuntimeErrorInfo:
        """把内部异常转换成稳定错误协议，供 API 层直接返回或映射为事件。

        这里不记录敏感入参；具体异常仍会由 log_call 在调用栈入口按脱敏规则记录。
        """
        if isinstance(exc, HardRuleViolation):
            return RuntimeErrorInfo(code="HARD_RULE_VIOLATION", message=str(exc), retryable=False)
        if isinstance(exc, LLMError):
            return RuntimeErrorInfo(code="LLM_ERROR", message=str(exc), retryable=True)
        if isinstance(exc, KeyError):
            return RuntimeErrorInfo(code="CONFIG_NOT_FOUND", message=str(exc), retryable=False)
        if isinstance(exc, ValueError):
            return RuntimeErrorInfo(code="INVALID_REQUEST", message=str(exc), retryable=False)
        return RuntimeErrorInfo(code="RUNTIME_ERROR", message=str(exc), retryable=False)

    def _metrics_for(self, state: dict[str, Any]) -> dict[str, Any]:
        """按 Skill 结果类型提取测点、经营统计或诊断指标，数值全部复用执行状态中的工具输出。"""
        if state["skill"].result_type == "tag_query":
            return {
                item["name"]: {"value": item["value"], "unit": item["unit"], "tag": item["tag"]}
                for item in state["current_values"].get("values", [])
            }
        if state["skill"].result_type == "business_metric_analysis":
            return {
                "totals": state["warehouse_result"]["totals"],
                "rows": state["warehouse_result"]["rows"],
                "visualization": state["visualization"],
                "insights": state["business_insights"]["insights"],
            }
        return {
            "forecast_2h": state["forecast"]["forecast_2h"],
            "threshold": state["forecast"]["threshold"],
            "over_limit_probability": state["forecast"]["over_limit_probability"],
            "cooling_drop_rate": state["rule_check"]["cooling_drop_rate"],
            "feed_temperature_delta": state["rule_check"]["feed_delta"],
        }


def _json_dumps(payload: dict[str, Any]) -> str:
    """将结构化 LLM 请求编码为紧凑 JSON，保留中文字符以提高提示词可读性。"""
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
