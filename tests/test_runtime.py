from __future__ import annotations

import json
import unittest
from pathlib import Path

from zhuge_agent.llm import FixtureLLMClient
from zhuge_agent.runtime import ZhugeAgentRuntime
from zhuge_agent.schema import (
    AgentProgressEvent,
    AgentRunRequest,
    AnswerTokenEvent,
    MemoryItem,
    ParallelStructuredAnswer,
    RuntimeErrorInfo,
    RuntimeEvent,
    StructuredAnswer,
    WorkflowStructuredAnswer,
)
from zhuge_agent.utils.prompt_templates import LazyPrompt


class InvalidRoutingLLM(FixtureLLMClient):
    """模拟真实 LLM 路由漂移：返回不存在的子智能体 ID，用于验证 MasterAgent 安全兜底。"""

    def __init__(self) -> None:
        """记录路由调用次数，便于断言非法目标会触发一次重试。"""
        self.route_calls = 0

    def chat_json(self, **kwargs):
        """在 MasterAgent 路由阶段返回非法 agent，其余阶段复用 fixture 行为。"""
        schema_hint = kwargs.get("schema_hint", {})
        if "execution_mode" in schema_hint:
            self.route_calls += 1
            return {
                "execution_mode": "single",
                "agent_ids": ["unknown_agent"],
                "workflow_id": "",
                "intent": "未知专业问题",
                "confidence": 0.42,
                "reason": "模拟模型误选了未注册子智能体。",
            }
        if "agent_id" in schema_hint:
            self.route_calls += 1
            return {
                "route_type": "delegate",
                "agent_id": "unknown_agent",
                "intent": "未知专业问题",
                "confidence": 0.42,
                "reason": "模拟模型误选了未注册子智能体。",
            }
        if "agent_ids" in schema_hint:
            self.route_calls += 1
            return {
                "route_type": "delegate",
                "agent_ids": ["unknown_agent"],
                "intent": "未知专业问题",
                "confidence": 0.42,
                "reason": "模拟模型误选了未注册子智能体。",
            }
        return super().chat_json(**kwargs)


class RecoveringRoutingLLM(FixtureLLMClient):
    """模拟 LLM 首次生成非法路由，收到重试提示后修正为合法子智能体。"""

    def __init__(self) -> None:
        """记录路由调用次数，确认 Runtime 确实执行了纠偏重试。"""
        self.route_calls = 0

    def chat_json(self, **kwargs):
        """首次路由返回非法 ID；携带 previous_response 的重试请求返回 DCS 智能体。"""
        schema_hint = kwargs.get("schema_hint", {})
        user = str(kwargs.get("user", ""))
        is_retry = "previous_response" in user
        if "execution_mode" in schema_hint:
            self.route_calls += 1
            if is_retry:
                return {
                    "execution_mode": "single",
                    "agent_ids": ["dcs_query_agent"],
                    "workflow_id": "",
                    "intent": "工业位号查数",
                    "confidence": 0.86,
                    "reason": "重试后选择了已注册的 DCS 查数智能体。",
                }
            return {
                "execution_mode": "single",
                "agent_ids": ["missing_agent"],
                "workflow_id": "",
                "intent": "工业位号查数",
                "confidence": 0.51,
                "reason": "首次模拟非法子智能体 ID。",
            }
        if "agent_id" in schema_hint:
            self.route_calls += 1
            if is_retry:
                return {
                    "route_type": "delegate",
                    "agent_id": "dcs_query_agent",
                    "intent": "工业位号查数",
                    "confidence": 0.86,
                    "reason": "重试后选择了已注册的 DCS 查数智能体。",
                }
            return {
                "route_type": "delegate",
                "agent_id": "missing_agent",
                "intent": "工业位号查数",
                "confidence": 0.51,
                "reason": "首次模拟非法子智能体 ID。",
            }
        return super().chat_json(**kwargs)


CASE_PATH = Path(__file__).parent / "cases" / "reactor_temp_case.json"
QUERY_CASE_PATH = Path(__file__).parent / "cases" / "query_tag_value_case.json"
BUSINESS_CASE_PATH = Path(__file__).parent / "cases" / "business_sales_case.json"
WORKFLOW_CASE_PATH = Path(__file__).parent / "cases" / "business_dcs_workflow_case.json"


class ZhugeAgentRuntimeTest(unittest.TestCase):  # pylint: disable=too-many-public-methods
    def test_lazy_prompt_renders_only_when_consumed(self) -> None:
        """验证 LazyPrompt 初始化不渲染，首次消费后缓存结果，重复消费不会再次调用回调。"""
        calls = []
        prompt = LazyPrompt(lambda: calls.append("rendered") or "hello")

        self.assertEqual(calls, [])
        self.assertEqual(str(prompt), "hello")
        self.assertEqual(str(prompt), "hello")
        self.assertEqual(calls, ["rendered"])

    def test_skills_are_loaded_recursively_from_skill_folders(self) -> None:
        """验证递归加载能发现不同嵌套业务目录中的内置 Skill。"""
        runtime = ZhugeAgentRuntime(llm_client=FixtureLLMClient())
        skill_ids = {skill.id for skill in runtime.skills.all()}

        self.assertIn("query_tag_value", skill_ids)
        self.assertIn("reactor_temp_forecast_adjust", skill_ids)
        self.assertIn("business_metric_analysis", skill_ids)

    def test_sub_agents_see_public_and_own_private_skills(self) -> None:
        """验证公共 Skill 对所有子智能体可见，私有 Skill 仅出现在声明它的智能体候选中。"""
        runtime = ZhugeAgentRuntime(llm_client=FixtureLLMClient())

        dcs_agent = runtime.agents.get("dcs_query_agent")
        diagnosis_agent = runtime.agents.get("reactor_diagnosis_agent")
        business_agent = runtime.agents.get("business_metric_agent")

        dcs_visible = {skill.id for skill in runtime.skills.visible_to(dcs_agent.id, dcs_agent.skill_ids)}
        diagnosis_visible = {
            skill.id for skill in runtime.skills.visible_to(diagnosis_agent.id, diagnosis_agent.skill_ids)
        }
        business_visible = {
            skill.id for skill in runtime.skills.visible_to(business_agent.id, business_agent.skill_ids)
        }

        self.assertEqual(dcs_visible, {"query_tag_value"})
        self.assertEqual(diagnosis_visible, {"query_tag_value", "reactor_temp_forecast_adjust"})
        self.assertEqual(business_visible, {"query_tag_value", "business_metric_analysis"})

    def test_sub_agents_load_prompt_template_files(self) -> None:
        """验证每个内置子智能体的配置都能解析到存在的提示词模板文件。"""
        runtime = ZhugeAgentRuntime(llm_client=FixtureLLMClient())

        for agent in runtime.agents.all():
            self.assertIsNotNone(agent.prompt_template_path)
            self.assertTrue(agent.prompt_template_path.exists())

    def test_workflows_are_loaded_as_graph_specs(self) -> None:
        """验证内置 graph workflow 被加载，并包含 agent、人审和汇总节点。"""
        runtime = ZhugeAgentRuntime(llm_client=FixtureLLMClient())

        workflow = runtime.workflows.get("business_metric_dcs_workflow")
        node_types = {node.type for node in workflow.nodes}

        self.assertEqual(workflow.id, "business_metric_dcs_workflow")
        self.assertIn("agent", node_types)
        self.assertIn("human_review", node_types)
        self.assertIn("join", node_types)
        self.assertEqual(len(workflow.edges), 3)

    def test_reactor_temperature_case_uses_llm_only_for_allowed_steps(self) -> None:
        """验证诊断案例路由、校验及计算结果，并检查 LLM 参与步骤和文案关键数字。"""
        case_data = json.loads(CASE_PATH.read_text(encoding="utf-8"))
        runtime = ZhugeAgentRuntime(llm_client=FixtureLLMClient(), case_data=case_data)

        result = runtime.answer(case_data["request"])

        self.assertEqual(result.skill_id, "reactor_temp_forecast_adjust")
        self.assertTrue(result.verification["passed"])
        self.assertTrue(result.verification["post_check"]["passed"])
        self.assertEqual(result.metrics["cooling_drop_rate"], 0.18)
        self.assertEqual(result.agent_id, "reactor_diagnosis_agent")

        llm_steps = {event.step for event in result.trace if event.used_llm}
        self.assertEqual(llm_steps, {"A1", "A2", "C", "G"})

        deterministic_steps = {event.step for event in result.trace if not event.used_llm}
        self.assertTrue({"B", "D1", "D2", "D3", "D4", "E", "F", "H"}.issubset(deterministic_steps))

        self.assertIn("18.0%", result.answer)
        self.assertIn(str(result.metrics["forecast_2h"]), result.answer)
        self.assertIn(f"{result.metrics['over_limit_probability']:.0%}", result.answer)

    def test_query_tag_value_skill_reads_current_values(self) -> None:
        """验证查数案例委派到 DCS 智能体，测点、单位及回答数字来自案例工具结果。"""
        case_data = json.loads(QUERY_CASE_PATH.read_text(encoding="utf-8"))
        runtime = ZhugeAgentRuntime(llm_client=FixtureLLMClient(), case_data=case_data)

        result = runtime.answer(case_data["request"])

        self.assertEqual(result.skill_id, "query_tag_value")
        self.assertEqual(result.agent_id, "dcs_query_agent")
        self.assertTrue(result.verification["passed"])
        self.assertTrue(result.verification["post_check"]["passed"])
        self.assertEqual(result.metrics["温度"]["value"], 130.9)
        self.assertEqual(result.metrics["压力"]["unit"], "MPa")
        self.assertEqual(result.actions, [])

        llm_steps = {event.step for event in result.trace if event.used_llm}
        self.assertEqual(llm_steps, {"A1", "A2", "C", "G"})
        self.assertIn("130.9", result.answer)
        self.assertIn("0.42", result.answer)
        self.assertIn("82.0", result.answer)

    def test_single_answer_exposes_progress_callback(self) -> None:
        """验证单任务回调收到路由与执行进度，且与最终结果保存的 progress 一致。"""
        case_data = json.loads(QUERY_CASE_PATH.read_text(encoding="utf-8"))
        runtime = ZhugeAgentRuntime(llm_client=FixtureLLMClient(), case_data=case_data)
        progress = []

        result = runtime.answer(case_data["request"], progress_callback=progress.append)

        self.assertTrue(result.verification["passed"])
        self.assertEqual(progress, result.progress)
        self.assertEqual(progress[0].actor, "MasterAgent / Router")
        self.assertEqual(progress[0].agent_id, "dcs_query_agent")
        self.assertIn("A2", {event.step for event in progress})
        self.assertIn("G", {event.step for event in progress})

    def test_single_answer_can_be_streamed(self) -> None:
        """验证单任务迭代器输出步骤进度，最后返回通过校验的 StructuredAnswer。"""
        case_data = json.loads(QUERY_CASE_PATH.read_text(encoding="utf-8"))
        runtime = ZhugeAgentRuntime(llm_client=FixtureLLMClient(), case_data=case_data)

        items = list(runtime.iter_answer(case_data["request"]))

        self.assertIsInstance(items[-1], StructuredAnswer)
        self.assertTrue(items[-1].verification["passed"])
        self.assertGreaterEqual(len([item for item in items if isinstance(item, AgentProgressEvent)]), 4)
        self.assertTrue(any(isinstance(item, AnswerTokenEvent) for item in items))

    def test_general_chat_is_answered_by_master_without_sub_agent(self) -> None:
        """验证普通寒暄由 MasterAgent 直接回答，不触发子智能体、Skill 或工具链。"""
        runtime = ZhugeAgentRuntime(llm_client=FixtureLLMClient())
        progress = []

        result = runtime.answer("你好吗", progress_callback=progress.append)

        self.assertEqual(result.agent_id, "MasterAgent")
        self.assertEqual(result.agent_name, "MasterAgent")
        self.assertEqual(result.skill_id, "direct_answer")
        self.assertTrue(result.verification["passed"])
        self.assertEqual([event.step for event in result.trace], ["A1"])
        self.assertEqual(progress, result.progress)
        self.assertIsNone(progress[0].agent_id)
        self.assertIn("我很好", result.answer)

    def test_general_chat_streams_only_master_progress(self) -> None:
        """验证普通问答流式入口只输出 MasterAgent 路由进度和最终回答。"""
        runtime = ZhugeAgentRuntime(llm_client=FixtureLLMClient())

        items = list(runtime.iter_answer("你好吗"))

        self.assertIsInstance(items[0], AgentProgressEvent)
        self.assertIsInstance(items[-1], StructuredAnswer)
        self.assertEqual(len([item for item in items if isinstance(item, AgentProgressEvent)]), 1)
        self.assertTrue(any(isinstance(item, AnswerTokenEvent) for item in items))
        self.assertEqual(items[-1].agent_id, "MasterAgent")
        self.assertEqual(items[-1].skill_id, "direct_answer")
        streamed = "".join(item.delta for item in items if isinstance(item, AnswerTokenEvent))
        self.assertEqual(streamed, items[-1].answer)

    def test_uncovered_request_reports_sub_agent_capability_gap(self) -> None:
        """验证无子智能体承接的诉求被归类为能力不足，而不是普通寒暄或误路由。"""
        runtime = ZhugeAgentRuntime(llm_client=FixtureLLMClient())

        result = runtime.answer("帮我审一下这份合同风险")

        self.assertEqual(result.agent_id, "MasterAgent")
        self.assertEqual(result.skill_id, "direct_answer")
        self.assertEqual(result.trace[0].metadata["route_status"], "capability_gap")
        self.assertIn("能力不足", result.answer)
        self.assertIn("当前可用的子智能体", result.answer)

    def test_uncovered_request_streams_capability_gap_text(self) -> None:
        """验证能力不足的流式输出使用 Runtime 兜底文案，不再被 LLM 改写成寒暄回答。"""
        runtime = ZhugeAgentRuntime(llm_client=FixtureLLMClient())

        items = list(runtime.iter_answer("帮我审一下这份合同风险"))
        final_answer = items[-1].answer
        streamed = "".join(item.delta for item in items if isinstance(item, AnswerTokenEvent))

        self.assertEqual(streamed, final_answer)
        self.assertIn("能力不足", final_answer)

    def test_invalid_single_route_falls_back_to_master_capability_boundary(self) -> None:
        """验证 LLM 重试后仍返回非法子智能体时，归类为非法路由并返回能力清单。"""
        llm = InvalidRoutingLLM()
        runtime = ZhugeAgentRuntime(llm_client=llm)

        result = runtime.answer("帮我审一下这份合同风险")

        self.assertEqual(result.agent_id, "MasterAgent")
        self.assertEqual(result.skill_id, "direct_answer")
        self.assertEqual([event.step for event in result.trace], ["A1"])
        self.assertEqual(result.trace[0].metadata["route_status"], "invalid_route")
        self.assertEqual(result.trace[0].metadata["invalid_targets"], {"invalid_agent_id": "unknown_agent"})
        self.assertGreaterEqual(llm.route_calls, 2)
        self.assertIn("已经重试一次", result.answer)
        self.assertIn("当前可用的子智能体", result.answer)
        self.assertIn("经营指标分析智能体", result.answer)
        self.assertIn("DCS 查数智能体", result.answer)

    def test_invalid_auto_plan_falls_back_without_sub_agent_execution(self) -> None:
        """验证 auto 规划和后续单路由都非法时，不执行任何子智能体链路。"""
        llm = InvalidRoutingLLM()
        runtime = ZhugeAgentRuntime(llm_client=llm)
        progress = []

        result = runtime.answer_auto("帮我审一下这份合同风险", progress_callback=progress.append)

        self.assertIsInstance(result, StructuredAnswer)
        self.assertEqual(result.agent_id, "MasterAgent")
        self.assertEqual(result.skill_id, "direct_answer")
        self.assertEqual([event.step for event in progress], ["A1"])
        self.assertEqual(progress[0].metadata["route_status"], "invalid_route")
        self.assertGreaterEqual(llm.route_calls, 4)
        self.assertIn("当前可用的子智能体", result.answer)

    def test_invalid_single_route_retries_and_uses_corrected_agent(self) -> None:
        """验证非法 agent_id 可被 LLM 重试修正，修正后继续执行合法子智能体链路。"""
        case_data = json.loads(QUERY_CASE_PATH.read_text(encoding="utf-8"))
        llm = RecoveringRoutingLLM()
        runtime = ZhugeAgentRuntime(llm_client=llm, case_data=case_data)

        result = runtime.answer(case_data["request"])

        self.assertEqual(result.agent_id, "dcs_query_agent")
        self.assertEqual(result.skill_id, "query_tag_value")
        self.assertTrue(result.verification["passed"])
        self.assertGreaterEqual(llm.route_calls, 2)

    def test_invalid_auto_plan_retries_before_dispatch(self) -> None:
        """验证 auto 规划非法 agent_id 会先重试，修正后再进入合法子智能体链路。"""
        case_data = json.loads(QUERY_CASE_PATH.read_text(encoding="utf-8"))
        llm = RecoveringRoutingLLM()
        runtime = ZhugeAgentRuntime(llm_client=llm, case_data=case_data)

        result = runtime.answer_auto(case_data["request"])

        self.assertIsInstance(result, StructuredAnswer)
        self.assertEqual(result.agent_id, "dcs_query_agent")
        self.assertEqual(result.skill_id, "query_tag_value")
        self.assertGreaterEqual(llm.route_calls, 4)

    def test_structured_run_request_passes_backend_context(self) -> None:
        """验证后端结构化请求可以携带用户、租户、角色和数据域上下文进入执行链。"""
        case_data = json.loads(QUERY_CASE_PATH.read_text(encoding="utf-8"))
        runtime = ZhugeAgentRuntime(llm_client=FixtureLLMClient(), case_data=case_data)
        run_request = AgentRunRequest(
            request=case_data["request"],
            mode="single",
            session_id="session-001",
            tenant_id="tenant-east",
            user_id="user-001",
            role="区域分析员",
            data_scope={"region": "华东"},
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
                    metadata={"source": "fixture-memory"},
                )
            ],
            context={
                "asset": "后端指定反应釜",
                "hours": 2,
                "conversation": {"active_objects": ["2# 反应釜"]},
            },
        )

        result = runtime.answer_request(run_request)
        context = next(event.metadata for event in result.trace if event.step == "B")
        conversation = context["conversation"]

        self.assertIsInstance(result, StructuredAnswer)
        self.assertEqual(context["asset"], "后端指定反应釜")
        self.assertEqual(context["hours"], 2)
        self.assertEqual(context["tenant_id"], "tenant-east")
        self.assertEqual(context["user_id"], "user-001")
        self.assertEqual(context["role"], "区域分析员")
        self.assertEqual(context["data_scope"], {"region": "华东"})
        self.assertEqual(conversation["active_objects"], ["2# 反应釜"])
        self.assertEqual(conversation["summary"], "用户正在连续追问 2# 反应釜的运行状态。")
        self.assertEqual(conversation["recent_messages"][0]["role"], "user")
        self.assertEqual(
            conversation["retrieved_memories"][0]["content"],
            "用户偏好先看结论，再看证据。",
        )
        self.assertEqual(conversation["retrieved_memories"][0]["metadata"]["source"], "fixture-memory")
        self.assertEqual(result.memory_write_candidates, [])

    def test_structured_run_events_wrap_progress_token_and_final(self) -> None:
        """验证统一事件流包含进度、token 和最终结果，便于后端映射 SSE/WebSocket。"""
        runtime = ZhugeAgentRuntime(llm_client=FixtureLLMClient())
        run_request = AgentRunRequest(
            request="你好吗",
            mode="single",
            session_id="session-stream",
            metadata={"source": "unit-test"},
        )

        events = list(runtime.iter_run_events(run_request))

        self.assertTrue(all(isinstance(event, RuntimeEvent) for event in events))
        self.assertEqual([event.sequence for event in events], list(range(1, len(events) + 1)))
        self.assertEqual(events[-1].type, "final")
        self.assertEqual(events[-1].session_id, "session-stream")
        self.assertEqual(events[-1].metadata, {"source": "unit-test"})
        self.assertTrue(any(event.type == "progress" for event in events))
        self.assertTrue(any(event.type == "token" for event in events))
        self.assertIsInstance(events[-1].payload, StructuredAnswer)

    def test_structured_run_events_wrap_invalid_mode_as_error(self) -> None:
        """验证结构化运行入口不会向事件消费者抛出配置类异常，而是返回 error 事件。"""
        runtime = ZhugeAgentRuntime(llm_client=FixtureLLMClient())

        events = list(runtime.iter_run_events(AgentRunRequest(request="测试", mode="unknown")))

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].type, "error")
        self.assertIsInstance(events[0].payload, RuntimeErrorInfo)
        self.assertEqual(events[0].payload.code, "INVALID_REQUEST")
        self.assertFalse(events[0].payload.retryable)

    def test_master_agent_can_dispatch_sub_agents_in_parallel(self) -> None:
        """验证复合问题委派经营和 DCS 智能体，并按路由顺序汇总各自的 Skill 结果及数字。"""
        case_data = json.loads(WORKFLOW_CASE_PATH.read_text(encoding="utf-8"))
        runtime = ZhugeAgentRuntime(llm_client=FixtureLLMClient(), case_data=case_data)
        request = case_data["request"]

        result = runtime.answer_parallel(request)

        self.assertTrue(result.verification["passed"])
        self.assertEqual(result.verification["sub_agent_count"], 2)
        self.assertEqual(
            [answer.agent_id for answer in result.sub_answers],
            ["business_metric_agent", "dcs_query_agent"],
        )
        self.assertEqual(
            [answer.skill_id for answer in result.sub_answers],
            ["business_metric_analysis", "query_tag_value"],
        )
        self.assertIn("37320000", result.answer)
        self.assertIn("130.9", result.answer)
        self.assertEqual(result.trace[0].actor, "MasterAgent / Parallel Router")

        auto_result = runtime.answer_request(AgentRunRequest(request=request, mode="auto"))
        self.assertIsInstance(auto_result, ParallelStructuredAnswer)
        self.assertEqual(
            [answer.agent_id for answer in auto_result.sub_answers],
            ["business_metric_agent", "dcs_query_agent"],
        )

    def test_parallel_dispatch_exposes_sub_agent_progress(self) -> None:
        """验证并行进度包含主路由及两个子智能体的工具步骤，并完整保存在最终结果中。"""
        case_data = json.loads(WORKFLOW_CASE_PATH.read_text(encoding="utf-8"))
        runtime = ZhugeAgentRuntime(llm_client=FixtureLLMClient(), case_data=case_data)
        request = case_data["request"]
        progress = []

        result = runtime.answer_parallel(request, progress_callback=progress.append)

        self.assertTrue(result.verification["passed"])
        self.assertEqual(progress, result.progress)
        self.assertEqual(progress[0].actor, "MasterAgent / Parallel Router")
        self.assertIn("business_metric_agent", {event.agent_id for event in progress})
        self.assertIn("dcs_query_agent", {event.agent_id for event in progress})
        self.assertIn("D4", {event.step for event in progress if event.agent_id == "business_metric_agent"})
        self.assertIn("D1", {event.step for event in progress if event.agent_id == "dcs_query_agent"})

    def test_parallel_dispatch_can_be_streamed(self) -> None:
        """验证并行迭代器先提供进度事件，最后返回通过校验的 ParallelStructuredAnswer。"""
        case_data = json.loads(BUSINESS_CASE_PATH.read_text(encoding="utf-8"))
        case_data["context"]["asset"] = "2# 反应釜"
        runtime = ZhugeAgentRuntime(llm_client=FixtureLLMClient(), case_data=case_data)
        request = "本月华东区域销售额、环比多少？同时查询现在 2# 反应釜温度和压力是多少？"

        items = list(runtime.iter_answer_parallel(request))

        self.assertIsInstance(items[-1], ParallelStructuredAnswer)
        self.assertTrue(items[-1].verification["passed"])
        self.assertGreaterEqual(len([item for item in items if isinstance(item, AgentProgressEvent)]), 2)
        self.assertTrue(any(isinstance(item, AnswerTokenEvent) for item in items))

    def test_graph_workflow_runs_agent_nodes_and_human_review(self) -> None:
        """验证 graph workflow 并行执行两个 agent 节点，再经过人审检查点和汇总节点。"""
        case_data = json.loads(BUSINESS_CASE_PATH.read_text(encoding="utf-8"))
        case_data["context"]["asset"] = "2# 反应釜"
        runtime = ZhugeAgentRuntime(llm_client=FixtureLLMClient(), case_data=case_data)
        request = "本月华东区域销售额、环比多少？同时查询现在 2# 反应釜温度和压力是多少？"
        progress = []

        result = runtime.answer_workflow(
            request,
            workflow_id="business_metric_dcs_workflow",
            progress_callback=progress.append,
        )

        self.assertTrue(result.verification["passed"])
        self.assertEqual(result.workflow_id, "business_metric_dcs_workflow")
        self.assertEqual(
            [answer.agent_id for answer in result.sub_answers],
            ["business_metric_agent", "dcs_query_agent"],
        )
        self.assertTrue(result.node_outputs["human_review"]["passed"])
        self.assertEqual(result.node_outputs["final_join"]["node_type"], "join")
        self.assertIn("37320000", result.answer)
        self.assertIn("130.9", result.answer)
        self.assertIn("business_metric_dcs_workflow", {event.agent_id for event in progress})
        self.assertIn("business_metric_agent", {event.agent_id for event in progress})
        self.assertIn("dcs_query_agent", {event.agent_id for event in progress})
        self.assertIn("W:human_review", {event.step for event in progress})
        self.assertIn("W:final_join", {event.step for event in progress})

        auto_result = runtime.answer_request(
            AgentRunRequest(
                request=f"{request} 请走工作流并经过人审。",
                mode="auto",
            )
        )
        self.assertIsInstance(auto_result, WorkflowStructuredAnswer)
        self.assertEqual(auto_result.workflow_id, "business_metric_dcs_workflow")

    def test_graph_workflow_can_be_streamed(self) -> None:
        """验证 workflow 迭代器输出图节点进度、回答 token，并最终返回 WorkflowStructuredAnswer。"""
        case_data = json.loads(BUSINESS_CASE_PATH.read_text(encoding="utf-8"))
        case_data["context"]["asset"] = "2# 反应釜"
        runtime = ZhugeAgentRuntime(llm_client=FixtureLLMClient(), case_data=case_data)
        request = "本月华东区域销售额、环比多少？同时查询现在 2# 反应釜温度和压力是多少？"

        items = list(runtime.iter_answer_workflow(request, workflow_id="business_metric_dcs_workflow"))

        self.assertIsInstance(items[-1], WorkflowStructuredAnswer)
        self.assertTrue(items[-1].verification["passed"])
        self.assertTrue(any(isinstance(item, AgentProgressEvent) for item in items))
        self.assertTrue(any(isinstance(item, AnswerTokenEvent) for item in items))

    def test_business_metric_analysis_chain(self) -> None:
        """验证经营指标链的路由、销售额、城市排序、图表配置、LLM 边界及文案关键数字。"""
        case_data = json.loads(BUSINESS_CASE_PATH.read_text(encoding="utf-8"))
        runtime = ZhugeAgentRuntime(llm_client=FixtureLLMClient(), case_data=case_data)

        result = runtime.answer(case_data["request"])

        self.assertEqual(result.skill_id, "business_metric_analysis")
        self.assertEqual(result.agent_id, "business_metric_agent")
        self.assertTrue(result.verification["passed"])
        self.assertTrue(result.verification["post_check"]["passed"])
        self.assertEqual(result.metrics["totals"]["sales_amount"], 37320000)
        self.assertEqual(result.metrics["rows"][0]["city"], "上海")
        self.assertEqual(result.metrics["visualization"]["charts"][0]["type"], "bar")
        self.assertEqual(result.actions, [])

        llm_steps = {event.step for event in result.trace if event.used_llm}
        self.assertEqual(llm_steps, {"A1", "A2", "C", "D1", "G"})

        deterministic_steps = {event.step for event in result.trace if not event.used_llm}
        self.assertTrue({"B", "D2", "D3", "D4", "D5", "D6", "E", "F", "H"}.issubset(deterministic_steps))

        self.assertIn("37320000", result.answer)
        self.assertIn("4.3%", result.answer)
        self.assertIn("12860000", result.answer)


if __name__ == "__main__":
    unittest.main()
