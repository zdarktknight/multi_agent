import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

from zhuge_agent import (
    AgentProgressEvent,
    AnswerTokenEvent,
    ParallelStructuredAnswer,
    StructuredAnswer,
    WorkflowStructuredAnswer,
    ZhugeAgentRuntime,
)
from zhuge_agent.llm import LLMError, OpenAICompatibleLLMClient
from zhuge_agent.utils.config import ConfigError, DEFAULT_CONFIG_PATH
from zhuge_agent.utils.logging_utils import log_call, log_value

logger = logging.getLogger(__name__)

def main() -> None:
    """读取命令行案例和模式，运行 Demo 并打印回答、执行链、建议、证据及结构化结果。

    --config 指定 INI 配置，--stream 展示单任务进度，--parallel 展示并行进度。
    未指定案例时使用反应釜诊断案例；启动时显示当前使用的客户端，不输出密钥。
    """
    default_case = Path(__file__).parent / "tests" / "cases" / "reactor_temp_case.json"
    parser = argparse.ArgumentParser(description="诸葛 AI 智能体架构演示")
    parser.add_argument("case", nargs="?", type=Path, default=default_case, help="案例 JSON 路径")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="LLM INI 配置路径")
    parser.add_argument("--parallel", action="store_true", help="并行调用子智能体并显示进度")
    parser.add_argument("--stream", action="store_true", help="显示单任务步骤进度")
    parser.add_argument("--workflow", action="store_true", help="执行 graph workflow；未指定 ID 时按请求匹配")
    parser.add_argument("--workflow-id", help="指定要执行的 workflow ID")
    parser.add_argument("--no-typewriter", action="store_true", help="关闭流式模式下最终回答的实时 token 打印")
    args = parser.parse_args()
    case_data = json.loads(args.case.read_text(encoding="utf-8"))
    request = case_data["request"]
    try:
        runtime = ZhugeAgentRuntime(case_data=case_data, config_path=args.config)
    except (ConfigError, LLMError) as exc:
        parser.error(str(exc))
    logger.info("Demo 输入 case=%s config=%s request=%s parallel=%s stream=%s",
                args.case, args.config, log_value(request), args.parallel, args.stream)

    print("=== 诸葛 AI 智能体架构演示 ===", flush=True)
    print(f"配置文件：{args.config}", flush=True)
    if isinstance(runtime.llm, OpenAICompatibleLLMClient):
        print(f"LLM 模式：真实 API，模型：{runtime.llm.model}", flush=True)
    else:
        print("LLM 模式：离线 fixture（不调用真实 API）", flush=True)
    print(f"用户请求：{request}\n", flush=True)
    if args.workflow:
        result = _run_workflow_demo(
            runtime,
            request,
            workflow_id=args.workflow_id,
            print_tokens=not args.no_typewriter,
        )
    elif args.parallel:
        result = _run_parallel_demo(runtime, request, print_tokens=not args.no_typewriter)
    elif args.stream:
        result = _run_stream_demo(runtime, request, print_tokens=not args.no_typewriter)
    else:
        result = asdict(runtime.answer_auto(request))

    print("推理链路：", flush=True)
    for event in result["trace"]:
        flag = "LLM参与" if event["used_llm"] else "确定性"
        print(f"- {event['step']} {event['actor']} [{flag}] {event['message']}", flush=True)

    streamed_run = bool(args.stream or args.parallel or args.workflow)
    if streamed_run and not args.no_typewriter:
        print("\n最终回答：已在上方实时输出。", flush=True)
    else:
        print("\n最终回答：", flush=True)
        print(result["answer"], flush=True)

    print("\n建议动作：", flush=True)
    for action in _actions_from_result(result):
        print(f"- {action}", flush=True)

    print("\n证据链：", flush=True)
    for item in _evidence_from_result(result):
        print(f"- {item['source']}：{item['detail']}", flush=True)

    print("\n结构化输出：", flush=True)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    logger.info("Demo 输出 answer=%s verification=%s", log_value(result["answer"]), result["verification"])


@log_call("request", level=logging.DEBUG)
def _run_parallel_demo(runtime: ZhugeAgentRuntime, request: str, *, print_tokens: bool = True) -> dict:
    """消费并行迭代器并实时打印子智能体进度，返回最终回答字典；缺少最终结果时抛出 RuntimeError。"""
    final_answer = None
    printed_tokens = False
    # token_line_open 表示当前 stdout 行正在输出回答 token；下一条进度到来前需要先换行。
    token_line_open = False
    print("并行进度：", flush=True)
    print("- [MasterAgent] A0 Runtime [确定性] 已收到请求，正在等待主智能体路由...", flush=True)
    for item in runtime.iter_answer_parallel(request):
        if isinstance(item, AgentProgressEvent):
            if token_line_open:
                print(flush=True)
                token_line_open = False
            agent_name = item.agent_name or "MasterAgent"
            flag = "LLM参与" if item.used_llm else "确定性"
            print(f"- [{agent_name}] {item.step} {item.actor} [{flag}] {item.message}", flush=True)
        elif isinstance(item, AnswerTokenEvent):
            if print_tokens:
                if not printed_tokens:
                    print("\n最终回答（实时 token）：", flush=True)
                    printed_tokens = True
                # AnswerTokenEvent.delta 已经是模型返回的增量片段，不能再人为切分或延迟。
                sys.stdout.write(item.delta)
                sys.stdout.flush()
                token_line_open = True
        elif isinstance(item, ParallelStructuredAnswer):
            final_answer = item
    if final_answer is None:
        raise RuntimeError("Parallel runtime did not return a final answer.")
    print(flush=True)
    return asdict(final_answer)


@log_call("request", level=logging.DEBUG)
def _run_stream_demo(runtime: ZhugeAgentRuntime, request: str, *, print_tokens: bool = True) -> dict:
    """消费单任务迭代器并打印步骤进度，返回最终回答字典；缺少最终结果时抛出 RuntimeError。"""
    final_answer = None
    printed_tokens = False
    # token_line_open 表示当前 stdout 行正在输出回答 token；下一条进度到来前需要先换行。
    token_line_open = False
    print("流式进度：", flush=True)
    print("- [MasterAgent] A0 Runtime [确定性] 已收到请求，正在等待主智能体路由...", flush=True)
    for item in runtime.iter_answer(request):
        if isinstance(item, AgentProgressEvent):
            if token_line_open:
                print(flush=True)
                token_line_open = False
            agent_name = item.agent_name or "MasterAgent"
            flag = "LLM参与" if item.used_llm else "确定性"
            print(f"- [{agent_name}] {item.step} {item.actor} [{flag}] {item.message}", flush=True)
        elif isinstance(item, AnswerTokenEvent):
            if print_tokens:
                if not printed_tokens:
                    print("\n最终回答（实时 token）：", flush=True)
                    printed_tokens = True
                # AnswerTokenEvent.delta 已经是模型返回的增量片段，不能再人为切分或延迟。
                sys.stdout.write(item.delta)
                sys.stdout.flush()
                token_line_open = True
        elif isinstance(item, StructuredAnswer):
            final_answer = item
    if final_answer is None:
        raise RuntimeError("Runtime did not return a final answer.")
    print(flush=True)
    return asdict(final_answer)


@log_call("request", "workflow_id", level=logging.DEBUG)
def _run_workflow_demo(
    runtime: ZhugeAgentRuntime,
    request: str,
    *,
    workflow_id: str | None = None,
    print_tokens: bool = True,
) -> dict:
    """消费 workflow 迭代器并打印图节点、子智能体进度及 token，返回最终回答字典。"""
    final_answer = None
    printed_tokens = False
    token_line_open = False
    print("工作流进度：", flush=True)
    print("- [WorkflowRuntime] W0 Graph [确定性] 已收到请求，正在准备图编排...", flush=True)
    for item in runtime.iter_answer_workflow(request, workflow_id=workflow_id):
        if isinstance(item, AgentProgressEvent):
            if token_line_open:
                print(flush=True)
                token_line_open = False
            agent_name = item.agent_name or "WorkflowRuntime"
            flag = "LLM参与" if item.used_llm else "确定性"
            print(f"- [{agent_name}] {item.step} {item.actor} [{flag}] {item.message}", flush=True)
        elif isinstance(item, AnswerTokenEvent):
            if print_tokens:
                if not printed_tokens:
                    print("\n最终回答（实时 token）：", flush=True)
                    printed_tokens = True
                sys.stdout.write(item.delta)
                sys.stdout.flush()
                token_line_open = True
        elif isinstance(item, WorkflowStructuredAnswer):
            final_answer = item
    if final_answer is None:
        raise RuntimeError("Workflow runtime did not return a final answer.")
    print(flush=True)
    return asdict(final_answer)


def _actions_from_result(result: dict) -> list[str]:
    """读取单任务的建议动作，或按并行子回答顺序汇总动作，供终端展示。"""
    if "actions" in result:
        return result["actions"]
    actions = []
    for sub_answer in result.get("sub_answers", []):
        actions.extend(sub_answer.get("actions", []))
    return actions


def _evidence_from_result(result: dict) -> list[dict]:
    """读取单任务证据列表，或汇总所有并行子回答的证据，供终端展示。"""
    if "evidence" in result:
        return result["evidence"]
    evidence = []
    for sub_answer in result.get("sub_answers", []):
        evidence.extend(sub_answer.get("evidence", []))
    return evidence


if __name__ == "__main__":
    main()
