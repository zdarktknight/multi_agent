from __future__ import annotations

import json
import logging
from pathlib import Path

from .schema import WorkflowEdge, WorkflowNode, WorkflowSpec
from .utils.logging_utils import log_call
from .utils.text import tokenize_for_match


BUNDLED_WORKFLOWS_DIR = Path(__file__).resolve().parent / "workflows"


class WorkflowLoadError(RuntimeError):
    """Workflow 配置缺失、重复、引用错误或存在环路时抛出的异常。"""


class WorkflowRegistry:
    """加载并管理 graph-based workflow 配置，提供按 ID 查询和请求文本备用路由能力。"""

    @log_call("workflows_dir")
    def __init__(self, workflows_dir: str | Path | None = None) -> None:
        """从指定目录递归加载 JSON 工作流；目录不存在时保留空注册表，便于渐进启用。"""
        self.workflows_dir = Path(workflows_dir or BUNDLED_WORKFLOWS_DIR)
        self._workflows = self._load_recursive(self.workflows_dir) if self.workflows_dir.exists() else {}

    @log_call("workflow_id", level=logging.DEBUG)
    def get(self, workflow_id: str) -> WorkflowSpec:
        """按 ID 返回工作流配置；ID 未注册时抛出 KeyError。"""
        return self._workflows[workflow_id]

    def has(self, workflow_id: str | None) -> bool:
        """判断非空工作流 ID 是否已注册。"""
        return bool(workflow_id and workflow_id in self._workflows)

    def all(self) -> list[WorkflowSpec]:
        """返回全部已注册工作流，供主智能体或 API 层展示可用 pipeline。"""
        return list(self._workflows.values())

    @log_call("request")
    def route(self, request: str) -> WorkflowSpec:
        """按请求与工作流说明、触发示例的词匹配选择最合适的 workflow。"""
        if not self._workflows:
            raise WorkflowLoadError("No workflows loaded.")
        text = request.lower()
        scored: list[tuple[int, WorkflowSpec]] = []
        for workflow in self._workflows.values():
            haystack = " ".join((workflow.name, workflow.description, *workflow.trigger_examples)).lower()
            score = sum(1 for token in tokenize_for_match(text) if token and token in haystack)
            scored.append((score, workflow))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1]

    @log_call("root", level=logging.DEBUG)
    def _load_recursive(self, root: Path) -> dict[str, WorkflowSpec]:
        """按路径排序递归读取 workflow JSON，构建 ID 索引并拒绝重复 ID。"""
        workflows: dict[str, WorkflowSpec] = {}
        for config_path in sorted(root.rglob("*.json")):
            workflow = load_workflow_file(config_path)
            if workflow.id in workflows:
                raise WorkflowLoadError(f"Duplicate workflow id: {workflow.id}")
            workflows[workflow.id] = workflow
        return workflows


@log_call("config_path")
def load_workflow_file(config_path: Path) -> WorkflowSpec:
    """读取并校验单个 workflow JSON，返回可执行的 WorkflowSpec。"""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    workflow_id = str(config["id"])
    nodes = tuple(_node_from_dict(item) for item in config.get("nodes", []))
    edges = tuple(_edge_from_dict(item) for item in config.get("edges", []))
    if not nodes:
        raise WorkflowLoadError(f"Workflow {workflow_id} has no nodes.")
    _validate_graph(workflow_id, nodes, edges)
    return WorkflowSpec(
        id=workflow_id,
        name=str(config.get("name") or workflow_id),
        description=str(config.get("description") or ""),
        trigger_examples=tuple(config.get("trigger_examples", [])),
        nodes=nodes,
        edges=edges,
    )


def _node_from_dict(item: dict) -> WorkflowNode:
    """将节点配置转换为 WorkflowNode，保留 config 给后续人审、分支和 UI 扩展使用。"""
    node_id = str(item["id"])
    return WorkflowNode(
        id=node_id,
        type=str(item["type"]),
        name=str(item.get("name") or node_id),
        agent_id=str(item["agent_id"]) if item.get("agent_id") else None,
        config=dict(item.get("config", {})),
    )


def _edge_from_dict(item: dict) -> WorkflowEdge:
    """将边配置转换为 WorkflowEdge；未声明 condition 时默认始终启用。"""
    return WorkflowEdge(
        from_node=str(item["from"]),
        to_node=str(item["to"]),
        condition=str(item.get("condition") or "always"),
    )


def _validate_graph(workflow_id: str, nodes: tuple[WorkflowNode, ...], edges: tuple[WorkflowEdge, ...]) -> None:
    """校验节点唯一性、边引用和 DAG 约束，避免运行时遇到不可恢复的图结构错误。"""
    node_ids = [node.id for node in nodes]
    if len(node_ids) != len(set(node_ids)):
        raise WorkflowLoadError(f"Workflow {workflow_id} has duplicate node ids.")
    node_id_set = set(node_ids)
    for node in nodes:
        if node.type == "agent" and not node.agent_id:
            raise WorkflowLoadError(f"Workflow {workflow_id} agent node {node.id} missing agent_id.")
        if node.type not in {"agent", "join", "human_review"}:
            raise WorkflowLoadError(f"Workflow {workflow_id} node {node.id} has unsupported type: {node.type}")
    for edge in edges:
        if edge.from_node not in node_id_set or edge.to_node not in node_id_set:
            raise WorkflowLoadError(f"Workflow {workflow_id} edge references unknown node.")
    _assert_acyclic(workflow_id, node_ids, edges)


def _assert_acyclic(workflow_id: str, node_ids: list[str], edges: tuple[WorkflowEdge, ...]) -> None:
    """使用 Kahn 算法检查有向图是否无环；有环 workflow 不允许进入运行时。"""
    indegree = {node_id: 0 for node_id in node_ids}
    outgoing: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
    for edge in edges:
        outgoing[edge.from_node].append(edge.to_node)
        indegree[edge.to_node] += 1
    ready = [node_id for node_id in node_ids if indegree[node_id] == 0]
    visited = 0
    while ready:
        current = ready.pop()
        visited += 1
        for next_node in outgoing[current]:
            indegree[next_node] -= 1
            if indegree[next_node] == 0:
                ready.append(next_node)
    if visited != len(node_ids):
        raise WorkflowLoadError(f"Workflow {workflow_id} must be a directed acyclic graph.")
