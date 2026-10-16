from __future__ import annotations

import json
import logging
from pathlib import Path

from .schema import AgentSpec
from .utils.logging_utils import log_call
from .utils.text import tokenize_for_match


BUNDLED_AGENTS_DIR = Path(__file__).resolve().parent / "agents"


class AgentLoadError(RuntimeError):
    pass


class AgentRegistry:
    """加载并管理子智能体配置，为主智能体提供查询和备用路由能力。"""

    @log_call("agents_dir")
    def __init__(self, agents_dir: str | Path | None = None) -> None:
        """从指定目录递归加载配置；未指定时使用内置目录，未加载到智能体时抛出 AgentLoadError。"""
        self.agents_dir = Path(agents_dir or BUNDLED_AGENTS_DIR)
        self._agents = self._load_recursive(self.agents_dir)
        if not self._agents:
            raise AgentLoadError(f"No agents loaded from: {self.agents_dir}")

    @log_call("agent_id", level=logging.DEBUG)
    def get(self, agent_id: str) -> AgentSpec:
        """按 ID 返回子智能体配置；ID 未注册时抛出 KeyError。"""
        return self._agents[agent_id]

    def has(self, agent_id: str | None) -> bool:
        """判断非空智能体 ID 是否已注册。"""
        return bool(agent_id and agent_id in self._agents)

    def all(self) -> list[AgentSpec]:
        """返回全部已注册子智能体，供主智能体构建路由候选列表。"""
        return list(self._agents.values())

    @log_call("request")
    def route(self, request: str) -> AgentSpec:
        """按请求词与角色描述的匹配数量选择子智能体，作为 LLM 路由返回无效 ID 时的备用方案。"""
        text = request.lower()
        scored: list[tuple[int, AgentSpec]] = []
        for agent in self._agents.values():
            haystack = " ".join((agent.name, agent.role, agent.description, *agent.skill_ids)).lower()
            score = sum(1 for token in tokenize_for_match(text) if token and token in haystack)
            scored.append((score, agent))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1]

    @log_call("root", level=logging.DEBUG)
    def _load_recursive(self, root: Path) -> dict[str, AgentSpec]:
        """按路径排序递归读取 JSON 配置，构建 ID 索引；重复 ID 会触发 AgentLoadError。"""
        agents: dict[str, AgentSpec] = {}
        for config_path in sorted(root.rglob("*.json")):
            agent = load_agent_file(config_path)
            if agent.id in agents:
                raise AgentLoadError(f"Duplicate agent id: {agent.id}")
            agents[agent.id] = agent
        return agents


@log_call("config_path")
def load_agent_file(config_path: Path) -> AgentSpec:
    """读取单个 JSON 配置，解析私有 Skill ID 和提示词模板路径，返回 AgentSpec。"""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    agent_id = str(config["id"])
    template_value = config.get("prompt_template")
    prompt_template_path = _resolve_template_path(config_path, str(template_value)) if template_value else None
    return AgentSpec(
        id=agent_id,
        name=str(config.get("name") or agent_id),
        role=str(config.get("role") or "specialist"),
        description=str(config.get("description") or ""),
        skill_ids=tuple(config.get("skill_ids", [])),
        prompt_template_path=prompt_template_path,
    )


def _resolve_template_path(config_path: Path, template_value: str) -> Path:
    """保留绝对模板路径，并以智能体配置所在目录为基准解析相对路径。"""
    template_path = Path(template_value)
    if template_path.is_absolute():
        return template_path
    return (config_path.parent / template_path).resolve()
