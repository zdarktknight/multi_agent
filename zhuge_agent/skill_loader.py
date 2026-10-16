from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from .schema import Skill, SkillStep, StepKind
from .utils.logging_utils import log_call
from .utils.text import tokenize_for_match


BUNDLED_SKILLS_DIR = Path(__file__).resolve().parent / "skills"
EXTERNAL_SKILLS_DIR = Path(__file__).resolve().parents[1] / "skills"


class SkillLoadError(RuntimeError):
    pass


class SkillRegistry:
    """递归加载 Anthropic 风格的 Skill 文件夹，并管理公共和私有能力的可见性。

    SKILL.md 保存能力说明和简单 frontmatter，skill.json 保存执行图。
    内置目录、显式或环境变量指定的扩展目录、根目录 skills 均参与加载。
    """

    @log_call("skills_dir")
    def __init__(self, skills_dir: str | Path | None = None) -> None:
        """加载内置及扩展 Skill；无可用能力或出现重复 ID 时抛出 SkillLoadError。"""
        self.skills_dirs = self._resolve_skill_dirs(skills_dir)
        self._skills = self._load_many(self.skills_dirs)
        if not self._skills:
            paths = ", ".join(str(path) for path in self.skills_dirs)
            raise SkillLoadError(f"No skills loaded from: {paths}")

    @log_call("request", "visible_skills")
    def route(self, request: str, visible_skills: list[Skill] | None = None) -> Skill:
        """在非空候选列表中按描述和示例匹配请求；未提供候选或列表为空时使用全部 Skill。"""
        text = request.lower()
        scored: list[tuple[int, Skill]] = []
        for skill in visible_skills or self.all():
            haystack = " ".join((skill.name, skill.description, *skill.intent_examples)).lower()
            score = sum(1 for token in tokenize_for_match(text) if token and token in haystack)
            scored.append((score, skill))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1]

    @log_call("skill_id", level=logging.DEBUG)
    def get(self, skill_id: str) -> Skill:
        """按 ID 获取完整 Skill 定义；ID 不存在时抛出 KeyError。"""
        return self._skills[skill_id]

    def has(self, skill_id: str | None) -> bool:
        """判断非空 Skill ID 是否已经加载。"""
        return bool(skill_id and skill_id in self._skills)

    def all(self) -> list[Skill]:
        """返回已加载的全部 Skill，保持注册顺序。"""
        return list(self._skills.values())

    @log_call("agent_id", "private_skill_ids", level=logging.DEBUG)
    def visible_to(self, agent_id: str, private_skill_ids: tuple[str, ...] = ()) -> list[Skill]:
        """返回公共 Skill 和智能体显式声明的私有 Skill，同时兼容旧 allowed_agents 配置。"""
        private_ids = set(private_skill_ids)
        return [
            skill
            for skill in self._skills.values()
            if skill.visibility == "public" or skill.id in private_ids or agent_id in skill.allowed_agents
        ]

    @log_call("skills_dir", level=logging.DEBUG)
    def _resolve_skill_dirs(self, skills_dir: str | Path | None) -> list[Path]:
        """组合加载目录并忽略不存在或重复的路径；显式扩展目录优先于 ZHUGE_SKILLS_DIR。"""
        paths = [BUNDLED_SKILLS_DIR]
        if skills_dir:
            paths.append(Path(skills_dir))
        elif os.getenv("ZHUGE_SKILLS_DIR"):
            paths.extend(Path(part) for part in os.getenv("ZHUGE_SKILLS_DIR", "").split(os.pathsep) if part)
        paths.append(EXTERNAL_SKILLS_DIR)
        resolved: list[Path] = []
        for path in paths:
            if path.exists() and path not in resolved:
                resolved.append(path)
        return resolved

    @log_call("roots", level=logging.DEBUG)
    def _load_many(self, roots: list[Path]) -> dict[str, Skill]:
        """递归查找各目录的 SKILL.md，构建 Skill 索引，并拒绝跨目录重复的 ID。"""
        skills: dict[str, Skill] = {}
        for root in roots:
            for skill_md in sorted(root.rglob("SKILL.md")):
                skill = load_skill_folder(skill_md.parent)
                if skill.id in skills:
                    raise SkillLoadError(f"Duplicate skill id: {skill.id}")
                skills[skill.id] = skill
        return skills


@log_call("folder")
def load_skill_folder(folder: Path) -> Skill:
    """合并 SKILL.md 说明和 skill.json 执行图，返回可执行 Skill。

    文件缺失或执行步骤为空时抛出 SkillLoadError；配置值优先于 frontmatter 的同名信息。
    """
    skill_md = folder / "SKILL.md"
    skill_json = folder / "skill.json"
    if not skill_md.exists():
        raise SkillLoadError(f"Missing SKILL.md in {folder}")
    if not skill_json.exists():
        raise SkillLoadError(f"Missing skill.json in {folder}")

    frontmatter, instructions = parse_skill_md(skill_md.read_text(encoding="utf-8"))
    config = json.loads(skill_json.read_text(encoding="utf-8"))
    steps = tuple(_step_from_dict(item) for item in config.get("steps", []))
    if not steps:
        raise SkillLoadError(f"Skill {folder} has no executable steps")

    skill_id = config.get("id") or frontmatter.get("name") or folder.name
    return Skill(
        id=str(skill_id),
        name=str(config.get("name") or frontmatter.get("name") or skill_id),
        description=str(config.get("description") or frontmatter.get("description") or ""),
        mode=str(config.get("mode", "问答")),
        intent_examples=tuple(config.get("intent_examples", [])),
        steps=steps,
        composer_template=str(config.get("composer_template", "")),
        result_type=str(config.get("result_type", "generic")),
        visibility=str(config.get("visibility", "private")),
        allowed_agents=tuple(config.get("allowed_agents", [])),
        instructions=instructions.strip(),
        root_path=folder,
    )


@log_call("text", level=logging.DEBUG)
def parse_skill_md(text: str) -> tuple[dict[str, str], str]:
    """解析由 --- 包围的简单键值元信息，返回元信息字典和 Markdown 正文。

    这里只支持单行键值，不是完整 YAML 解析器；没有元信息时保留全文，未闭合时抛出 SkillLoadError。
    """
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines()
    frontmatter: dict[str, str] = {}
    end = None
    for idx, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end = idx
            break
        if ":" in line:
            key, value = line.split(":", 1)
            frontmatter[key.strip()] = value.strip().strip('"')
    if end is None:
        raise SkillLoadError("SKILL.md frontmatter is not closed with ---")
    return frontmatter, "\n".join(lines[end + 1 :])


def _step_from_dict(item: dict[str, Any]) -> SkillStep:
    """将执行图中的步骤字典转换为 SkillStep，统一步骤类型、输入键及 no_llm 默认值。"""
    return SkillStep(
        id=str(item["id"]),
        name=str(item["name"]),
        kind=StepKind(str(item["kind"])),
        tool_id=item.get("tool_id"),
        input_keys=tuple(item.get("input_keys", [])),
        output_key=str(item["output_key"]),
        no_llm=bool(item.get("no_llm", True)),
    )
