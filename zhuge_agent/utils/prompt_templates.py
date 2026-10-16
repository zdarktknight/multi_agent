from __future__ import annotations

import logging
import re
from functools import cached_property
from pathlib import Path
from typing import Any, Callable

from jinja2 import Template

from .logging_utils import log_call

class PromptTemplateError(RuntimeError):
    pass


class AgentPromptTemplate:
    """按 Markdown 注释标记拆分提示词，并缓存各段的 Jinja2 编译结果。"""

    _SECTION_PATTERN = re.compile(r"<!--\s*([\w_]+)\s*-->\s*([\s\S]*?)(?=(<!--\s*[\w_]+\s*-->|$))")

    @log_call("path", level=logging.DEBUG)
    def __init__(self, path: Path) -> None:
        """读取模板段落但不渲染内容；文件不存在或没有命名段落时抛出 PromptTemplateError。"""
        self.path = path
        if not self.path.exists():
            raise PromptTemplateError(f"Prompt template not found: {path}")
        self._sections = self._load_sections(path)
        self._compiled: dict[str, Template] = {}

    @log_call("section", "variables", level=logging.DEBUG)
    def lazy_render(self, section: str, fallback: str, **variables: Any) -> "LazyPrompt":
        """捕获段落名、备用模板及变量，返回被消费时才执行渲染的 LazyPrompt。"""
        return LazyPrompt(lambda: self.render(section, fallback, **variables))

    @log_call("section", "variables", level=logging.DEBUG)
    def render(self, section: str, fallback: str, **variables: Any) -> str:
        """用变量立即渲染指定段落，去除结果首尾空白后返回提示词文本。"""
        template = self._template(section, fallback)
        return template.render(**variables).strip()

    def _template(self, section: str, fallback: str) -> Template:
        """获取或首次编译段落模板；缺失时使用 fallback，两者均为空时抛出 PromptTemplateError。"""
        if section not in self._compiled:
            source = self._sections.get(section, fallback)
            if not source:
                raise PromptTemplateError(f"Prompt section '{section}' missing in {self.path}")
            self._compiled[section] = Template(source)
        return self._compiled[section]

    @classmethod
    @log_call("path", level=logging.DEBUG)
    def _load_sections(cls, path: Path) -> dict[str, str]:
        """从 UTF-8 Markdown 中提取 <!-- 段落名 --> 后的文本，返回段落名到模板源码的映射。"""
        content = path.read_text(encoding="utf-8")
        sections: dict[str, str] = {}
        for match in cls._SECTION_PATTERN.finditer(content):
            sections[match.group(1).strip()] = match.group(2).strip()
        if not sections:
            raise PromptTemplateError(f"Prompt template has no named sections: {path}")
        return sections


class LazyPrompt:
    """首次消费时渲染并缓存提示词，避免构建智能体时提前生成未使用的文本。"""

    def __init__(self, render: Callable[[], str]) -> None:
        """保存无参渲染回调，初始化时不执行回调。"""
        self._render = render

    @cached_property
    def text(self) -> str:
        """首次访问时执行渲染，后续访问返回缓存文本。"""
        return self._render()

    def __str__(self) -> str:
        """在 LLM 客户端将提示词转成字符串时触发首次渲染，并复用缓存。"""
        return self.text

    def __bool__(self) -> bool:
        """消费提示词并判断渲染结果是否非空；布尔判断也会触发首次渲染。"""
        return bool(self.text)
