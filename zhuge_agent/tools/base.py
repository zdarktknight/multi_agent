"""Tool 基础接口，供 Runtime 和具体工具实现共同依赖。"""

from __future__ import annotations

import abc
from typing import Any

from ..schema import ToolSpec


class Tool(abc.ABC):
    """工具扩展接口：通过 spec 声明能力与 LLM 边界，通过 invoke 返回结构化结果。"""

    spec: ToolSpec

    @abc.abstractmethod
    def invoke(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """执行来自 Skill 输入键的参数，返回业务结果及可选 evidence；具体逻辑由子类实现。"""
        raise NotImplementedError
