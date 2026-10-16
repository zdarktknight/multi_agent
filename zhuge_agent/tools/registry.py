"""工具注册表，集中管理 Tool ID 到具体实现的映射。"""

from __future__ import annotations

import logging
from typing import Any

from ..utils.logging_utils import log_call
from .base import Tool
from .demo.business import (
    BusinessInsightTool,
    MetricSemanticTool,
    QueryPlanTool,
    QueryPolicyTool,
    VisualizationSpecTool,
    WarehouseQueryTool,
)
from .demo.industrial import (
    DcsCurrentValuesTool,
    DcsHistoryTool,
    ForecastModelTool,
    KnowledgeSearchTool,
    RuleEngineTool,
)


class ToolRegistry:
    """维护工具 ID 到实现的映射，为 Runtime 提供统一执行入口。"""

    def __init__(self) -> None:
        """创建空工具注册表，具体能力由 register 或 default 注入。"""
        self._tools: dict[str, Tool] = {}

    @log_call("tool", level=logging.DEBUG)
    def register(self, tool: Tool) -> None:
        """按 tool.spec.id 注册工具；已有同名工具时替换其实现。"""
        self._tools[tool.spec.id] = tool

    @log_call("tool_id", level=logging.DEBUG)
    def get(self, tool_id: str) -> Tool:
        """按 ID 返回工具实例；未注册时抛出带工具 ID 的 KeyError。"""
        if tool_id not in self._tools:
            raise KeyError(f"Tool not registered: {tool_id}")
        return self._tools[tool_id]

    @classmethod
    @log_call(level=logging.DEBUG)
    def default(cls, case_data: dict[str, Any] | None = None) -> "ToolRegistry":
        """注册内置 demo 工业和经营分析工具，并向支持夹具的工具传入 case_data。"""
        registry = cls()
        registry.register(DcsHistoryTool(case_data))
        registry.register(DcsCurrentValuesTool(case_data))
        registry.register(ForecastModelTool())
        registry.register(RuleEngineTool())
        registry.register(KnowledgeSearchTool(case_data))
        registry.register(MetricSemanticTool(case_data))
        registry.register(QueryPlanTool(case_data))
        registry.register(QueryPolicyTool(case_data))
        registry.register(WarehouseQueryTool(case_data))
        registry.register(VisualizationSpecTool())
        registry.register(BusinessInsightTool())
        return registry
