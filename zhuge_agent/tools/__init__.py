"""工具层公开入口，区分基础接口、注册表和 demo 工具实现。"""

from .base import Tool
from .registry import ToolRegistry

__all__ = ["Tool", "ToolRegistry"]
