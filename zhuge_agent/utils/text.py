"""文本处理类公共函数，供路由、Skill 匹配等模块复用。"""

from __future__ import annotations


def tokenize_for_match(text: str) -> list[str]:
    """按空白和常见中英文标点切分文本，返回非空匹配词列表。"""
    separators = " ，。！？,.?;；:/\\|()（）[]【】"
    for sep in separators:
        text = text.replace(sep, " ")
    return [part.strip() for part in text.split() if part.strip()]
