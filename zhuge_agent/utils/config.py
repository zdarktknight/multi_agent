"""读取并校验本地 LLM 配置，不依赖环境变量或额外解析库。"""

from __future__ import annotations

import json
from configparser import ConfigParser, Error as ConfigParserError
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .logging_utils import log_call, register_secret

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.ini"


class ConfigError(ValueError):
    """配置文件缺失、格式错误或字段不合法。"""


@dataclass(frozen=True)
class LLMConfig:
    """LLM 客户端配置；密钥不出现在数据类的默认字符串表示中。"""

    provider: str
    api_key: str = field(default="", repr=False)
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    timeout_seconds: int = 60
    transport: str = "urllib"
    verify_ssl: bool = True
    extra_body: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LoggingConfig:
    """日志级别和可选文件输出路径，未配置时使用 INFO 和控制台输出。"""

    level: str = "INFO"
    file: Path | None = None


def _read_config(path: str | Path | None) -> tuple[Path, ConfigParser]:
    """读取 UTF-8 INI 并隐藏解析器的原始错误内容，避免配置密钥出现在异常中。"""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    parser = ConfigParser(interpolation=None)
    try:
        with config_path.open(encoding="utf-8") as config_file:
            parser.read_file(config_file)
    except (OSError, UnicodeError, ConfigParserError):
        raise ConfigError(
            f"无法读取配置文件 {config_path}，请确认文件存在且为有效的 UTF-8 INI。"
        ) from None
    return config_path, parser


@log_call("path")
def load_logging_config(path: str | Path | None = None) -> LoggingConfig:
    """读取可选 [logging] 段，日志相对路径按配置文件所在目录解析，无该段时使用默认值。"""
    config_path, parser = _read_config(path)
    if not parser.has_section("logging"):
        return LoggingConfig()
    section = parser["logging"]
    level = section.get("level", "INFO").strip().upper()
    if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise ConfigError("logging.level 必须为 DEBUG、INFO、WARNING、ERROR 或 CRITICAL。")
    file_value = section.get("file", "").strip()
    file_path = Path(file_value) if file_value else None
    if file_path is not None and not file_path.is_absolute():
        file_path = (config_path.parent / file_path).resolve()
    return LoggingConfig(level=level, file=file_path)


@log_call("path")
def load_llm_config(path: str | Path | None = None) -> LLMConfig:
    """读取指定 INI 或项目根目录 config.ini，返回校验后的 LLM 配置。

    必须显式选择 openai 或 fixture 模式；真实模式还需配置密钥、地址、模型和正整数超时。
    配置错误统一抛出 ConfigError，错误信息不包含文件内容或密钥，不自动降级为离线模式。
    """
    _, parser = _read_config(path)
    if not parser.has_section("llm"):
        raise ConfigError("配置文件必须包含 [llm] 段。")
    section = parser["llm"]
    provider = section.get("provider", "").strip().lower()
    if provider not in {"openai", "fixture"}:
        raise ConfigError("llm.provider 必须为 openai 或 fixture。")
    if provider == "fixture":
        return LLMConfig(provider=provider)

    api_key = section.get("api_key", "").strip()
    register_secret(api_key)
    base_url = section.get("base_url", "https://api.openai.com/v1").strip().rstrip("/")
    model = section.get("model", "gpt-4o-mini").strip()
    transport = section.get("transport", "urllib").strip().lower()
    if not api_key:
        raise ConfigError("真实 LLM 模式缺少 llm.api_key，请在配置文件中填写密钥。")
    try:
        url = urlsplit(base_url)
        valid_url = url.scheme in {"http", "https"} and bool(url.hostname) and not url.query and not url.fragment
    except ValueError:
        valid_url = False
    if not valid_url:
        raise ConfigError("llm.base_url 必须为有效的 HTTP 或 HTTPS 服务地址，且不能包含查询参数或片段。")
    if not model:
        raise ConfigError("llm.model 不能为空。")
    if transport not in {"urllib", "openai_sdk"}:
        raise ConfigError("llm.transport 必须为 urllib 或 openai_sdk。")
    try:
        timeout = section.getint("timeout_seconds", 60)
    except ValueError:
        raise ConfigError("llm.timeout_seconds 必须为正整数。") from None
    if timeout <= 0:
        raise ConfigError("llm.timeout_seconds 必须为正整数。")
    try:
        verify_ssl = section.getboolean("verify_ssl", True)
    except ValueError:
        raise ConfigError("llm.verify_ssl 必须为 true 或 false。") from None
    extra_body = _load_extra_body(section.get("extra_body", "").strip())
    return LLMConfig(
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout_seconds=timeout,
        transport=transport,
        verify_ssl=verify_ssl,
        extra_body=extra_body,
    )


def _load_extra_body(raw_value: str) -> dict[str, Any]:
    """解析 OpenAI 兼容服务的扩展请求体参数，空值表示不传额外参数。"""
    if not raw_value:
        return {}
    try:
        extra_body = json.loads(raw_value)
    except json.JSONDecodeError:
        raise ConfigError("llm.extra_body 必须为合法 JSON 对象。") from None
    if not isinstance(extra_body, dict):
        raise ConfigError("llm.extra_body 必须为 JSON 对象。")
    return extra_body
