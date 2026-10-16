"""统一日志格式、会话上下文和函数输入输出记录，避免日志泄露密钥或消费延迟提示词。"""

from __future__ import annotations

import inspect
import json
import logging
import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import fields, is_dataclass
from enum import Enum
from functools import wraps
from pathlib import Path
from threading import RLock
from time import perf_counter
from typing import Any, Callable, Iterator, TextIO, TypeVar
from uuid import uuid4


LOG_FORMAT = (
    "%(agent_id)s %(session_id)s %(asctime)s.%(msecs)03d %(levelname)s "
    "%(module)s@%(lineno)d - %(message)s"
)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_AGENT_ID: ContextVar[str] = ContextVar("zhuge_agent_id", default="-")
_SESSION_ID: ContextVar[str] = ContextVar("zhuge_session_id", default="-")
_LOCK = RLock()
_SECRETS: set[str] = set()
_SENSITIVE_KEYS = {"apikey", "authorization", "password", "secret", "token", "accesstoken", "connectionstring"}
_CREDENTIAL_PATTERN = re.compile(
    r"(?i)([\"']?(?:api[_-]?key|authorization|password|secret|(?:access[_-]?)?token)[\"']?\s*[:=]\s*)"
    r"(?:\"[^\"]*\"|'[^']*'|[^\s,;}]+)"
)
_BEARER_PATTERN = re.compile(r"(?i)\bBearer\s+[^\s\"',;}]+")
_CONFIGURED = False
_Result = TypeVar("_Result")


def register_secret(value: str) -> None:
    """登记实际密钥，使其即使出现在普通文本或异常信息中也会被日志脱敏。"""
    if value:
        with _LOCK:
            _SECRETS.add(value)


def redact_text(text: str) -> str:
    """脱敏已登记密钥以及常见凭证字段，并把换行转义为单行日志内容。"""
    with _LOCK:
        secrets = sorted(_SECRETS, key=len, reverse=True)
    for secret in secrets:
        text = text.replace(secret, "***")
    text = _CREDENTIAL_PATTERN.sub(r"\1***", text)
    text = _BEARER_PATTERN.sub("Bearer ***", text)
    return text.replace("\r", "\\r").replace("\n", "\\n")


def current_session_id() -> str:
    """返回当前请求的会话 ID；未绑定请求时返回占位符。"""
    return _SESSION_ID.get()


@contextmanager
def log_context(*, agent_id: str | None = None, session_id: str | None = None) -> Iterator[None]:
    """临时绑定智能体和会话身份，退出或异常时恢复上层上下文，隔离并发请求。"""
    tokens = []
    try:
        if agent_id is not None:
            tokens.append((_AGENT_ID, _AGENT_ID.set(agent_id)))
        if session_id is not None:
            tokens.append((_SESSION_ID, _SESSION_ID.set(session_id)))
        yield
    finally:
        for variable, token in reversed(tokens):
            variable.reset(token)


class AgentLogFormatter(logging.Formatter):
    """按统一格式动态填入当前身份，并对最终日志文本和异常堆栈进行脱敏。"""

    def __init__(self) -> None:
        """使用带毫秒、级别、模块和行号的项目统一格式。"""
        super().__init__(LOG_FORMAT, datefmt=DATE_FORMAT)

    def format(self, record: logging.LogRecord) -> str:
        """补齐上下文字段；没有上下文的启动或第三方日志使用占位身份。"""
        record.agent_id = getattr(record, "agent_id", _AGENT_ID.get())
        record.session_id = getattr(record, "session_id", _SESSION_ID.get())
        return redact_text(super().format(record))


def configure_logging(
    level: str | int = "INFO", *, stream: TextIO | None = None, log_file: str | Path | None = None
) -> None:
    """配置全局日志到 stderr 和可选 UTF-8 文件，重复调用仅替换项目自身的 Handler。

    保留宿主应用已有 Handler；可通过 stream 注入输出流供测试或嵌入应用使用。
    """
    global _CONFIGURED  # pylint: disable=global-statement
    numeric_level = logging.getLevelName(level.upper()) if isinstance(level, str) else level
    if not isinstance(numeric_level, int):
        raise ValueError("日志级别无效。")
    formatter = AgentLogFormatter()
    handlers: list[logging.Handler] = [logging.StreamHandler(stream)]
    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))
    for index, handler in enumerate(handlers):
        handler.set_name(f"zhuge_{index}")
        handler.setFormatter(formatter)
        handler.setLevel(numeric_level)
    with _LOCK:
        root = logging.getLogger()
        for handler in root.handlers[:]:
            if (handler.name or "").startswith("zhuge_"):
                root.removeHandler(handler)
                handler.close()
        root.setLevel(numeric_level)
        for handler in handlers:
            root.addHandler(handler)
        _CONFIGURED = True


def ensure_logging_configured() -> None:
    """直接使用 Runtime 时按 INFO 初始化控制台日志，保留已配置的日志级别和输出目标。"""
    with _LOCK:
        if not _CONFIGURED:
            configure_logging()


def log_value(value: Any, depth: int = 0) -> Any:
    """生成有限长度的可序列化摘要，递归脱敏凭证，并避免调用任意对象的 str 或 repr。

    列表最多保留十项、字典最多二十项，字符串最多一千五百字符；大数据额外记录条目数。
    未知对象仅记录类型，所以 LazyPrompt 不会因日志而提前渲染。
    """
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        text = redact_text(value)
        return text if len(text) <= 1500 else text[:1500] + f"... <length={len(text)}>"
    if isinstance(value, Path):
        return redact_text(str(value))
    if isinstance(value, Enum):
        return log_value(value.value, depth + 1)
    if depth > 6:
        return f"<{type(value).__name__}:depth-limit>"
    if is_dataclass(value) and not isinstance(value, type):
        return log_value({field.name: getattr(value, field.name) for field in fields(value)}, depth)
    if isinstance(value, dict):
        output = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= 20:
                output["_total_keys"] = len(value)
                break
            name = key if isinstance(key, str) else f"<{type(key).__name__}>"
            sensitive = name.lower().replace("_", "").replace("-", "") in _SENSITIVE_KEYS
            output[redact_text(name)] = "***" if sensitive else log_value(item, depth + 1)
        return output
    if isinstance(value, (list, tuple)):
        result = [log_value(item, depth + 1) for item in value[:10]]
        if len(value) > 10:
            result.append({"_total_items": len(value)})
        return result
    return f"<{type(value).__name__}>"


def _call_record(logger: logging.Logger, function: Callable, level: int, message: str, payload: Any) -> None:
    """使用被记录函数的真实模块和定义行号写日志，而非装饰器的模块和行号。"""
    if logger.isEnabledFor(level):
        data = json.dumps(log_value(payload), ensure_ascii=False, separators=(",", ":"))
        if len(data) > 12000:
            data = data[:12000] + f"... <length={len(data)}>"
        record = logger.makeRecord(
            logger.name, level, function.__code__.co_filename, function.__code__.co_firstlineno,
            "%s function=%s %s", (message, function.__qualname__, data), None,
            func=function.__name__,
            extra={"agent_id": _AGENT_ID.get(), "session_id": _SESSION_ID.get()},
        )
        logger.handle(record)


def log_call(
    *input_names: str, level: int = logging.INFO, context: str | None = None
) -> Callable[[Callable[..., _Result]], Callable[..., _Result]]:
    """记录关键入参、输出摘要、耗时及异常，保留函数签名、返回值和异常传播行为。

    context=request 自动创建或复用会话并绑定主智能体；context=agent 从配置绑定子智能体。
    不用于生成器：流式入口在工作线程调用已记录的请求函数，避免上下文跨 yield 泄露。
    """
    def decorate(function: Callable[..., _Result]) -> Callable[..., _Result]:
        """缓存函数签名和模块 Logger，构建记录包装器。"""
        signature = inspect.signature(function)
        logger = logging.getLogger(function.__module__)

        @wraps(function)
        def wrapped(*args: Any, **kwargs: Any) -> _Result:
            """绑定关键参数，在正确身份下记录函数开始、完成或异常。"""
            arguments = signature.bind(*args, **kwargs).arguments
            agent_id = None
            session_id = None
            if context == "request":
                agent_id = "MasterAgent"
                session_id = arguments.get("session_id") or _SESSION_ID.get()
                if session_id == "-":
                    session_id = uuid4().hex
            elif context == "agent":
                spec = arguments.get("agent") or getattr(arguments.get("self"), "spec", None)
                agent_id = spec.id if spec is not None else None
            with log_context(agent_id=agent_id, session_id=session_id):
                inputs = {name: arguments[name] for name in input_names if name in arguments}
                _call_record(logger, function, level, "函数输入", {"inputs": inputs})
                start = perf_counter()
                try:
                    result = function(*args, **kwargs)
                except Exception as exc:
                    _call_record(logger, function, logging.ERROR, "函数异常", {
                        "error_type": type(exc).__name__, "error": str(exc),
                        "elapsed_ms": round((perf_counter() - start) * 1000, 3),
                    })
                    raise
                _call_record(logger, function, level, "函数输出", {
                    "output": result, "elapsed_ms": round((perf_counter() - start) * 1000, 3),
                })
                return result

        return wrapped

    return decorate
