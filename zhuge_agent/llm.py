from __future__ import annotations

import importlib
import json
import logging
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterator, Protocol

from .utils.config import load_llm_config
from .utils.logging_utils import log_call, register_secret


class LLMError(RuntimeError):
    """LLM 调用、传输、解析或配置初始化失败时抛出的统一异常。"""


class LLMClient(Protocol):
    """运行时依赖的 LLM 能力协议，隔离真实 API 客户端与离线 fixture 实现。"""

    def chat_json(
        self,
        *,
        system: object,
        user: str,
        schema_hint: dict[str, Any],
        temperature: float = 0.0,
        extra_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """约定结构化 LLM 调用接口，返回供路由、规划或回答生成使用的 JSON 结果。

        system 可以是 LazyPrompt，实现应在消费时转成文本；user 为请求内容，
        schema_hint 描述期望结果形状，temperature 控制生成温度，extra_body 承载模型服务扩展参数。
        """
        raise NotImplementedError

    def stream_text(
        self,
        *,
        system: object,
        user: str,
        temperature: float = 0.2,
        extra_body: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        """约定自然语言流式调用接口，逐段返回最终回答文本。

        本接口只用于 direct answer 或 Composer 最终文案，不用于 Router、Planner 或确定性工具。
        """
        raise NotImplementedError


class OpenAICompatibleLLMClient:
    """调用 OpenAI 兼容的 Chat Completions API，支持 urllib 和 OpenAI SDK 两种传输。"""

    @log_call("base_url", "model", "timeout_seconds")
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: int = 60,
        transport: str = "urllib",
        verify_ssl: bool = True,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        """使用显式传入的密钥、地址、模型及超时构建客户端，缺少密钥时抛出 LLMError。

        本构造函数不读取环境变量；配置文件入口为 llm_client_from_config。
        """
        self.api_key = api_key
        register_secret(api_key or "")
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.model = model or "gpt-4o-mini"
        self.timeout_seconds = timeout_seconds
        self.transport = transport
        self.verify_ssl = verify_ssl
        self.extra_body = dict(extra_body or {})
        if not self.api_key:
            raise LLMError("缺少 API key，请通过配置文件或显式参数传入密钥。")
        if self.transport not in {"urllib", "openai_sdk"}:
            raise LLMError("transport 必须为 urllib 或 openai_sdk。")

    @log_call("system", "user", "schema_hint", "temperature", "extra_body")
    def chat_json(
        self,
        *,
        system: object,
        user: str,
        schema_hint: dict[str, Any],
        temperature: float = 0.0,
        extra_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """消费系统提示词并发送同步 HTTP 请求，解析模型返回的 JSON 内容。

        schema_hint 作为文字提示附加到用户消息，不执行 JSON Schema 校验。
        extra_body 用于 vLLM/Qwen/LiteLLM 等 OpenAI 兼容服务的非标准参数。
        HTTP 或连接错误转换为 LLMError；本接口一次返回完整结果，不提供 token 流。
        """
        system_prompt = str(system)
        payload = {
            "model": self.model,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"{user}\n\n"
                        "Return a JSON object only. Required shape:\n"
                        f"{json.dumps(schema_hint, ensure_ascii=False)}"
                    ),
                },
            ],
        }
        request_extra_body = _merge_extra_body(self.extra_body, extra_body)
        if self.transport == "openai_sdk":
            return self._chat_json_with_openai_sdk(payload, request_extra_body)
        return self._chat_json_with_urllib(payload, request_extra_body)

    def _chat_json_with_urllib(
        self,
        payload: dict[str, Any],
        extra_body: dict[str, Any],
    ) -> dict[str, Any]:
        """使用标准库 urllib 调用 Chat Completions，并将扩展字段合并进请求体。"""
        payload = {**payload, **extra_body}
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise LLMError(f"LLM API HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise LLMError(f"LLM API connection failed: {exc}") from exc

        content = body["choices"][0]["message"]["content"]
        return parse_json_object(content)

    def _chat_json_with_openai_sdk(
        self,
        payload: dict[str, Any],
        extra_body: dict[str, Any],
    ) -> dict[str, Any]:
        """使用 openai SDK 和 httpx 客户端调用 OpenAI 兼容网关，支持关闭 SSL 校验。

        该路径用于 LiteLLM、企业网关或需要自定义 httpx.Client 的环境；依赖缺失时给出
        明确错误，不回退到其他传输方式，避免真实链路被悄悄替换。
        """
        try:
            httpx = importlib.import_module("httpx")
            openai = importlib.import_module("openai")
        except ImportError as exc:
            raise LLMError("transport=openai_sdk 需要安装 openai 和 httpx。") from exc

        try:
            with httpx.Client(verify=self.verify_ssl, timeout=self.timeout_seconds) as http_client:
                client = openai.OpenAI(
                    base_url=self.base_url,
                    api_key=self.api_key,
                    http_client=http_client,
                )
                response = client.chat.completions.create(
                    **payload,
                    **({"extra_body": extra_body} if extra_body else {}),
                )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            raise LLMError(f"LLM API connection failed: {exc}") from exc

        content = response.choices[0].message.content or ""
        return parse_json_object(content)

    @log_call("system", "user", "temperature", "extra_body")
    def stream_text(
        self,
        *,
        system: object,
        user: str,
        temperature: float = 0.2,
        extra_body: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        """使用模型原生 stream=True 逐段返回自然语言文本。

        urllib transport 解析 OpenAI 兼容 SSE；openai_sdk transport 使用 SDK 的流式迭代。
        """
        system_prompt = str(system)
        payload = {
            "model": self.model,
            "temperature": temperature,
            "stream": True,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user},
            ],
        }
        request_extra_body = _merge_extra_body(self.extra_body, extra_body)
        if self.transport == "openai_sdk":
            yield from self._stream_text_with_openai_sdk(payload, request_extra_body)
        else:
            yield from self._stream_text_with_urllib(payload, request_extra_body)

    def _stream_text_with_urllib(
        self,
        payload: dict[str, Any],
        extra_body: dict[str, Any],
    ) -> Iterator[str]:
        """使用 urllib 读取 OpenAI 兼容 SSE，并逐段产出 delta.content。

        该路径用于不安装 SDK 的环境；只解析 `data:` 行，其余心跳或空行会被跳过。
        """
        payload = {**payload, **extra_body}
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                for raw_line in resp:
                    # OpenAI 兼容流式接口以 SSE 行返回：data: {...}，结束时为 data: [DONE]。
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    chunk = line.removeprefix("data:").strip()
                    if chunk == "[DONE]":
                        break
                    content = _content_from_stream_chunk(json.loads(chunk))
                    if content:
                        yield content
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise LLMError(f"LLM API HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise LLMError(f"LLM API connection failed: {exc}") from exc

    def _stream_text_with_openai_sdk(
        self,
        payload: dict[str, Any],
        extra_body: dict[str, Any],
    ) -> Iterator[str]:
        """使用 OpenAI SDK 的流式响应迭代 delta.content。

        该路径匹配企业 LiteLLM/内网网关需要自定义 httpx.Client 的接入方式。
        """
        try:
            httpx = importlib.import_module("httpx")
            openai = importlib.import_module("openai")
        except ImportError as exc:
            raise LLMError("transport=openai_sdk 需要安装 openai 和 httpx。") from exc

        try:
            with httpx.Client(verify=self.verify_ssl, timeout=self.timeout_seconds) as http_client:
                client = openai.OpenAI(
                    base_url=self.base_url,
                    api_key=self.api_key,
                    http_client=http_client,
                )
                stream = client.chat.completions.create(
                    **payload,
                    **({"extra_body": extra_body} if extra_body else {}),
                )
                for chunk in stream:
                    content = _content_from_sdk_stream_chunk(chunk)
                    if content:
                        yield content
        except Exception as exc:  # pylint: disable=broad-exception-caught
            raise LLMError(f"LLM API connection failed: {exc}") from exc


class FixtureLLMClient:
    """为离线测试和 Demo 提供可重复的结构化响应，不访问真实 LLM 服务。"""

    @log_call("system", "user", "schema_hint", "temperature", "extra_body")
    def chat_json(
        self,
        *,
        system: object,
        user: str,
        schema_hint: dict[str, Any],
        temperature: float = 0.0,
        extra_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """按 schema_hint 区分路由、规划及回答任务，基于输入夹具返回确定性结果。

        回答中的数字复用 user 提供的工具输出；忽略 system 和 temperature，
        因此此客户端不会消费或渲染 LazyPrompt，也不用于验证真实模型表现。
        """
        del system
        del temperature
        del extra_body
        if "execution_mode" in schema_hint:
            payload = parse_json_object(user)
            request = str(payload.get("request", ""))
            agent_ids = []
            if any(word in request for word in ("销售额", "环比", "华东", "城市", "经营指标")):
                agent_ids.append("business_metric_agent")
            if any(word in request for word in ("为什么", "偏高", "超限", "调整", "诊断")):
                agent_ids.append("reactor_diagnosis_agent")
            if any(word in request for word in ("多少", "当前", "现在", "查询", "位号", "压力", "温度")):
                agent_ids.append("dcs_query_agent")
            if not agent_ids:
                if self._is_general_chat(request):
                    return {
                        "execution_mode": "direct",
                        "agent_ids": [],
                        "workflow_id": "",
                        "intent": "通用问答",
                        "confidence": 0.95,
                        "reason": "请求不需要业务子智能体或确定性工具。",
                        "direct_answer": "我很好，随时可以帮你查看诸葛 AI 的业务问题或一起排查系统流程。",
                    }
                return {
                    "execution_mode": "direct",
                    "agent_ids": [],
                    "workflow_id": "",
                    "intent": "未覆盖问题",
                    "confidence": 0.72,
                    "reason": "当前注册子智能体不覆盖该诉求。",
                }
            if len(agent_ids) > 1 and any(word in request for word in ("工作流", "流程", "审批", "人审")):
                return {
                    "execution_mode": "workflow",
                    "agent_ids": agent_ids,
                    "workflow_id": "business_metric_dcs_workflow",
                    "intent": "复合工作流",
                    "confidence": 0.91,
                    "reason": "问题包含多个业务节点，并明确要求流程或人审。",
                }
            return {
                "execution_mode": "parallel" if len(agent_ids) > 1 else "single",
                "agent_ids": agent_ids,
                "workflow_id": "",
                "intent": "自动并行委派" if len(agent_ids) > 1 else "自动单智能体委派",
                "confidence": 0.9,
                "reason": "MasterAgent 根据问题中的业务域和依赖关系选择执行模式。",
            }
        if "agent_ids" in schema_hint:
            payload = parse_json_object(user)
            request = str(payload.get("request", ""))
            agent_ids = []
            if any(word in request for word in ("销售额", "环比", "华东", "城市", "经营指标")):
                agent_ids.append("business_metric_agent")
            if any(word in request for word in ("为什么", "偏高", "超限", "调整", "诊断")):
                agent_ids.append("reactor_diagnosis_agent")
            if any(word in request for word in ("多少", "当前", "现在", "查询", "位号", "压力", "温度")):
                agent_ids.append("dcs_query_agent")
            if not agent_ids:
                if self._is_general_chat(request):
                    return {
                        "route_type": "direct",
                        "agent_ids": [],
                        "intent": "通用问答",
                        "confidence": 0.95,
                        "reason": "请求不需要业务子智能体或确定性工具。",
                        "direct_answer": "我很好，随时可以帮你查看诸葛 AI 的业务问题或一起排查系统流程。",
                    }
                return {
                    "route_type": "direct",
                    "agent_ids": [],
                    "intent": "未覆盖问题",
                    "confidence": 0.72,
                    "reason": "当前注册子智能体不覆盖该诉求。",
                }
            return {
                "route_type": "delegate",
                "agent_ids": agent_ids,
                "intent": "并行业务委派" if len(agent_ids) > 1 else "单智能体委派",
                "confidence": 0.9,
                "reason": "问题包含多个可并行处理的业务子任务。",
            }
        if "agent_id" in schema_hint:
            payload = parse_json_object(user)
            request = str(payload.get("request", ""))
            if any(word in request for word in ("销售额", "环比", "华东", "城市", "经营指标")):
                return {
                    "route_type": "delegate",
                    "agent_id": "business_metric_agent",
                    "intent": "企业指标分析",
                    "confidence": 0.94,
                    "reason": "问题属于经营指标查询和分析。",
                }
            if any(word in request for word in ("为什么", "偏高", "超限", "调整", "诊断")):
                return {
                    "route_type": "delegate",
                    "agent_id": "reactor_diagnosis_agent",
                    "intent": "反应釜异常诊断",
                    "confidence": 0.92,
                    "reason": "问题需要诊断和建议。",
                }
            if any(word in request for word in ("多少", "当前", "现在", "查询", "位号", "压力", "温度")):
                return {
                    "route_type": "delegate",
                    "agent_id": "dcs_query_agent",
                    "intent": "工业位号查数",
                    "confidence": 0.9,
                    "reason": "问题是在询问当前或最近时刻的装置位号数值。",
                }
            if self._is_general_chat(request):
                return {
                    "route_type": "direct",
                    "agent_id": None,
                    "intent": "通用问答",
                    "confidence": 0.95,
                    "reason": "请求不需要业务子智能体或确定性工具。",
                    "direct_answer": "我很好，随时可以帮你查看诸葛 AI 的业务问题或一起排查系统流程。",
                }
            return {
                "route_type": "direct",
                "agent_id": None,
                "intent": "未覆盖问题",
                "confidence": 0.72,
                "reason": "当前注册子智能体不覆盖该诉求。",
            }
        if "skill_id" in schema_hint:
            payload = parse_json_object(user)
            request = str(payload.get("request", ""))
            if any(word in request for word in ("销售额", "环比", "华东", "城市", "经营指标")):
                return {
                    "skill_id": "business_metric_analysis",
                    "intent": "企业指标语义分析",
                    "confidence": 0.94,
                    "mode": "经营分析",
                    "reason": "问题包含销售额、区域、时间、环比和城市排序要求。",
                }
            if any(word in request for word in ("多少", "当前", "现在", "查询", "位号", "压力")):
                return {
                    "skill_id": "query_tag_value",
                    "intent": "DCS 位号查询",
                    "confidence": 0.93,
                    "mode": "问数",
                    "reason": "问题是在询问当前或最近时刻的装置位号数值。",
                }
            return {
                "skill_id": "reactor_temp_forecast_adjust",
                "intent": "反应釜温度偏高归因与调整建议",
                "confidence": 0.91,
                "mode": "问答 / 推演",
                "reason": "问题包含反应釜、温度偏高和调整建议。",
            }
        if "steps" in schema_hint and "no_llm_steps" in schema_hint:
            payload = parse_json_object(user)
            skill_steps = payload.get("skill", {}).get("steps", [])
            return {
                "steps": [item["name"] for item in skill_steps],
                "no_llm_steps": [item["name"] for item in skill_steps if item.get("no_llm")],
                "reason": "数值、规则和检索均由确定性工具完成。",
            }
        if "answer" in schema_hint:
            payload = parse_json_object(user)
            if payload.get("result_type") == "tag_query":
                current_values = payload["current_values"]
                parts = [
                    f"{item['name']} {item['value']} {item['unit']}（{item['tag']}，质量：{item['quality']}）"
                    for item in current_values["values"]
                ]
                return {
                    "answer": (
                        f"{current_values['asset']} 当前读取值："
                        + "；".join(parts)
                        + f"。数据时间：{current_values['timestamp']}；来源：{current_values['source']}。"
                    )
                }
            if payload.get("result_type") == "business_metric_analysis":
                semantic = payload["semantic_parse"]
                warehouse = payload["warehouse_result"]
                insights = payload["business_insights"]["insights"]
                total = warehouse["totals"]
                rows = warehouse["rows"]
                row_text = "；".join(
                    f"{row['city']} {row['sales_amount']} 元，环比 {row['mom_rate']:.1%}"
                    for row in rows
                )
                return {
                    "answer": (
                        f"按企业指标口径「{semantic['metric']['name']}」统计，"
                        f"{semantic['time']['current']} {semantic['filters']['region']}区域销售额为 "
                        f"{total['sales_amount']} 元，整体环比 {total['mom_rate']:.1%}。"
                        f"按城市排序：{row_text}。"
                        f"可视化已生成柱状图和环比折线图配置。"
                        f"洞察：{' '.join(insights)}"
                    )
                }
            metrics = payload["metrics"]
            context = payload["context"]
            top_case = payload["case_evidence"]["hits"][0]
            return {
                "answer": (
                    f"结论：{context['asset']} 温度偏高的主因更可能是冷却侧换热能力下降。"
                    f"过去 {context['hours']} 小时冷却水流量下降 {metrics['cooling_drop_rate_text']}，"
                    f"2 小时温度预测为 {metrics['forecast_2h']} ℃，"
                    f"越界概率 {metrics['over_limit_probability_text']}。"
                    f"历史案例最高相似度 {top_case['similarity']}，指向冷却水管结垢或过滤器堵塞。"
                )
            }
        return dict(schema_hint)

    @log_call("system", "user", "temperature", "extra_body")
    def stream_text(
        self,
        *,
        system: object,
        user: str,
        temperature: float = 0.2,
        extra_body: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        """离线流式文本，用已有结构化 fixture 结果切片模拟最终回答增量。"""
        del system
        del temperature
        del extra_body
        payload = _parse_json_object_silent(user) or {"request": user}
        if payload.get("result_type"):
            answer = self.chat_json(system="", user=_json_dumps(payload), schema_hint={"answer": "string"})["answer"]
        else:
            answer = "我很好，随时可以帮你查看诸葛 AI 的业务问题或一起排查系统流程。"
        # fixture 不访问网络，用固定长度切片模拟 token 增量，保证测试可重复。
        for index in range(0, len(answer), 4):
            yield answer[index:index + 4]

    def _is_general_chat(self, request: str) -> bool:
        """识别离线 fixture 中的普通寒暄，避免把未知业务诉求误当成闲聊回答。"""
        return any(word in request for word in ("你好", "你好吗", "您好", "在吗", "谢谢"))


@log_call("path")
def llm_client_from_config(path: str | Path | None = None) -> LLMClient:
    """读取 INI 配置并创建真实或离线客户端，配置和初始化错误直接向调用方传播。"""
    config = load_llm_config(path)
    if config.provider == "fixture":
        return FixtureLLMClient()
    return OpenAICompatibleLLMClient(
        api_key=config.api_key,
        base_url=config.base_url,
        model=config.model,
        timeout_seconds=config.timeout_seconds,
        transport=config.transport,
        verify_ssl=config.verify_ssl,
        extra_body=config.extra_body,
    )


@log_call("text", level=logging.DEBUG)
def parse_json_object(text: str) -> dict[str, Any]:
    """先直接解析 JSON，失败时提取文本中花括号包围的片段再次解析。

    没有可提取片段时抛出 LLMError，片段格式错误仍抛出 JSONDecodeError；不校验结果结构。
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise LLMError("LLM response did not contain a JSON object.") from exc
        return json.loads(match.group(0))


def _content_from_stream_chunk(chunk: dict[str, Any]) -> str:
    """从 OpenAI 兼容 SSE JSON chunk 中提取 delta.content。

    流式响应可能包含 role、finish_reason 或空 delta；这些都不应变成展示 token。
    """
    choices = chunk.get("choices") or []
    if not choices:
        return ""
    delta = choices[0].get("delta") or {}
    content = delta.get("content")
    return content if isinstance(content, str) else ""


def _merge_extra_body(
    base: dict[str, Any],
    override: dict[str, Any] | None,
) -> dict[str, Any]:
    """合并配置级和单次调用级扩展请求体参数，嵌套字典递归合并。

    单次调用参数优先级更高；返回新字典，避免修改配置对象或调用方传入对象。
    """
    merged = dict(base)
    for key, value in (override or {}).items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _merge_extra_body(merged[key], value)
        else:
            merged[key] = value
    return merged


def _parse_json_object_silent(text: str) -> dict[str, Any] | None:
    """尽力从文本中提取 JSON 对象；仅供 fixture 使用，失败时不写错误日志。

    Composer 流式 prompt 会在 JSON 后追加自然语言指令，因此这里允许从混合文本中截取对象。
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


def _content_from_sdk_stream_chunk(chunk: object) -> str:
    """从 OpenAI SDK chunk 对象中提取 delta.content，兼容对象和字典两类结构。"""
    if isinstance(chunk, dict):
        return _content_from_stream_chunk(chunk)
    choices = getattr(chunk, "choices", None) or []
    if not choices:
        return ""
    delta = getattr(choices[0], "delta", None)
    content = getattr(delta, "content", "") if delta is not None else ""
    return content if isinstance(content, str) else ""


def _json_dumps(payload: dict[str, Any]) -> str:
    """以 UTF-8 友好的格式序列化 fixture 流式输入。"""
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)
