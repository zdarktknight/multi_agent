"""工业场景 demo 工具，使用案例夹具或确定性默认数据模拟真实 DCS/模型/知识库。"""

from __future__ import annotations

import math
from typing import Any

from ...schema import Evidence, StepKind, ToolSpec
from ...utils.logging_utils import log_call
from ..base import Tool


class DcsHistoryTool(Tool):
    """工业历史位号数据 demo 适配器；当前支持案例数据和确定性的演示时序。"""

    spec = ToolSpec(
        id="dcs.history",
        name="DCS / 时序库",
        kind=StepKind.DATA,
        description="读取装置实时与历史位号数据。",
    )

    def __init__(self, case_data: dict[str, Any] | None = None) -> None:
        """保存可选案例数据，其中 dcs_data 用于替换默认演示时序。"""
        self.case_data = case_data or {}

    @log_call("inputs")
    def invoke(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """优先返回案例 dcs_data，否则按 context 的装置和小时窗口构造演示数据及来源证据。"""
        if self.case_data.get("dcs_data"):
            data = dict(self.case_data["dcs_data"])
            evidence = data.pop("evidence", None) or [
                Evidence(
                    source="DCS historian",
                    detail=f"{data.get('asset', '装置')} 案例时序数据，点数 {len(data.get('points', []))}。",
                )
            ]
            return {**data, "evidence": evidence}

        context = inputs.get("context", inputs)
        reactor = context.get("asset", "2# 反应釜")
        hours = int(context.get("hours", 72))
        points = []
        for i in range(hours):
            cooling_drop = 0.18 if i >= hours - 54 else 0
            temp = 128.1 + max(0, i - 18) * 0.055 + math.sin(i / 6) * 0.22
            feed_temp = 41.8 + max(0, i - 32) * 0.018
            points.append(
                {
                    "hour": i,
                    "reactor_temp": round(temp, 2),
                    "cooling_flow": round(100 * (1 - cooling_drop), 2),
                    "feed_temp": round(feed_temp, 2),
                    "agitator_current": round(42.0 + math.sin(i / 4) * 0.4, 2),
                }
            )
        return {
            "asset": reactor,
            "window_hours": hours,
            "points": points,
            "quality": {"complete_rate": 0.997, "sample_interval": "1 min"},
            "evidence": [
                Evidence(
                    source="DCS historian",
                    detail=f"{reactor} 最近 {hours}h：釜温、冷却水流量、进料温度、搅拌电流。",
                )
            ],
        }


class DcsCurrentValuesTool(Tool):
    """当前测点查询 demo 适配器，输出位号数值、单位、时间、质量和来源。"""

    spec = ToolSpec(
        id="dcs.current_values",
        name="DCS / 当前位号值",
        kind=StepKind.DATA,
        description="读取装置当前或最近时刻的温度、压力、流量、液位等位号数值。",
    )

    def __init__(self, case_data: dict[str, Any] | None = None) -> None:
        """保存可选案例数据，其中 current_values 提供测点查询结果。"""
        self.case_data = case_data or {}

    @log_call("inputs")
    def invoke(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """读取案例测点结果并附加证据；没有案例时返回 context 装置的固定演示值。"""
        if self.case_data.get("current_values"):
            data = dict(self.case_data["current_values"])
            values = data.get("values", [])
            return {
                **data,
                "evidence": [
                    Evidence(
                        source="DCS current values",
                        detail=f"{data.get('asset', '装置')} 当前值，读取 {len(values)} 个位号。",
                    )
                ],
            }

        context = inputs.get("context", inputs)
        asset = context.get("asset", "2# 反应釜")
        return {
            "asset": asset,
            "timestamp": "2026-10-12 10:30:00",
            "values": [
                {"name": "温度", "tag": "TI-201A", "value": 130.9, "unit": "℃", "quality": "good"},
                {"name": "压力", "tag": "PI-201", "value": 0.42, "unit": "MPa", "quality": "good"},
            ],
            "source": "DCS historian",
            "evidence": [
                Evidence(source="DCS current values", detail=f"{asset} 当前值，读取 2 个位号。")
            ],
        }


class ForecastModelTool(Tool):
    """基于历史温度斜率生成演示预测，数值由确定性公式产生。"""

    spec = ToolSpec(
        id="model.temp_forecast",
        name="小模型服务 / 趋势预测",
        kind=StepKind.MODEL,
        description="固定版本时序模型，预测温度越界概率。",
    )

    @log_call("inputs")
    def invoke(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """从非空 dcs_data.points 估算两小时温度、趋势和越界概率，返回模型版本及证据。"""
        series = inputs["dcs_data"]["points"]
        last_temp = series[-1]["reactor_temp"]
        lag = min(12, max(1, len(series) - 1))
        slope = (series[-1]["reactor_temp"] - series[-lag]["reactor_temp"]) / lag
        forecast_2h = round(last_temp + slope * 2, 2)
        threshold = 130.0
        over_limit_probability = min(0.95, max(0.05, 0.5 + (forecast_2h - threshold) * 0.18))
        return {
            "model_version": "TS-v2.1.0",
            "forecast_2h": forecast_2h,
            "threshold": threshold,
            "over_limit_probability": round(over_limit_probability, 2),
            "trend": "rising" if slope > 0 else "stable",
            "evidence": [
                Evidence(
                    source="model.temp_forecast@TS-v2.1.0",
                    detail=f"2h 预测 {forecast_2h} ℃，阈值 {threshold} ℃，越界概率 {round(over_limit_probability, 2)}。",
                )
            ],
        }


class RuleEngineTool(Tool):
    """依据演示工艺规则判断温度异常原因，并给出需要人工确认的操作建议。"""

    spec = ToolSpec(
        id="rule.reactor_temp",
        name="SOP 阈值 / 工艺卡",
        kind=StepKind.RULE,
        description="按工艺规则判断异常原因和操作边界。",
    )

    @log_call("inputs")
    def invoke(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """比较历史与近期冷却流量、进料温度，结合 forecast 返回归因、建议和证据。"""
        series = inputs["dcs_data"]["points"]
        forecast = inputs["forecast"]
        width = min(12, max(1, len(series) // 2))
        cooling_baseline = sum(p["cooling_flow"] for p in series[:width]) / width
        cooling_recent = sum(p["cooling_flow"] for p in series[-width:]) / width
        cooling_drop_rate = round((cooling_baseline - cooling_recent) / cooling_baseline, 3)
        feed_delta = round(series[-1]["feed_temp"] - series[0]["feed_temp"], 2)
        root_cause = "cooling_side_capacity_drop" if cooling_drop_rate >= 0.12 else "needs_manual_review"
        actions = [
            "核查冷却水总管压力与过滤器压差",
            "检查夹套换热结垢或阀门开度异常",
            "将进料预热温度下调 3-5 ℃，经工艺负责人确认后执行",
        ]
        if forecast["over_limit_probability"] >= 0.7:
            actions.append("将温度控制回路纳入临时重点监控，报警阈值调整需走业务系统确认")
        return {
            "rule_version": "SOP-REACTOR-TEMP-v4.2",
            "root_cause": root_cause,
            "cooling_drop_rate": cooling_drop_rate,
            "feed_delta": feed_delta,
            "actions": actions,
            "hard_limits": ["禁止直接写入 DCS 控制参数", "涉及报警阈值变更必须人工确认"],
            "evidence": [
                Evidence(
                    source="rule.reactor_temp@SOP-REACTOR-TEMP-v4.2",
                    detail=f"冷却水流量下降 {cooling_drop_rate:.1%}，进料温度上升 {feed_delta} ℃。",
                )
            ],
        }


class KnowledgeSearchTool(Tool):
    """相似处置案例查询 demo 适配器；当前返回案例夹具或固定演示命中结果。"""

    spec = ToolSpec(
        id="knowledge.case_search",
        name="向量知识库 / 案例库",
        kind=StepKind.KNOWLEDGE,
        description="检索历史相似处置案例。",
    )

    def __init__(self, case_data: dict[str, Any] | None = None) -> None:
        """保存可选 knowledge_hits，用于替换默认案例检索结果。"""
        self.case_data = case_data or {}

    @log_call("inputs")
    def invoke(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """返回相似案例和命中证据；当前不使用 inputs 发起真实向量检索。"""
        if self.case_data.get("knowledge_hits"):
            hits = self.case_data["knowledge_hits"]
            return {
                "hits": hits,
                "evidence": [
                    Evidence(
                        source="knowledge.case_search",
                        detail=f"命中 {len(hits)} 条相似案例，最高相似度 {hits[0]['similarity']}。",
                    )
                ],
            }
        return {
            "hits": [
                {
                    "case_id": "CASE-2025-11-014",
                    "similarity": 0.86,
                    "summary": "冷却水管结垢导致夹套换热能力下降，表现为釜温缓慢爬升。",
                },
                {
                    "case_id": "CASE-2026-03-008",
                    "similarity": 0.71,
                    "summary": "过滤器堵塞导致冷却水流量不足，清洗后温度恢复。",
                },
            ],
            "evidence": [
                Evidence(
                    source="knowledge.case_search",
                    detail="命中 2 条相似案例，最高相似度 0.86。",
                )
            ],
        }
