"""经营分析 demo 工具，模拟指标语义库、查询计划、权限校验、数仓和可视化链路。"""

from __future__ import annotations

from typing import Any

from ...schema import Evidence, StepKind, ToolSpec
from ...utils.logging_utils import log_call
from ..base import Tool


class MetricSemanticTool(Tool):
    """企业指标语义 demo 适配器，输出指标口径、时间、筛选条件、维度和排序要求。"""

    spec = ToolSpec(
        id="semantic.metric_resolver",
        name="企业指标语义库",
        kind=StepKind.SEMANTIC,
        description="识别指标、时间、维度和筛选条件，并绑定统一业务口径。",
        no_llm=False,
    )

    def __init__(self, case_data: dict[str, Any] | None = None) -> None:
        """保存案例中的 semantic_parse，作为语义解析结果的可替换输入。"""
        self.case_data = case_data or {}

    @log_call("inputs")
    def invoke(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """返回案例语义或默认销售额语义，并附加口径证据。

        当前实现不调用 LLM；spec.no_llm=False 表示该适配器允许后续接入 LLM 语义解析。
        """
        if self.case_data.get("semantic_parse"):
            parsed = dict(self.case_data["semantic_parse"])
        else:
            context = inputs.get("context", {})
            request = context.get("request", "")
            parsed = {
                "metric": {
                    "id": "sales_amount",
                    "name": "销售额",
                    "definition": "已审核订单的不含税销售金额汇总",
                    "formula": "sum(net_sales_amount)",
                    "currency": "CNY",
                },
                "time": {"grain": "month", "current": "2026-10", "compare": "2026-09"},
                "filters": {"region": "华东"},
                "dimensions": ["city"],
                "sort": {"field": "sales_amount", "order": "desc"},
                "request": request,
            }
        return {
            **parsed,
            "evidence": [
                Evidence(
                    source="semantic.metric_catalog",
                    detail=f"指标「{parsed['metric']['name']}」绑定口径：{parsed['metric']['definition']}。",
                )
            ],
        }


class QueryPlanTool(Tool):
    """将指标语义转换为结构化查询计划，交给后续策略校验和数仓执行。"""

    spec = ToolSpec(
        id="query.plan_builder",
        name="查询计划生成器",
        kind=StepKind.QUERY,
        description="把指标语义映射到底层数仓表、字段、过滤条件和计算表达式。",
    )

    def __init__(self, case_data: dict[str, Any] | None = None) -> None:
        """保存可选 query_plan，允许通过案例替换默认计划生成结果。"""
        self.case_data = case_data or {}

    @log_call("inputs")
    def invoke(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """根据 semantic_parse 生成表、表达式、筛选、分组及排序计划，或返回案例计划。"""
        semantic = inputs["semantic_parse"]
        if self.case_data.get("query_plan"):
            plan = dict(self.case_data["query_plan"])
        else:
            metric = semantic["metric"]
            plan = {
                "dialect": "analytics-json-plan",
                "source": "dw.fact_sales_order",
                "metric_id": metric["id"],
                "select": [
                    {"expr": "city", "alias": "city"},
                    {"expr": metric["formula"], "alias": metric["id"]},
                    {"expr": f"mom({metric['formula']})", "alias": "mom_rate"},
                ],
                "where": [
                    {"field": "region", "op": "=", "value": semantic["filters"]["region"]},
                    {
                        "field": "biz_month",
                        "op": "in",
                        "value": [semantic["time"]["current"], semantic["time"]["compare"]],
                    },
                    {"field": "order_status", "op": "=", "value": "approved"},
                ],
                "group_by": semantic["dimensions"],
                "order_by": [semantic["sort"]],
            }
        return {
            **plan,
            "evidence": [
                Evidence(
                    source="query.plan_builder",
                    detail=f"映射到 {plan['source']}，按 {','.join(plan['group_by'])} 分组并计算环比。",
                )
            ],
        }


class QueryPolicyTool(Tool):
    """查询策略校验 demo 适配器；当前实现区域访问范围和数据表白名单的演示检查。"""

    spec = ToolSpec(
        id="policy.query_guard",
        name="权限 / 口径校验",
        kind=StepKind.RULE,
        description="校验用户数据域、字段权限、指标口径和查询计划安全边界。",
    )

    def __init__(self, case_data: dict[str, Any] | None = None) -> None:
        """保存案例权限设置，其中 allowed_regions 指定可访问区域。"""
        self.case_data = case_data or {}

    @log_call("inputs")
    def invoke(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """检查 semantic_parse 区域及 query_plan 表名，返回通过标志、违规项、访问主体和证据。"""
        context = inputs["context"]
        semantic = inputs["semantic_parse"]
        plan = inputs["query_plan"]
        allowed_regions = set(self.case_data.get("allowed_regions", ["华东", "华北", "华南"]))
        requested_region = semantic["filters"].get("region")
        violations = []
        if requested_region not in allowed_regions:
            violations.append(f"无权访问区域：{requested_region}")
        if plan["source"] != "dw.fact_sales_order":
            violations.append(f"未授权数据表：{plan['source']}")
        passed = not violations
        return {
            "passed": passed,
            "violations": violations,
            "metric_version": "metric.sales_amount.v2026.10",
            "principal": {"tenant": context.get("tenant"), "role": context.get("role")},
            "evidence": [
                Evidence(
                    source="policy.query_guard",
                    detail="已校验区域数据域、订单状态口径、指标版本和表级权限。"
                    if passed
                    else "查询计划未通过权限或口径校验。",
                )
            ],
        }


class WarehouseQueryTool(Tool):
    """数仓计算 demo 适配器，只有策略校验通过后才提供销售统计结果。"""

    spec = ToolSpec(
        id="warehouse.sales_query",
        name="数仓 / 业务库计算",
        kind=StepKind.WAREHOUSE,
        description="执行已校验查询计划并返回指标表格。",
    )

    def __init__(self, case_data: dict[str, Any] | None = None) -> None:
        """保存可选 warehouse_result，用于返回已准备好的数仓结果夹具。"""
        self.case_data = case_data or {}

    @log_call("inputs")
    def invoke(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """依据 policy_check 决定是否返回结果；通过后读取夹具或计算演示城市销售额和环比。"""
        policy = inputs["policy_check"]
        if not policy["passed"]:
            return {
                "rows": [],
                "totals": {},
                "evidence": [Evidence(source="warehouse.sales_query", detail="权限校验未通过，未执行查询。")],
            }
        if self.case_data.get("warehouse_result"):
            result = dict(self.case_data["warehouse_result"])
        else:
            rows = [
                {"city": "上海", "sales_amount": 12860000, "last_month_sales_amount": 11500000},
                {"city": "杭州", "sales_amount": 9360000, "last_month_sales_amount": 8850000},
                {"city": "南京", "sales_amount": 8120000, "last_month_sales_amount": 9020000},
                {"city": "苏州", "sales_amount": 6980000, "last_month_sales_amount": 6410000},
            ]
            for row in rows:
                row["mom_rate"] = round(
                    (row["sales_amount"] - row["last_month_sales_amount"]) / row["last_month_sales_amount"],
                    4,
                )
            rows.sort(key=lambda item: item["sales_amount"], reverse=True)
            total = sum(row["sales_amount"] for row in rows)
            last_total = sum(row["last_month_sales_amount"] for row in rows)
            result = {
                "currency": "CNY",
                "rows": rows,
                "totals": {
                    "sales_amount": total,
                    "last_month_sales_amount": last_total,
                    "mom_rate": round((total - last_total) / last_total, 4),
                },
            }
        return {
            **result,
            "evidence": [
                Evidence(
                    source="warehouse.sales_query",
                    detail=f"返回 {len(result['rows'])} 个城市的销售额和环比，已按销售额排序。",
                )
            ],
        }


class VisualizationSpecTool(Tool):
    """将数仓统计结果映射为前端可消费的图表配置。"""

    spec = ToolSpec(
        id="viz.sales_chart",
        name="可视化结果生成器",
        kind=StepKind.VISUALIZATION,
        description="把表格结果转换成前端可渲染的柱状图/折线图配置。",
    )

    @log_call("inputs")
    def invoke(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """从 warehouse_result 复用城市、销售额和环比数据，返回柱状图及折线图配置和证据。"""
        result = inputs["warehouse_result"]
        rows = result["rows"]
        return {
            "charts": [
                {
                    "type": "bar",
                    "title": "华东区域本月销售额按城市排序",
                    "x": [row["city"] for row in rows],
                    "y": [row["sales_amount"] for row in rows],
                    "unit": result.get("currency", "CNY"),
                },
                {
                    "type": "line",
                    "title": "城市销售额环比",
                    "x": [row["city"] for row in rows],
                    "y": [row["mom_rate"] for row in rows],
                    "unit": "%",
                },
            ],
            "evidence": [
                Evidence(source="viz.sales_chart", detail="已生成柱状图和环比折线图配置。")
            ],
        }


class BusinessInsightTool(Tool):
    """依据销售额和环比排序生成确定性洞察，供 Composer 组织文字结论。"""

    spec = ToolSpec(
        id="insight.sales_mom",
        name="原因洞察 / 文字结论",
        kind=StepKind.MODEL,
        description="基于结构化结果生成确定性洞察要点。",
    )

    @log_call("inputs")
    def invoke(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """从非空且按销售额降序的 warehouse_result.rows 提取最高销售额和环比极值城市。"""
        rows = inputs["warehouse_result"]["rows"]
        top = rows[0]
        weakest = min(rows, key=lambda item: item["mom_rate"])
        fastest = max(rows, key=lambda item: item["mom_rate"])
        return {
            "insights": [
                f"{top['city']}销售额最高，是本月华东区域主要贡献城市。",
                f"{fastest['city']}环比增速最高，建议复盘其渠道或客户结构变化。",
                f"{weakest['city']}环比最低，需要进一步下钻到客户、品类或订单状态。",
            ],
            "evidence": [
                Evidence(
                    source="insight.sales_mom",
                    detail="洞察基于销售额排序和环比排序确定性生成，未引入外部事实。",
                )
            ],
        }
