"""验证日志格式、凭证脱敏及多线程执行链中的会话关联。"""

from __future__ import annotations

import io
import json
import logging
import re
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier
from unittest.mock import patch

from zhuge_agent.llm import FixtureLLMClient, LLMError, OpenAICompatibleLLMClient
from zhuge_agent.runtime import ZhugeAgentRuntime
from zhuge_agent.schema import ParallelStructuredAnswer, StructuredAnswer
from zhuge_agent.utils.config import ConfigError, load_logging_config
from zhuge_agent.utils.logging_utils import (
    AgentLogFormatter, configure_logging, current_session_id, log_call, log_context, log_value, register_secret,
)
from zhuge_agent.utils.prompt_templates import AgentPromptTemplate, LazyPrompt


class LoggingTest(unittest.TestCase):
    """所有用例使用内存输出和离线客户端，不创建项目日志文件或调用真实 API。"""

    def setUp(self) -> None:
        """临时配置内存日志，结束后恢复宿主 Handler 和日志级别。"""
        root = logging.getLogger()
        original_handlers = root.handlers[:]
        original_level = root.level
        self.addCleanup(self._restore_logging, original_handlers, original_level)
        self.stream = io.StringIO()
        configure_logging("DEBUG", stream=self.stream)

    def _restore_logging(self, handlers: list[logging.Handler], level: int) -> None:
        """清理测试 Handler 并恢复原有日志设置，避免用例之间的输出目标干扰。"""
        root = logging.getLogger()
        for handler in root.handlers[:]:
            if handler not in handlers:
                handler.close()
        root.handlers = handlers
        root.setLevel(level)

    def test_format_has_identity_milliseconds_module_and_line(self) -> None:
        """验证格式精确包含身份、毫秒时间、级别及 module@lineno。"""
        record = logging.LogRecord("demo", logging.INFO, "/tmp/runtime.py", 42, "测试日志", (), None)
        record.created = 1700000000.123
        record.msecs = 123
        with log_context(agent_id="agent-a", session_id="session-a"):
            text = AgentLogFormatter().format(record)
        self.assertRegex(text, r"^agent-a session-a \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.123 INFO runtime@42 - 测试日志$")
        record = logging.LogRecord("demo", logging.INFO, "/tmp/runtime.py", 42, "启动日志", (), None)
        self.assertTrue(AgentLogFormatter().format(record).startswith("- - "))

    def test_configuration_is_idempotent_and_respects_level(self) -> None:
        """验证重复配置只保留一个项目控制台 Handler，且 WARNING 不输出 INFO。"""
        configure_logging("WARNING", stream=self.stream)
        configure_logging("WARNING", stream=self.stream)
        logging.getLogger("zhuge_agent.test").info("不可见")
        logging.getLogger("zhuge_agent.test").warning("仅一次")
        self.assertNotIn("不可见", self.stream.getvalue())
        self.assertEqual(self.stream.getvalue().count("仅一次"), 1)
        handlers = [handler for handler in logging.getLogger().handlers if (handler.name or "").startswith("zhuge_")]
        self.assertEqual(len(handlers), 1)

    def test_function_logging_does_not_render_lazy_prompt(self) -> None:
        """验证 DEBUG 输入输出记录保留延迟模板，且日志使用被记录函数的源文件和行号。"""
        runtime = ZhugeAgentRuntime(llm_client=FixtureLLMClient())
        calls = []
        prompt = LazyPrompt(lambda: calls.append("rendered") or "提示词")
        runtime.llm.chat_json(system=prompt, user="{}", schema_hint={"answer_type": "string"})
        template = AgentPromptTemplate(runtime.agents.get("dcs_query_agent").prompt_template_path)
        with patch.object(template, "render", wraps=template.render) as render:
            lazy = template.lazy_render("skill_router_system", "备用", agent=runtime.agents.get("dcs_query_agent"))
            log_value(lazy)
            render.assert_not_called()
        self.assertEqual(calls, [])
        lines = self.stream.getvalue()
        self.assertIn("<LazyPrompt>", lines)
        self.assertRegex(lines, r"llm@\d+ - 函数输入 function=FixtureLLMClient.chat_json")
        self.assertRegex(lines, r"llm@\d+ - 函数输出 function=FixtureLLMClient.chat_json")

    def test_credentials_are_redacted_in_input_output_and_errors(self) -> None:
        """验证凭证字段、登记密钥和异常中的 Authorization 不会出现在日志。"""
        secret = "test-only-logging-secret"
        register_secret(secret)

        @log_call("payload")
        def echo(payload: dict) -> dict:
            """返回测试参数以检查输入和输出的递归脱敏。"""
            return payload

        payload = {"api_key": secret, "nested": {"Authorization": "Bearer unrelated-secret", "password": "hidden"},
                   "text": f"upstream rejected {secret}"}
        self.assertEqual(echo(payload), payload)
        client = OpenAICompatibleLLMClient(api_key=secret, base_url="https://example.invalid/v1")
        with patch("zhuge_agent.llm.urllib.request.urlopen", side_effect=LLMError(f"Authorization: Bearer {secret}")):
            with self.assertRaises(LLMError):
                client.chat_json(system="system", user="user", schema_hint={"answer": "string"})
        text = self.stream.getvalue()
        self.assertNotIn(secret, text)
        self.assertNotIn("unrelated-secret", text)
        self.assertNotIn("hidden", text)
        self.assertIn("***", text)
        self.assertIn("函数异常", text)

    def test_two_parallel_requests_keep_sessions_and_agents_isolated(self) -> None:
        """让两次请求的主路由同时执行，验证各自子线程日志保留独立会话和智能体身份。"""
        barrier = Barrier(2)

        class ConcurrentFixture(FixtureLLMClient):
            """同步主路由以强制不同会话交错执行。"""

            def chat_json(self, **kwargs) -> dict:
                """在主路由处等待另一请求，其他任务按离线客户端执行。"""
                if "agent_ids" in kwargs["schema_hint"]:
                    barrier.wait(timeout=10)
                return super().chat_json(**kwargs)

        case_path = Path(__file__).parent / "cases" / "parallel_business_dcs_case.json"
        case = json.loads(case_path.read_text(encoding="utf-8"))
        runtime = ZhugeAgentRuntime(llm_client=ConcurrentFixture(), case_data=case)
        self.stream.seek(0)
        self.stream.truncate()
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(runtime.answer_parallel, case["request"], session_id=session)
                       for session in ("session-one", "session-two")]
            results = [future.result(timeout=15) for future in futures]
        self.assertTrue(all(result.verification["passed"] for result in results))
        for session in ("session-one", "session-two"):
            lines = [line for line in self.stream.getvalue().splitlines() if line.split()[1] == session]
            self.assertTrue(any(line.startswith(f"MasterAgent {session} ") for line in lines))
            expected_modules = {
                "business_metric_agent": "business@",
                "dcs_query_agent": "industrial@",
            }
            for agent_id, module_marker in expected_modules.items():
                self.assertTrue(
                    any(line.startswith(f"{agent_id} {session} ") and module_marker in line for line in lines)
                )
        self.assertEqual(current_session_id(), "-")

    def test_streaming_threads_preserve_explicit_and_parent_sessions(self) -> None:
        """验证单任务与并行流式入口继承会话上下文，结束后恢复调用线程身份。"""
        case = json.loads((Path(__file__).parent / "cases" / "parallel_business_dcs_case.json").read_text("utf-8"))
        runtime = ZhugeAgentRuntime(llm_client=FixtureLLMClient(), case_data=case)
        with log_context(session_id="parent-session"):
            items = list(runtime.iter_answer("现在 2# 反应釜温度和压力是多少？"))
            self.assertEqual(current_session_id(), "parent-session")
        parallel = list(runtime.iter_answer_parallel(case["request"], session_id="stream-session"))
        self.assertIsInstance(items[-1], StructuredAnswer)
        self.assertIsInstance(parallel[-1], ParallelStructuredAnswer)
        text = self.stream.getvalue()
        self.assertIn("dcs_query_agent parent-session ", text)
        self.assertIn("business_metric_agent stream-session ", text)
        self.assertEqual(current_session_id(), "-")

    def test_exception_restores_context_and_next_request_gets_new_session(self) -> None:
        """验证异常日志关联失败会话，之后新请求自动生成会话且无身份残留。"""

        class FailingFixture(FixtureLLMClient):
            """模拟路由请求失败。"""

            def chat_json(self, **kwargs) -> dict:
                """抛出固定异常，检查装饰器仍保持原异常传播。"""
                raise LLMError("测试失败")

        failed_runtime = ZhugeAgentRuntime(llm_client=FailingFixture())
        with self.assertRaises(LLMError):
            failed_runtime.answer("测试请求", session_id="failed-session")
        self.assertEqual(current_session_id(), "-")
        runtime = ZhugeAgentRuntime(llm_client=FixtureLLMClient())
        runtime.answer("现在 2# 反应釜温度是多少？")
        text = self.stream.getvalue()
        self.assertRegex(text, r"MasterAgent failed-session .*ERROR runtime@\d+ - 函数异常")
        self.assertRegex(text, r"MasterAgent [0-9a-f]{32} .*函数输入 function=ZhugeAgentRuntime.answer")

    def test_logging_config_resolves_file_and_utf8_output(self) -> None:
        """验证日志文件相对配置解析、UTF-8 写入及无效日志级别报错。"""
        with TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.ini"
            config_path.write_text("[logging]\nlevel = INFO\nfile = logs/demo.log\n", encoding="utf-8")
            config = load_logging_config(config_path)
            self.assertEqual(config.file, Path(directory) / "logs" / "demo.log")
            configure_logging(config.level, stream=self.stream, log_file=config.file)
            with log_context(agent_id="demo", session_id="file-session"):
                logging.getLogger("zhuge_agent.test").info("中文日志")
            for handler in logging.getLogger().handlers:
                handler.flush()
            self.assertIn("demo file-session", config.file.read_text("utf-8"))
            self.assertIn("中文日志", config.file.read_text("utf-8"))
            configure_logging(stream=self.stream)
            config_path.write_text("[logging]\nlevel = INVALID\n", encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_logging_config(config_path)

    def test_large_outputs_are_summarized_without_mutation(self) -> None:
        """验证长列表记录条目数量，保留关键数值，且原结果不被日志裁剪修改。"""
        rows = [{"value": index} for index in range(100)]
        result = log_value({"rows": rows, "total": 4950})
        self.assertEqual(result["rows"][-1], {"_total_items": 100})
        self.assertEqual(result["total"], 4950)
        self.assertEqual(len(rows), 100)
        self.assertIsNotNone(re.search("4950", json.dumps(result)))


if __name__ == "__main__":
    unittest.main()
