"""验证文件配置、模式选择、错误边界及真实客户端的请求参数。"""

from __future__ import annotations

import io
import json
import subprocess
import sys
import types
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from zhuge_agent.llm import FixtureLLMClient, OpenAICompatibleLLMClient, llm_client_from_config
from zhuge_agent.runtime import ZhugeAgentRuntime
from zhuge_agent.utils.config import ConfigError, DEFAULT_CONFIG_PATH, load_llm_config
from zhuge_agent.utils.prompt_templates import LazyPrompt
from zhuge_agent_demo import main


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_CONFIG = PROJECT_ROOT / "tests" / "cases" / "fixture_config.ini"
REAL_CONFIG = """[llm]
provider = openai
api_key = test-only-key%with-percent
base_url = https://example.invalid/v1/
model = test-model
timeout_seconds = 17
"""
SDK_CONFIG = REAL_CONFIG + "transport = openai_sdk\nverify_ssl = false\n"
EXTRA_BODY_CONFIG = REAL_CONFIG + (
    'extra_body = {"chat_template_kwargs": {"enable_thinking": false, "foo": "bar"}}\n'
)


class LLMConfigTest(unittest.TestCase):
    """使用临时配置和 HTTP 模拟验证接入，不访问真实 LLM 服务。"""

    def setUp(self) -> None:
        """为每个用例创建隔离的配置目录，并注册自动清理。"""
        # 生命周期由 unittest.addCleanup 管理，覆盖测试失败和异常退出。
        directory = TemporaryDirectory()  # pylint: disable=consider-using-with
        self.addCleanup(directory.cleanup)
        self.config_path = Path(directory.name) / "config.ini"

    def _write_config(self, content: str) -> Path:
        """在临时目录写入测试配置，返回文件路径。"""
        self.config_path.write_text(content, encoding="utf-8")
        return self.config_path

    def test_real_client_uses_file_settings(self) -> None:
        """验证真实客户端从文件取得密钥、地址、模型和超时，并忽略旧环境变量。"""
        path = self._write_config(REAL_CONFIG)
        with patch.dict("os.environ", {"ZHUGE_LLM_API_KEY": "unused", "ZHUGE_LLM_MODEL": "unused"}):
            runtime = ZhugeAgentRuntime(config_path=path)
        client = runtime.llm
        self.assertIsInstance(client, OpenAICompatibleLLMClient)
        self.assertEqual(client.api_key, "test-only-key%with-percent")
        self.assertEqual(client.base_url, "https://example.invalid/v1")
        self.assertEqual(client.model, "test-model")
        self.assertEqual(client.timeout_seconds, 17)
        self.assertEqual(client.transport, "urllib")
        self.assertTrue(client.verify_ssl)
        self.assertEqual(client.extra_body, {})
        self.assertNotIn(client.api_key, repr(load_llm_config(path)))

    def test_sdk_transport_uses_file_settings(self) -> None:
        """验证 OpenAI SDK transport 和 SSL 校验开关可以通过配置文件显式控制。"""
        client = llm_client_from_config(self._write_config(SDK_CONFIG))
        self.assertIsInstance(client, OpenAICompatibleLLMClient)
        self.assertEqual(client.transport, "openai_sdk")
        self.assertFalse(client.verify_ssl)

    def test_extra_body_is_loaded_for_model_specific_options(self) -> None:
        """验证 Qwen/vLLM 等模型扩展请求体可以通过配置文件声明。"""
        config = load_llm_config(self._write_config(EXTRA_BODY_CONFIG))

        self.assertEqual(
            config.extra_body,
            {"chat_template_kwargs": {"enable_thinking": False, "foo": "bar"}},
        )

    def test_fixture_mode_is_explicit_and_offline(self) -> None:
        """验证 fixture 配置无需密钥，仍能离线运行原有查数链路。"""
        runtime = ZhugeAgentRuntime(config_path=FIXTURE_CONFIG)
        self.assertIsInstance(runtime.llm, FixtureLLMClient)
        result = runtime.answer("现在 2# 反应釜温度和压力是多少？")
        self.assertEqual(result.skill_id, "query_tag_value")
        self.assertTrue(result.verification["passed"])

    def test_injected_client_skips_config_loading(self) -> None:
        """验证显式注入客户端时忽略配置路径，测试无需真实密钥或本地文件。"""
        client = FixtureLLMClient()
        runtime = ZhugeAgentRuntime(llm_client=client, config_path=self.config_path)
        self.assertIs(runtime.llm, client)

    def test_default_path_is_independent_of_working_directory(self) -> None:
        """验证默认配置固定指向项目根目录，而显式相对路径仍由调用方决定。"""
        self.assertEqual(DEFAULT_CONFIG_PATH, PROJECT_ROOT / "config.ini")
        self.assertTrue(DEFAULT_CONFIG_PATH.is_absolute())

    def test_missing_file_does_not_fall_back(self) -> None:
        """验证缺失配置明确报错，不自动切换为离线演示。"""
        with self.assertRaises(ConfigError):
            llm_client_from_config(self.config_path)

    def test_missing_key_does_not_use_environment(self) -> None:
        """验证真实模式缺少文件密钥时，即使环境变量有值也报错。"""
        path = self._write_config("[llm]\nprovider = openai\n")
        with patch.dict("os.environ", {"ZHUGE_LLM_API_KEY": "unused", "OPENAI_API_KEY": "unused"}):
            with self.assertRaisesRegex(ConfigError, "api_key"):
                llm_client_from_config(path)

    def test_invalid_fields_are_rejected(self) -> None:
        """验证模式、段落、服务地址、模型和超时的无效配置被拒绝。"""
        invalid_configs = [
            "[other]\nprovider = fixture\n",
            "[llm]\nprovider = unknown\n",
            "[llm]\napi_key = test-only-key\n",
            REAL_CONFIG.replace("https://example.invalid/v1/", "file:///tmp/api"),
            REAL_CONFIG.replace("https://example.invalid/v1/", "https://example.invalid/v1?key=test"),
            REAL_CONFIG.replace("model = test-model", "model ="),
            REAL_CONFIG.replace("timeout_seconds = 17", "timeout_seconds = 0"),
            REAL_CONFIG.replace("timeout_seconds = 17", "timeout_seconds = invalid"),
            REAL_CONFIG + "transport = invalid\n",
            REAL_CONFIG + "verify_ssl = maybe\n",
            REAL_CONFIG + "extra_body = invalid\n",
            REAL_CONFIG + "extra_body = []\n",
        ]
        for content in invalid_configs:
            with self.subTest(content=content):
                with self.assertRaises(ConfigError):
                    load_llm_config(self._write_config(content))

    def test_invalid_ini_does_not_expose_contents(self) -> None:
        """验证 INI 解析失败时错误信息和 CLI 输出不包含文件中的密钥。"""
        secret = "test-only-secret-that-must-not-be-printed"
        path = self._write_config(secret)
        with self.assertRaises(ConfigError) as caught:
            load_llm_config(path)
        self.assertNotIn(secret, str(caught.exception))
        stderr = io.StringIO()
        with patch.object(sys, "argv", ["zhuge_agent_demo.py", "--config", str(path)]):
            with redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as exit_error:
                    main()
        self.assertEqual(exit_error.exception.code, 2)
        self.assertNotIn(secret, stderr.getvalue())

    def test_configured_client_sends_request_and_consumes_lazy_prompt(self) -> None:
        """验证文件设置传入真实 HTTP 请求，且系统提示词在发送时才渲染。"""
        client = llm_client_from_config(self._write_config(REAL_CONFIG))
        calls = []
        prompt = LazyPrompt(lambda: calls.append("rendered") or "中文系统提示词")
        with patch("zhuge_agent.llm.urllib.request.urlopen") as urlopen:
            response = urlopen.return_value.__enter__.return_value
            response.read.return_value = b'{"choices":[{"message":{"content":"{\\"answer\\":\\"ok\\"}"}}]}'
            result = client.chat_json(system=prompt, user="用户问题", schema_hint={"answer": "string"})
        request = urlopen.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(calls, ["rendered"])
        self.assertEqual(result, {"answer": "ok"})
        self.assertEqual(request.full_url, "https://example.invalid/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-only-key%with-percent")
        self.assertEqual(payload["model"], "test-model")
        self.assertNotIn("chat_template_kwargs", payload)
        self.assertEqual(payload["messages"][0]["content"], "中文系统提示词")
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 17)

    def test_configured_client_merges_extra_body_into_urllib_payload(self) -> None:
        """验证 urllib transport 将配置和单次调用扩展字段合并到请求体。"""
        client = llm_client_from_config(self._write_config(EXTRA_BODY_CONFIG))
        with patch("zhuge_agent.llm.urllib.request.urlopen") as urlopen:
            response = urlopen.return_value.__enter__.return_value
            response.read.return_value = b'{"choices":[{"message":{"content":"{\\"answer\\":\\"ok\\"}"}}]}'
            result = client.chat_json(
                system="系统",
                user="用户",
                schema_hint={"answer": "string"},
                extra_body={"chat_template_kwargs": {"foo": "override", "bar": "baz"}},
            )

        payload = json.loads(urlopen.call_args.args[0].data)
        self.assertEqual(result, {"answer": "ok"})
        self.assertEqual(
            payload["chat_template_kwargs"],
            {"enable_thinking": False, "foo": "override", "bar": "baz"},
        )

    def test_configured_sdk_client_sends_request_with_httpx_options(self) -> None:
        """验证 SDK transport 使用 httpx.Client(verify=配置值, timeout=配置值)，且不访问网络。"""
        client = llm_client_from_config(self._write_config(SDK_CONFIG + 'extra_body = {"top_k": 20}\n'))
        calls = []

        class FakeHTTPClient:
            """模拟 httpx.Client 上下文管理器，记录 SSL 和超时参数。"""

            def __init__(self, *, verify: bool, timeout: int) -> None:
                """保存构造参数供断言使用。"""
                calls.append(("httpx", verify, timeout))

            def __enter__(self) -> "FakeHTTPClient":
                """返回自身以模拟上下文进入。"""
                return self

            def __exit__(self, *args: object) -> None:
                """模拟上下文退出，不做额外处理。"""

        class FakeCompletions:
            """模拟 chat.completions.create 接口，记录请求并返回 JSON 内容。"""

            @staticmethod
            def create(**payload: object) -> object:
                """返回具备 choices[0].message.content 结构的响应对象。"""
                calls.append(("create", payload))
                message = types.SimpleNamespace(content='{"answer":"ok"}')
                choice = types.SimpleNamespace(message=message)
                return types.SimpleNamespace(choices=[choice])

        class FakeOpenAI:
            """模拟 OpenAI SDK 客户端，记录 base_url、api_key 和 http_client。"""

            def __init__(self, *, base_url: str, api_key: str, http_client: FakeHTTPClient) -> None:
                """保存 SDK 客户端构造参数供断言使用。"""
                calls.append(("openai", base_url, api_key, http_client))
                self.chat = types.SimpleNamespace(completions=FakeCompletions())

        fake_httpx = types.SimpleNamespace(Client=FakeHTTPClient)
        fake_openai = types.SimpleNamespace(OpenAI=FakeOpenAI)
        with patch("zhuge_agent.llm.importlib.import_module", side_effect=[fake_httpx, fake_openai]):
            result = client.chat_json(system="系统", user="用户", schema_hint={"answer": "string"})

        self.assertEqual(result, {"answer": "ok"})
        self.assertEqual(calls[0], ("httpx", False, 17))
        self.assertEqual(calls[1][0], "openai")
        self.assertEqual(calls[1][1], "https://example.invalid/v1")
        self.assertEqual(calls[1][2], "test-only-key%with-percent")
        payload = calls[2][1]
        self.assertEqual(payload["model"], "test-model")
        self.assertEqual(payload["messages"][0]["content"], "系统")
        self.assertEqual(payload["extra_body"], {"top_k": 20})

    def test_cli_accepts_config_from_another_working_directory(self) -> None:
        """验证 CLI 支持从其他工作目录用 --config 运行流式案例，且显示离线模式。"""
        process = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "zhuge_agent_demo.py"),
                "--config", str(FIXTURE_CONFIG),
                "--stream", str(PROJECT_ROOT / "tests" / "cases" / "query_tag_value_case.json"),
            ],
            cwd=self.config_path.parent,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=15,
            check=False,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn("LLM 模式：离线 fixture", process.stdout)
        self.assertIn("流式进度", process.stdout)
        self.assertIn("130.9", process.stdout)


if __name__ == "__main__":
    unittest.main()
