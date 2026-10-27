"""Offline startup gates, safe diagnostics and fixed-entrypoint execution."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from llm_agent import ConfigurationError, Settings
import run_chat


class LauncherChecks(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        for name in run_chat.SOURCE_FILES:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(run_chat.ROOT / name, path)
        return root

    def test_missing_source_stops_before_packages_or_key_access(self):
        with tempfile.TemporaryDirectory() as directory, patch("run_chat.dependency_check") as dependencies:
            report = run_chat.check_setup(Path(directory))
            self.assertFalse(report["ready"])
            self.assertIn("source_files_missing", [c["code"] for c in report["checks"]])
            dependencies.assert_not_called()

    def test_old_python_stops_before_loading_provider_packages(self):
        with patch("run_chat.sys.version_info", (3, 9)), patch("run_chat.dependency_check") as dependencies:
            report = run_chat.check_setup()
            self.assertFalse(report["ready"])
            self.assertEqual(report["checks"][0]["code"], "python_too_old")
            dependencies.assert_not_called()

    def test_pins_must_be_exact_and_missing_or_mismatched_packages_fail_safely(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock = root / "requirements.lock.txt"
            for text in ("", "pydantic>=2", "https://unrelated.invalid/code.py"):
                lock.write_text(text)
                self.assertEqual(run_chat.dependency_check(root).code, "invalid_dependency_lock")
            lock.write_text("pydantic==2.14.1\n")
            with patch("run_chat.metadata.version", return_value="different"):
                self.assertEqual(run_chat.dependency_check(root).code, "dependency_mismatch")
            with patch("run_chat.metadata.version", side_effect=run_chat.metadata.PackageNotFoundError("never-show-secret")):
                result = run_chat.dependency_check(root)
                self.assertEqual(result.code, "dependency_missing")
                self.assertNotIn("never-show-secret", result.message)

    def test_missing_index_is_a_notice_without_client_or_embedding_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture(directory)
            with patch("llm_agent.Settings.from_local_env", return_value=Settings(api_key="fixture")), \
                 patch("semantic_search.KnowledgeSearch") as search, patch("openai.OpenAI") as client:
                checks = run_chat.resource_checks(root)
            self.assertEqual(checks[-1].code, "index_not_built")
            self.assertFalse(any(c.status == "failed" for c in checks))
            search.assert_not_called()
            client.assert_not_called()

    def test_invalid_key_message_is_safe_and_never_prints_exception_text(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture(directory)
            with patch("llm_agent.Settings.from_local_env", side_effect=ConfigurationError("secret-key-never-print")):
                checks = run_chat.resource_checks(root)
            self.assertEqual(checks[0].code, "key_setup_needed")
            self.assertNotIn("secret-key-never-print", str(checks))

    def test_bad_faq_and_empty_prompt_fail_without_echoing_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture(directory)
            with patch("llm_agent.Settings.from_local_env", return_value=Settings(api_key="fixture")):
                (root / "octakidz_ai_tutor_faq.json").write_text('{"private":"never-echo-content"}')
                checks = run_chat.resource_checks(root)
                self.assertEqual(checks[-1].code, "support_files_invalid")
                self.assertNotIn("never-echo-content", str(checks))
                shutil.copyfile(run_chat.ROOT / "octakidz_ai_tutor_faq.json", root / "octakidz_ai_tutor_faq.json")
                (root / "prompts/common.md").write_text(" \n")
                self.assertEqual(run_chat.resource_checks(root)[-1].code, "support_files_invalid")

    def test_corrupt_index_uses_existing_validator_without_provider_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture(directory)
            cache = root / ".cache/faq_embeddings.json"
            cache.parent.mkdir()
            cache.write_text("not-json-never-echo")
            with patch("llm_agent.Settings.from_local_env", return_value=Settings(api_key="fixture")), \
                 patch("openai.OpenAI") as client:
                checks = run_chat.resource_checks(root)
            self.assertEqual(checks[-1].code, "index_needs_rebuild")
            self.assertNotIn("not-json-never-echo", str(checks))
            client.assert_not_called()

    def test_environment_selection_uses_prefix_without_symlink_loop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / ".venv"
            candidate = target / ("Scripts/python.exe" if run_chat.os.name == "nt" else "bin/python")
            candidate.parent.mkdir(parents=True)
            candidate.touch()
            with patch("run_chat.sys.prefix", str(root)):
                self.assertEqual(run_chat.environment_python(root), candidate)
            with patch("run_chat.sys.prefix", str(target)):
                self.assertIsNone(run_chat.environment_python(root))

    def test_check_mode_reports_but_does_not_launch_or_write_logs(self):
        report = {"ready": True, "api_requests": 0, "checks": []}
        with patch("run_chat.environment_python", return_value=None), \
             patch("run_chat.check_setup", return_value=report), \
             patch("run_chat.os.execv") as execute, patch("run_chat.write_startup_event") as log, \
             redirect_stdout(StringIO()) as output:
            self.assertEqual(run_chat.main(["--check", "--json"]), 0)
        self.assertEqual(json.loads(output.getvalue()), report)
        execute.assert_not_called()
        log.assert_not_called()

    def test_failed_setup_never_starts_chat(self):
        report = {"ready": False, "checks": [{"code": "key_setup_needed", "status": "failed", "message": "Check local key."}]}
        with patch("run_chat.environment_python", return_value=None), \
             patch("run_chat.check_setup", return_value=report), patch("run_chat.os.execv") as execute, \
             patch("run_chat.write_startup_event") as log, redirect_stdout(StringIO()):
            self.assertEqual(run_chat.main([]), 1)
        execute.assert_not_called()
        log.assert_called_once_with(report)

    def test_healthy_launch_executes_only_fixed_chat_with_debug_and_survives_log_failure(self):
        class ReplacedProcess(Exception):
            pass
        report = {"ready": True, "checks": []}
        with patch("run_chat.environment_python", return_value=None), \
             patch("run_chat.check_setup", return_value=report), \
             patch("run_chat.write_startup_event", side_effect=OSError("private-raw-error")), \
             patch("run_chat.os.execv", side_effect=ReplacedProcess) as execute, \
             redirect_stdout(StringIO()), redirect_stderr(StringIO()) as errors:
            with self.assertRaises(ReplacedProcess):
                run_chat.main(["--debug"])
        self.assertEqual(execute.call_args.args, (run_chat.sys.executable,
                         [run_chat.sys.executable, str(run_chat.ROOT / "conversation_agent.py"), "--debug"]))
        self.assertNotIn("private-raw-error", errors.getvalue())
        self.assertIn("Could not save", errors.getvalue())

    def test_startup_logs_only_codes_status_latency_and_version(self):
        report = {"ready": False, "latency_ms": 12.3, "api_key": "never-log-key", "question": "never-log-question",
                  "checks": [{"code": "key_setup_needed", "status": "failed", "message": "never-log-message"}]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_chat.write_startup_event(report, root)
            text = (root / "logs/startup_events.jsonl").read_text()
        for private in ("never-log-key", "never-log-question", "never-log-message"):
            self.assertNotIn(private, text)
        self.assertEqual(json.loads(text)["latency_ms"], 12.3)


if __name__ == "__main__":
    unittest.main()
