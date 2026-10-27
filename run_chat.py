"""Phase 8: check the local setup, then start the existing terminal chatbot."""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata
import json
import os
from pathlib import Path
import re
import sys
from time import perf_counter


VERSION = "phase8_launcher_v1"
ROOT = Path(__file__).resolve().parent
SOURCE_FILES = (
    "conversation_agent.py", "conversation_memory.py", "response_feedback.py", "grounded_checker.py",
    "tool_agent.py", "support_tools.py", "rag_agent.py", "semantic_search.py", "llm_agent.py",
    "support_safety.py", "baseline_agent.py", "octakidz_ai_tutor_faq.json", "requirements.lock.txt",
    "prompts/common.md", "prompts/v3_parent.md", "prompts/tool_router.md", "prompts/retrieval_check.md",
)
INSTALL_HELP = "Create .venv with python3 -m venv .venv, then run .venv/bin/python -m pip install -r requirements.lock.txt."
REBUILD_HELP = ('Rebuild with .venv/bin/python semantic_search.py --rebuild --question "What is Octakidz?" '
                '(uses OpenAI).')


@dataclass(frozen=True)
class Check:
    code: str
    status: str
    message: str


def environment_python(root=ROOT):
    """Use the project environment when available; compare prefixes, not symlinks."""
    target = root / ".venv"
    candidate = target / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if candidate.is_file() and Path(sys.prefix).resolve() != target.resolve():
        return candidate
    return None


def dependency_check(root):
    """Read exact pins without installing packages or invoking a shell."""
    try:
        pins = []
        for line in (root / "requirements.lock.txt").read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([A-Za-z0-9_.+!-]+)", line.strip())
            if not match:
                return Check("invalid_dependency_lock", "failed", "Restore requirements.lock.txt from the project sources.")
            pins.append(match.groups())
        if not pins:
            return Check("invalid_dependency_lock", "failed", "Restore the non-empty pinned dependency list.")
        for name, expected in pins:
            if metadata.version(name) != expected:
                return Check("dependency_mismatch", "failed", "The installed packages differ from the tested versions. " + INSTALL_HELP)
    except metadata.PackageNotFoundError:
        return Check("dependency_missing", "failed", "A required Python package is missing. " + INSTALL_HELP)
    except (OSError, ValueError):
        return Check("dependency_check_failed", "failed", "Could not read the package setup. " + INSTALL_HELP)
    return Check("dependencies_ready", "passed", "Installed packages match requirements.lock.txt.")


def resource_checks(root):
    """Validate local configuration/data with the existing code, without a client."""
    try:
        from llm_agent import ConfigurationError, Settings
        from baseline_agent import load_faq
        from semantic_search import KnowledgeSearch, SearchError
    except Exception:
        return [Check("dependency_import_failed", "failed", "Python packages could not be loaded. " + INSTALL_HELP)]
    checks = []
    try:
        # Settings already implements environment-over-.env precedence. Never
        # return the key, settings object, model value or exception text.
        Settings.from_local_env()
        checks.append(Check("key_configured", "passed", "A local API key is configured; it has not been checked online."))
    except (ConfigurationError, OSError, ValueError):
        checks.append(Check("key_setup_needed", "failed",
                            "Add or correct OPENAI_API_KEY in your local .env file or environment. Never paste it into chat."))
    try:
        load_faq(root / "octakidz_ai_tutor_faq.json")
        for name in SOURCE_FILES:
            if name.startswith("prompts/") and not (root / name).read_text(encoding="utf-8").strip():
                raise ValueError("empty_prompt")
        checks.append(Check("support_files_valid", "passed", "The curated support data and required prompts are readable."))
    except (OSError, ValueError, TypeError, KeyError):
        checks.append(Check("support_files_invalid", "failed", "Restore the curated FAQ and prompt files; no file contents were printed."))
        return checks
    cache = root / ".cache/faq_embeddings.json"
    if not cache.exists():
        checks.append(Check("index_not_built", "notice",
                            "The first knowledge search will build the public FAQ index using OpenAI; this check does not build it."))
        return checks

    class NoNetworkEmbedder:
        def embed(self, texts):
            raise AssertionError("The setup check must never create embeddings.")

    try:
        KnowledgeSearch(NoNetworkEmbedder(), cache_path=cache).prepare()
        checks.append(Check("index_valid", "passed", "The saved FAQ index matches the current knowledge and vector format."))
    except (SearchError, OSError, ValueError, TypeError, AssertionError):
        checks.append(Check("index_needs_rebuild", "failed", "The saved FAQ index is invalid or outdated. " + REBUILD_HELP))
    return checks


def check_setup(root=ROOT):
    started = perf_counter()
    checks = []
    if sys.version_info < (3, 10):
        checks.append(Check("python_too_old", "failed", "Use Python 3.10 or newer, then recreate .venv."))
    else:
        checks.append(Check("python_ready", "passed", "Python meets the minimum version requirement."))
    if not all((root / name).is_file() for name in SOURCE_FILES):
        checks.append(Check("source_files_missing", "failed", "One or more required capstone files are missing. Restore the project sources."))
    else:
        checks.append(Check("source_files_present", "passed", "Required capstone source files are present."))
    if all(c.status != "failed" for c in checks):
        checks.append(dependency_check(root))
        if checks[-1].status != "failed":
            checks.extend(resource_checks(root))
    return {"launcher_version": VERSION, "ready": all(c.status != "failed" for c in checks),
            "api_requests": 0, "latency_ms": (perf_counter() - started) * 1000,
            "checks": [asdict(c) for c in checks]}


def write_startup_event(report, root=ROOT):
    event = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "launcher_version": VERSION,
             "ready": report["ready"], "latency_ms": report["latency_ms"],
             "checks": [{"code": c["code"], "status": c["status"]} for c in report["checks"]]}
    path = root / "logs/startup_events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check local setup without API calls or starting chat.")
    parser.add_argument("--json", action="store_true", help="Print the --check report as safe JSON.")
    parser.add_argument("--debug", action="store_true", help="Show existing safe diagnostics during chat.")
    args = parser.parse_args(argv)
    if args.json and not args.check:
        parser.error("--json requires --check")
    forwarded = list(sys.argv[1:] if argv is None else argv)
    candidate = environment_python()
    if candidate is not None:
        try:
            os.execv(str(candidate), [str(candidate), str(Path(__file__).resolve()), *forwarded])
        except OSError:
            print("Could not start the project Python environment. " + INSTALL_HELP, file=sys.stderr)
            return 1
    report = check_setup()
    if args.json:
        print(json.dumps(report, indent=2))
    elif args.check or not report["ready"]:
        for item in report["checks"]:
            print(f"[{item['status']}] {item['message']}")
        print("Ready for local chat." if report["ready"] else "Setup needs attention before chat can start.")
        print("This check made no API requests; key validity, quota and internet access are not verified.")
    if args.check:
        return 0 if report["ready"] else 1
    try:
        write_startup_event(report)
    except OSError:
        print("Could not save the setup log. Chat replies can still work.", file=sys.stderr)
    if not report["ready"]:
        return 1
    print("Local setup checked. Starting Octakidz chat…", flush=True)
    try:
        os.execv(sys.executable, [sys.executable, str(ROOT / "conversation_agent.py"),
                                 *(["--debug"] if args.debug else [])])
    except OSError:
        print("Could not start the chatbot. Check the project files and Python environment.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
