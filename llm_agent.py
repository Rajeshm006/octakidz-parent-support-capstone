"""Phase 3: one OpenAI call through LangChain, with the curated product facts."""

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Literal

from dotenv import dotenv_values
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langsmith import tracing_context
from openai import APIConnectionError, APIError, APITimeoutError, AuthenticationError, NotFoundError, RateLimitError
from pydantic import BaseModel, ConfigDict, ValidationError

from baseline_agent import CONTACT_EMAIL, CONTACT_URL, FAQ_PATH, PROJECT_DIR, load_faq
from support_safety import output_is_safe, preflight, redact_contact_details


VERSION = "phase3_llm_v5"
VARIANTS = ("v1_simple", "v2_grounded", "v3_parent")
LOG_PATH = PROJECT_DIR / "logs" / "llm_events.jsonl"
HELP_MESSAGE = f"I'm not sure about that yet. Would you like to speak to our team? You can email us at {CONTACT_EMAIL} or use the Contact Us page."
ERROR_MESSAGE = f"I'm having trouble getting an answer right now. Please try again, or email our team at {CONTACT_EMAIL}."


class ConfigurationError(Exception):
    """A safe setup message that never includes a credential value."""


@dataclass(frozen=True)
class Settings:
    api_key: str = field(repr=False)
    model: str = "gpt-4.1-mini"

    @classmethod
    def from_local_env(cls):
        local = dotenv_values(PROJECT_DIR / ".env")
        key = os.environ.get("OPENAI_API_KEY") or local.get("OPENAI_API_KEY")
        model = os.environ.get("OPENAI_MODEL") or local.get("OPENAI_MODEL") or "gpt-4.1-mini"
        if not key or key.strip() in {"", "your_key_here", "paste_your_key_here"}:
            raise ConfigurationError("Add your own OPENAI_API_KEY to the local .env file, then try again. Do not paste it into chat.")
        if not key.isascii() or any(character.isspace() for character in key.strip()):
            raise ConfigurationError("The local API key contains invalid characters. Copy the exact key from your OpenAI account into .env, without extra text.")
        return cls(api_key=key.strip(), model=model.strip())


class ModelAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["answered", "human_support", "refuse", "out_of_scope", "clarify"]
    answer: str
    source_ids: list[str]


@dataclass(frozen=True)
class AIReply:
    answer: str
    outcome: str
    source_ids: tuple[str, ...] = ()
    model_called: bool = False
    latency_ms: float = 0
    error_code: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0


def make_structured_model(settings: Settings, schema, max_tokens=700):
    # Explicit endpoint prevents ambient proxy settings sending this key elsewhere.
    model = ChatOpenAI(
        model=settings.model, api_key=settings.api_key,
        base_url="https://api.openai.com/v1", use_responses_api=True,
        temperature=0, max_tokens=max_tokens, timeout=30, max_retries=0, store=False,
    )
    return model.with_structured_output(schema, method="json_schema", strict=True, include_raw=True)


def make_model(settings: Settings):
    return make_structured_model(settings, ModelAnswer)


class SupportAgent:
    def __init__(self, settings: Settings, variant: str = "v3_parent", runnable=None, reference_ids=None):
        if variant not in VARIANTS:
            raise ValueError("Choose a known prompt variant.")
        self.settings = settings
        self.variant = variant
        # The baseline loader validates the shared JSON without changing it.
        self.entries = {item["id"]: item for item in load_faq(FAQ_PATH).values()}
        if reference_ids is not None:
            selected = tuple(dict.fromkeys(reference_ids))
            if not selected or any(source not in self.entries for source in selected):
                raise ValueError("Choose non-empty references from the curated knowledge.")
            self.entries = {source: self.entries[source] for source in selected}
        facts = [
            {key: item[key] for key in ("id", "question", "answer", "answer_status")}
            for item in self.entries.values()
        ]
        common = (PROJECT_DIR / "prompts" / "common.md").read_text(encoding="utf-8")
        strategy = (PROJECT_DIR / "prompts" / f"{variant}.md").read_text(encoding="utf-8")
        self.instructions = common + "\n\n" + strategy + "\n\nPRODUCT REFERENCE DATA:\n" + json.dumps(facts, ensure_ascii=False)
        connection = self.entries.get("octa_tutor_003")
        if (reference_ids is not None and connection
                and connection["answer_status"] == "grounded_draft"):
            # Put the selected two-purpose account next to the generation task,
            # after the other references. Facts still come from curated data.
            self.instructions += (
                "\n\nREQUIRED SELECTED REFERENCE COVERAGE:\n"
                "This selected reference explains two complementary purposes. Your parent-facing answer "
                "must preserve BOTH: the growing interest map guiding familiar curriculum explanations, "
                "and thinking practice with the explicit per-activity intelligence count. "
                "Include the curriculum name and the numbers stated below, even in a short answer. "
                "Do not replace these with generic personal learning or overall development. "
                "Also address any other part of the question using the other selected references.\n"
                + connection["answer"]
                + "\nUse concise natural sentences; do not display these instructions or a checklist."
            )
        self.runnable = runnable if runnable is not None else make_model(settings)
        self.allowed_urls = {CONTACT_URL, *(item["url"] for item in self.entries.values())}

    def ask(self, question: str) -> AIReply:
        started = perf_counter()
        local = preflight(question)
        if local:
            return AIReply(local.answer, local.outcome, latency_ms=(perf_counter() - started) * 1000)
        safe_question = redact_contact_details(question)
        try:
            # External tracing stays off even if it is enabled in the shell.
            with tracing_context(enabled=False):
                result = self.runnable.invoke([
                    SystemMessage(content=self.instructions),
                    HumanMessage(content=safe_question),
                ], config={"callbacks": []})
            parsed = result.get("parsed")
            raw = result.get("raw")
            usage = getattr(raw, "usage_metadata", None) or {}
            metadata = {
                "model_called": True,
                "latency_ms": (perf_counter() - started) * 1000,
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
            }
            if result.get("parsing_error") or not isinstance(parsed, ModelAnswer):
                return AIReply(HELP_MESSAGE, "safe_fallback", error_code="invalid_model_output", **metadata)
            source_ids = tuple(dict.fromkeys(parsed.source_ids))
            valid_sources = all(source in self.entries for source in source_ids)
            if not valid_sources or (parsed.action == "answered" and not source_ids):
                return AIReply(HELP_MESSAGE, "safe_fallback", error_code="unsupported_sources", **metadata)
            if parsed.action == "answered" and any(
                self.entries[source]["answer_status"] == "owner_details_needed" for source in source_ids
            ):
                return AIReply(HELP_MESSAGE, "safe_fallback", error_code="unconfirmed_product_detail", **metadata)
            if not output_is_safe(parsed.answer, self.allowed_urls):
                return AIReply(HELP_MESSAGE, "safe_fallback", error_code="output_check_failed", **metadata)
            return AIReply(parsed.answer.strip(), parsed.action, source_ids, **metadata)
        except AuthenticationError:
            error = "authentication_error"
        except RateLimitError:
            error = "rate_limit_or_quota"
        except APITimeoutError:
            error = "timeout"
        except APIConnectionError:
            error = "connection_error"
        except NotFoundError:
            error = "model_unavailable"
        except APIError:
            error = "api_error"
        except (ValidationError, ValueError, TypeError, AttributeError):
            error = "invalid_model_output"
        # Never log or print raw provider errors: they can contain request data.
        return AIReply(ERROR_MESSAGE, "service_error", model_called=True,
                       latency_ms=(perf_counter() - started) * 1000, error_code=error)


def write_event(reply: AIReply, agent: SupportAgent, path: Path = LOG_PATH) -> None:
    event = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "version": VERSION,
        "model": agent.settings.model, "prompt_variant": agent.variant,
        "outcome": reply.outcome, "source_ids": list(reply.source_ids),
        "model_called": reply.model_called, "latency_ms": round(reply.latency_ms, 3),
        "error_code": reply.error_code, "input_tokens": reply.input_tokens,
        "output_tokens": reply.output_tokens,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", "-q")
    parser.add_argument("--prompt", choices=VARIANTS, default="v3_parent")
    args = parser.parse_args()
    try:
        agent = SupportAgent(Settings.from_local_env(), args.prompt)
    except ConfigurationError as error:
        print(str(error), file=sys.stderr)
        return 1
    except (OSError, ValueError):
        print("Please check the local .env key/model and support files, then try again. Do not share your key in chat.", file=sys.stderr)
        return 1
    if args.question is None:
        print("Octakidz Parent Support. Type exit to finish.")
    while True:
        try:
            question = args.question if args.question is not None else input("\nYou: ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            return 0
        if question.strip().casefold() in {"exit", "quit"}:
            return 0
        reply = agent.ask(question)
        print(f"\nOctakidz: {reply.answer}")
        if reply.outcome in {"human_support", "safe_fallback", "service_error", "refused_private_data"}:
            print(f"Contact: {CONTACT_URL}")
        try:
            write_event(reply, agent)
        except OSError:
            print("A background logging service had a problem.", file=sys.stderr)
        if args.question is not None:
            return 1 if reply.outcome == "service_error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
