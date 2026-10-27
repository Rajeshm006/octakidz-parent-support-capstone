"""Phase 5, step 1: two bounded support tools, called manually for learning."""

import argparse
import json
from time import perf_counter
from typing import Literal

from langchain_core.tools import tool
from langsmith import tracing_context
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from baseline_agent import CONTACT_EMAIL, CONTACT_URL
from llm_agent import ConfigurationError, Settings
from semantic_search import KnowledgeSearch, OpenAIEmbedder, SearchError
from support_safety import preflight, redact_contact_details


VERSION = "phase5_tools_v2"
MAX_ATTEMPTS = 2
TOP_K = 5
SAFE_SEARCH_ERRORS = {
    "authentication_error", "rate_limit_or_quota", "timeout", "connection_error", "api_error",
    "stale_index_use_rebuild", "invalid_index_use_rebuild", "invalid_embeddings", "prepare_index_first",
}


class KnowledgeArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)
    query: str = Field(min_length=1, max_length=4000,
                       description="One Octakidz support question; omit personal contact details.")


class ContactArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    reason: Literal["human_request", "missing_information", "unresolved", "privacy", "account_support"] = Field(
        description="A category only; never include a child's name, account ID or free-text summary.")


class SupportToolSession:
    """Create a fresh instance for each question. No conversation history is kept.

    All execution goes through invoke(), including future model-proposed calls.
    Definitions alone do not enforce the dispatch budget.
    """

    def __init__(self, search=None, settings=None):
        self.search = search
        self.settings = settings
        self.attempts = 0
        self.used = set()

        @tool("search_support_knowledge", args_schema=KnowledgeArguments)
        def search_support_knowledge(query: str) -> dict:
            """Find public Octakidz FAQ candidates for product, activities and tutor questions.

            Results are candidate evidence only. They require the source-support check before
            generating an answer. Missing-detail entries do not confirm a price or policy.
            No private records, arbitrary files, URLs or executable code can be accessed.
            """
            return self._search(query)

        @tool("get_human_support", args_schema=ContactArguments)
        def get_human_support(reason: str) -> dict:
            """Return configured Octakidz contact guidance for human help or missing details.

            This returns an email and contact-page URL. It never sends a message, creates a
            ticket, transfers a parent or opens an account. Accept a category, not personal data.
            """
            return {"outcome": "contact_guidance", "data": {
                "reason": reason, "email": CONTACT_EMAIL, "url": CONTACT_URL,
                "message": f"You can email our team at {CONTACT_EMAIL} or use the Contact Us page for help.",
                "handoff_sent": False,
            }}

        self._tools = {t.name: t for t in (search_support_knowledge, get_human_support)}

    def schemas(self):
        return [{"name": t.name, "description": t.description,
                 "parameters": t.args_schema.model_json_schema()} for t in self._tools.values()]

    def invoke(self, name, arguments, *, expected_query=None):
        """Validate and dispatch one allowlisted call; never echo rejected input."""
        started = perf_counter()
        self.attempts += 1
        allowed = isinstance(name, str) and name in self._tools
        metadata = {"tool_name": name if allowed else None, "attempt": self.attempts,
                    "maximum_attempts": MAX_ATTEMPTS}

        def finish(payload):
            return {**metadata, **payload, "latency_ms": (perf_counter() - started) * 1000}

        def reject(code):
            return finish({"outcome": "blocked", "data": {}, "error_code": code})

        if self.attempts > MAX_ATTEMPTS:
            return reject("tool_attempt_limit")
        if not allowed:
            return reject("tool_not_allowed")
        if name in self.used:
            return reject("repeated_tool_call")
        # Invalid calls also consume their attempt; do not retry them indefinitely.
        self.used.add(name)
        if not isinstance(arguments, dict):
            return reject("invalid_tool_arguments")
        try:
            arguments = self._tools[name].args_schema.model_validate(arguments).model_dump()
            # A routing model must not omit or rewrite material parts of the parent's question.
            if name == "search_support_knowledge" and expected_query is not None:
                if redact_contact_details(arguments["query"]).strip() != expected_query.strip():
                    return reject("tool_query_mismatch")
            with tracing_context(enabled=False):
                payload = self._tools[name].invoke(arguments, config={"callbacks": []})
            return finish({"error_code": None, **payload})
        except ValidationError:
            return reject("invalid_tool_arguments")
        except Exception:
            # Provider/tool exceptions can contain input or credentials. Never dump them.
            return finish({"outcome": "tool_error", "data": {}, "error_code": "tool_backend_error"})

    def _search(self, query):
        local = preflight(query)
        if local:
            return {"outcome": local.outcome,
                    "data": {"message": local.answer, "candidates": [], "support_check_required": True},
                    "index_preparation_attempted": False, "index_embedding_calls": 0,
                    "query_embedding_attempted": False, "embedding_input_tokens": 0}
        metrics = {"index_preparation_attempted": False, "index_embedding_calls": 0,
                   "query_embedding_attempted": False, "embedding_input_tokens": 0}
        try:
            # Contact guidance and local refusals do not even need a provider key.
            if self.search is None:
                settings = self.settings if self.settings is not None else Settings.from_local_env()
                self.search = KnowledgeSearch(OpenAIEmbedder(settings))
            metrics.update(index_preparation_attempted=True, index_embedding_calls=None)
            prepared = self.search.prepare()
            metrics.update(index_embedding_calls=prepared["model_calls"],
                           embedding_input_tokens=prepared["input_tokens"])
            metrics["query_embedding_attempted"] = True
            found = self.search.search(redact_contact_details(query), top_k=TOP_K)
            metrics["embedding_input_tokens"] += found.input_tokens
            candidates = [{key: item[key] for key in
                           ("id", "question", "answer", "answer_status", "url", "score")}
                          for item in found.candidates]
            return {"outcome": found.outcome,
                    "data": {"candidates": candidates, "nearest_score": found.nearest_score,
                             "support_check_required": True}, **metrics}
        except SearchError as error:
            code = str(error) if str(error) in SAFE_SEARCH_ERRORS else "tool_backend_error"
        except ConfigurationError:
            code = "provider_configuration_needed"
        except Exception:
            code = "tool_backend_error"
        return {"outcome": "tool_error", "data": {}, "error_code": code, **metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("schemas", help="Show the two tool definitions without API calls.")
    contact = commands.add_parser("contact", help="Return contact guidance without API calls.")
    contact.add_argument("--reason", default="human_request")
    search = commands.add_parser("search", help="Find candidate sources using OpenAI embeddings.")
    search.add_argument("--question", "-q", required=True)
    args = parser.parse_args()
    session = SupportToolSession()
    if args.command == "schemas":
        print(json.dumps(session.schemas(), indent=2, ensure_ascii=False))
        return 0
    name, arguments = ("get_human_support", {"reason": args.reason}) if args.command == "contact" else (
        "search_support_knowledge", {"query": args.question})
    result = session.invoke(name, arguments)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 1 if result["outcome"] in {"blocked", "tool_error"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
