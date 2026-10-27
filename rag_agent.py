"""Phase 4: search, check source support, then answer from selected references."""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langsmith import tracing_context
from openai import APIConnectionError, APIError, APITimeoutError, AuthenticationError, NotFoundError, RateLimitError
from pydantic import BaseModel, ConfigDict

from baseline_agent import CONTACT_URL, PROJECT_DIR
from llm_agent import AIReply, ConfigurationError, ERROR_MESSAGE, HELP_MESSAGE, Settings, SupportAgent, make_model, make_structured_model
from semantic_search import KnowledgeSearch, OpenAIEmbedder, SearchError
from support_safety import output_is_safe, preflight, redact_contact_details


VERSION = "phase4_rag_v2"
LOG_PATH = PROJECT_DIR / "logs" / "rag_events.jsonl"
TOP_K = 5
MAX_SOURCES = 3
SCOPE_MESSAGE = "I'm here to help with Octakidz, its activities and AI tutor. Please ask about how Octakidz can support your child's learning."
REFUSAL_MESSAGE = "I can help with Octakidz questions, but I can't fulfil that request. Please contact our team for confirmed product details or account support."
CLARIFY_MESSAGE = "Could you tell me which part of Octakidz you'd like help with?"


class ContextDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["supported", "human_support", "out_of_scope", "refuse", "clarify"]
    source_ids: list[str]
    reason_code: Literal["direct_support", "missing_information", "partial_support", "unrelated", "unsafe", "ambiguous"]


@dataclass(frozen=True)
class RAGReply(AIReply):
    retrieved_ids: tuple[str, ...] = ()
    selected_ids: tuple[str, ...] = ()
    nearest_score: float | None = None
    embedding_called: bool = False
    index_embedding_calls: int = 0
    embedding_input_tokens: int = 0
    checker_called: bool = False
    answerer_called: bool = False
    decision_action: str | None = None
    decision_reason: str | None = None


class RAGAgent:
    def __init__(self, settings, search=None, checker=None, answerer=None, rebuild_index=False):
        self.settings = settings
        self.search = search if search is not None else KnowledgeSearch(OpenAIEmbedder(settings))
        self.checker = checker if checker is not None else make_structured_model(settings, ContextDecision, max_tokens=300)
        self.answerer = answerer if answerer is not None else make_model(settings)
        common = (PROJECT_DIR / "prompts/common.md").read_text(encoding="utf-8")
        marker = "\nReturn the required structured response:"
        if marker not in common:
            raise ValueError("Check the shared instruction format.")
        self.check_instructions = common.split(marker, 1)[0] + "\n\n" + (PROJECT_DIR / "prompts/retrieval_check.md").read_text(encoding="utf-8")
        self.prepared = False
        self.rebuild_index = rebuild_index

    def ask(self, question):
        started = perf_counter()
        metadata = dict(source_ids=(), retrieved_ids=(), selected_ids=(), nearest_score=None,
                        embedding_called=False, index_embedding_calls=0, embedding_input_tokens=0,
                        checker_called=False, answerer_called=False, decision_action=None, decision_reason=None,
                        input_tokens=0, output_tokens=0)

        def finish(answer, outcome, error_code=None):
            return RAGReply(answer=answer, outcome=outcome, error_code=error_code,
                            model_called=metadata["checker_called"] or metadata["answerer_called"],
                            latency_ms=(perf_counter()-started)*1000, **metadata)

        local = preflight(question)
        if local:
            return finish(local.answer, local.outcome)
        try:
            # Even index preparation waits until after local privacy/scope checks.
            if not self.prepared:
                index_info = self.search.prepare(rebuild=self.rebuild_index)
                metadata["index_embedding_calls"] = index_info["model_calls"]
                metadata["embedding_input_tokens"] += index_info["input_tokens"]
                self.prepared = True
            metadata["embedding_called"] = True
            found = self.search.search(question, top_k=TOP_K)
            metadata["embedding_input_tokens"] += found.input_tokens
            metadata["nearest_score"] = found.nearest_score
            candidates = {item["id"]: item for item in found.candidates}
            metadata["retrieved_ids"] = tuple(candidates)
            # Send only candidate references; scores/URLs are unnecessary for support checking.
            facts = [{key:item[key] for key in ("id", "question", "answer", "answer_status")} for item in candidates.values()]
            metadata["checker_called"] = True
            with tracing_context(enabled=False):
                checked = self.checker.invoke([
                    SystemMessage(content=self.check_instructions+"\n\nCANDIDATE REFERENCES:\n"+json.dumps(facts,ensure_ascii=False)),
                    HumanMessage(content=redact_contact_details(question)),
                ], config={"callbacks": []})
            usage = getattr(checked.get("raw"), "usage_metadata", None) or {}
            metadata["input_tokens"] += usage.get("input_tokens", 0)
            metadata["output_tokens"] += usage.get("output_tokens", 0)
            decision = checked.get("parsed")
            if checked.get("parsing_error") or not isinstance(decision, ContextDecision):
                return finish(HELP_MESSAGE, "safe_fallback", "invalid_context_decision")
            metadata["decision_action"] = decision.action
            metadata["decision_reason"] = decision.reason_code
            selected = tuple(dict.fromkeys(decision.source_ids))
            expected_reason = {"supported":"direct_support", "human_support":None,
                               "out_of_scope":"unrelated", "refuse":"unsafe", "clarify":"ambiguous"}
            coherent = decision.reason_code in {"missing_information", "partial_support"} if decision.action == "human_support" else decision.reason_code == expected_reason[decision.action]
            if (not coherent or len(selected) > MAX_SOURCES or any(source not in candidates for source in selected)
                    or (decision.action == "supported" and not selected)
                    or (decision.action in {"out_of_scope", "refuse", "clarify"} and selected)):
                return finish(HELP_MESSAGE, "safe_fallback", "unsupported_context_selection")
            metadata["selected_ids"] = selected
            unknown_detail = any(candidates[source]["answer_status"] == "owner_details_needed" for source in selected)
            if decision.action == "human_support" or unknown_detail:
                answer = HELP_MESSAGE
                # A single unknown-detail reference is itself a safe curated referral.
                if len(selected) == 1 and candidates[selected[0]]["answer_status"] == "owner_details_needed":
                    item = candidates[selected[0]]
                    if output_is_safe(item["answer"], {CONTACT_URL, item["url"]}):
                        answer = item["answer"]
                        metadata["source_ids"] = selected
                return finish(answer, "human_support", "unknown_detail_blocked" if unknown_detail and decision.action == "supported" else None)
            if decision.action == "out_of_scope":
                return finish(SCOPE_MESSAGE, "out_of_scope")
            if decision.action == "refuse":
                return finish(REFUSAL_MESSAGE, "refuse")
            if decision.action == "clarify":
                return finish(CLARIFY_MESSAGE, "clarify")
            # This constructor limits both the request context AND valid answer citations.
            narrowed = SupportAgent(self.settings, runnable=self.answerer, reference_ids=selected)
            metadata["answerer_called"] = True
            reply = narrowed.ask(question)
            metadata["input_tokens"] += reply.input_tokens
            metadata["output_tokens"] += reply.output_tokens
            metadata["source_ids"] = reply.source_ids
            return finish(reply.answer, reply.outcome, reply.error_code)
        except SearchError as error:
            return finish(ERROR_MESSAGE, "service_error", str(error))
        except AuthenticationError:
            error_code = "authentication_error"
        except RateLimitError:
            error_code = "rate_limit_or_quota"
        except APITimeoutError:
            error_code = "timeout"
        except APIConnectionError:
            error_code = "connection_error"
        except NotFoundError:
            error_code = "model_unavailable"
        except APIError:
            error_code = "api_error"
        except (OSError, ValueError, TypeError, AttributeError, KeyError):
            return finish(HELP_MESSAGE, "safe_fallback", "invalid_context_or_knowledge")
        return finish(ERROR_MESSAGE, "service_error", error_code)


def write_event(reply, agent, path=LOG_PATH):
    values = asdict(reply)
    # No raw question, answer, reference text or free-form model reasoning in logs.
    values.pop("answer")
    values.update(timestamp_utc=datetime.now(timezone.utc).isoformat(), version=VERSION, model=agent.settings.model)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(values)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", "-q")
    parser.add_argument("--rebuild-index", action="store_true")
    args = parser.parse_args()
    try:
        agent = RAGAgent(Settings.from_local_env(), rebuild_index=args.rebuild_index)
    except (ConfigurationError, OSError, ValueError):
        print("Check your local key, prompts and support files. Do not share your API key in chat.",file=sys.stderr)
        return 1
    if args.question is None:
        print("Octakidz Parent Support. Type exit to finish.")
    while True:
        try:
            question = args.question if args.question is not None else input("\nYou: ")
        except (EOFError,KeyboardInterrupt):
            print("\nGoodbye!")
            return 0
        if question.strip().casefold() in {"exit", "quit"}:
            return 0
        reply = agent.ask(question)
        print(f"\nOctakidz: {reply.answer}")
        if reply.outcome in {"human_support", "safe_fallback", "service_error", "refused_private_data"}:
            print(f"Contact: {CONTACT_URL}")
        try:
            write_event(reply,agent)
        except OSError:
            print("A background logging service had a problem.",file=sys.stderr)
        if args.question is not None:
            return 1 if reply.outcome == "service_error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
