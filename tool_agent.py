"""Phase 5: one model-selected tool, bounded execution and grounded replies."""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from time import perf_counter
import sys

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langsmith import tracing_context
from openai import APIConnectionError, APIError, APITimeoutError, AuthenticationError, NotFoundError, RateLimitError

from baseline_agent import CONTACT_URL, PROJECT_DIR
from llm_agent import ConfigurationError, ERROR_MESSAGE, HELP_MESSAGE, Settings, make_model, make_structured_model
from rag_agent import ContextDecision, RAGAgent, RAGReply, SCOPE_MESSAGE
from semantic_search import SearchResult
from support_safety import output_is_safe, preflight, redact_contact_details
from support_tools import SupportToolSession


VERSION = "phase5_agent_v1"
MAX_MODEL_CALLS = 3
LOG_PATH = PROJECT_DIR / "logs/tool_events.jsonl"
TOOL_NAMES = {"search_support_knowledge", "get_human_support"}


def make_router(settings, query_constraint=None):
    functions = []
    for schema in SupportToolSession().schemas():
        # Local Pydantic validation retains length limits; the wire uses a simple strict schema.
        if schema["name"] == "search_support_knowledge":
            schema["parameters"]["properties"]["query"].pop("minLength", None)
            schema["parameters"]["properties"]["query"].pop("maxLength", None)
            if query_constraint is not None:
                schema["parameters"]["properties"]["query"]["enum"] = [query_constraint]
        functions.append({"type": "function", "function": schema})
    model = ChatOpenAI(model=settings.model, api_key=settings.api_key,
                       base_url="https://api.openai.com/v1", use_responses_api=True,
                       temperature=0, max_tokens=350, timeout=30, max_retries=0, store=False)
    return model.bind_tools(functions, strict=True, parallel_tool_calls=False, tool_choice="auto")


class ModelBudget:
    """One router, one support check, one answerer; no autonomous tool loop."""
    def __init__(self, emit):
        self.count = 0
        self.stages = set()
        self.input_tokens = 0
        self.output_tokens = 0
        self.usage_complete = True
        self.emit = emit

    def wrap(self, runnable, stage):
        budget = self

        class BoundedModel:
            def invoke(self, messages, config=None):
                if budget.count >= MAX_MODEL_CALLS or stage in budget.stages:
                    raise ValueError("model_call_limit")
                budget.count += 1
                budget.stages.add(stage)
                budget.emit("model_stage", stage_name=stage, call=budget.count, maximum=MAX_MODEL_CALLS)
                try:
                    with tracing_context(enabled=False):
                        result = runnable.invoke(messages, config={"callbacks": []})
                except Exception:
                    budget.usage_complete = False
                    raise
                raw = result.get("raw") if isinstance(result, dict) else result
                parsed = result.get("parsed") if isinstance(result, dict) else None
                if stage == "support_check" and isinstance(parsed, ContextDecision):
                    public_ids = [s for s in parsed.source_ids if re.fullmatch(r"octa_tutor_\d{3}", s)]
                    budget.emit("checker_proposal", action=parsed.action, reason_code=parsed.reason_code,
                                proposed_source_ids=public_ids[:5], proposed_source_count=len(parsed.source_ids),
                                invalid_id_format_count=len(parsed.source_ids)-len(public_ids))
                usage = getattr(raw, "usage_metadata", None)
                if usage is None:
                    budget.usage_complete = False
                else:
                    budget.input_tokens += usage.get("input_tokens", 0)
                    budget.output_tokens += usage.get("output_tokens", 0)
                return result

        return BoundedModel()


class RetrievedToolContext:
    """Give the existing RAG checker tool results without a second embedding request."""
    def __init__(self, result, query):
        self.result = result
        self.query = query

    def prepare(self, rebuild=False):
        return {"model_calls": 0, "input_tokens": 0}

    def search(self, question, top_k):
        if redact_contact_details(question).strip() != self.query.strip():
            raise ValueError("tool_context_question_mismatch")
        data = self.result["data"]
        return SearchResult(self.result["outcome"], data["candidates"][:top_k],
                            nearest_score=data["nearest_score"])


@dataclass(frozen=True)
class ToolReply(RAGReply):
    index_embedding_calls: int | None = 0
    router_called: bool = False
    proposed_tool: str | None = None
    model_calls: int = 0
    maximum_model_calls: int = MAX_MODEL_CALLS
    usage_complete: bool = True
    tool_attempts: int = 0
    tool_calls: tuple[dict, ...] = ()
    contact_url: str | None = None
    retrieved_context_reused: bool = False


class ToolAgent:
    def __init__(self, settings, router=None, checker=None, answerer=None, session_factory=None):
        self.settings = settings
        self.router = router if router is not None else make_router(settings)
        self.checker = checker if checker is not None else make_structured_model(settings, ContextDecision, max_tokens=300)
        self.answerer = answerer if answerer is not None else make_model(settings)
        self.session_factory = session_factory or (lambda: SupportToolSession(settings=settings))
        self.instructions = (PROJECT_DIR / "prompts/tool_router.md").read_text()

    def ask(self, question, emit=None):
        started = perf_counter()
        emit = emit or (lambda *args, **kwargs: None)
        budget = ModelBudget(emit)
        metadata = asdict(ToolReply(answer="", outcome=""))
        events = []
        session = None

        def finish(answer, outcome, error_code=None):
            metadata.update(answer=answer, outcome=outcome, error_code=error_code,
                            latency_ms=(perf_counter()-started)*1000, model_called=budget.count > 0,
                            model_calls=budget.count, input_tokens=budget.input_tokens,
                            output_tokens=budget.output_tokens, usage_complete=budget.usage_complete,
                            tool_attempts=session.attempts if session else 0, tool_calls=tuple(events))
            reply = ToolReply(**metadata)
            emit("validated_result", **{k:v for k,v in asdict(reply).items() if k != "answer"})
            return reply

        def call_tool(name, arguments, expected_query=None):
            result = session.invoke(name, arguments, expected_query=expected_query)
            fields = ("tool_name", "attempt", "outcome", "error_code", "index_preparation_attempted",
                      "index_embedding_calls", "query_embedding_attempted", "embedding_input_tokens")
            event = {key: result[key] for key in fields if key in result}
            if result["outcome"] == "contact_guidance":
                event["handoff_sent"] = result["data"]["handoff_sent"]
            events.append(event)
            emit("tool_result", **event)
            return result

        def contact(answer=None, outcome="human_support", error_code=None, reason="unresolved"):
            result = call_tool("get_human_support", {"reason": reason})
            if result["outcome"] == "contact_guidance" and result["data"].get("handoff_sent") is False:
                route = result["data"]["url"]
                message = result["data"]["message"]
                if route == CONTACT_URL and output_is_safe(message, {CONTACT_URL}):
                    metadata["contact_url"] = route
                    return finish(answer or message, outcome, error_code)
            return finish(HELP_MESSAGE, "safe_fallback", error_code or "invalid_contact_result")

        local = preflight(question)
        if local:
            emit("local_check", outcome=local.outcome, router="skipped")
            if local.outcome in {"human_support", "refused_private_data"}:
                metadata["contact_url"] = CONTACT_URL
            return finish(local.answer, local.outcome)
        emit("local_check", outcome="passed")
        safe_question = redact_contact_details(question)
        try:
            session = self.session_factory()
            metadata["router_called"] = True
            routed = budget.wrap(self.router, "router").invoke([
                SystemMessage(content=self.instructions), HumanMessage(content=safe_question)])
            calls = getattr(routed, "tool_calls", None)
            invalid = getattr(routed, "invalid_tool_calls", None)
            if not isinstance(calls, list) or invalid:
                return contact(outcome="safe_fallback", error_code="invalid_router_call")
            if len(calls) > 1:
                # Reject the whole batch before executing any proposed operation.
                emit("router_choice", outcome="rejected_multiple_calls")
                return contact(outcome="safe_fallback", error_code="multiple_router_calls")
            if not calls:
                emit("router_choice", outcome="no_tool")
                # Never expose ungrounded free-form router text as a parent answer.
                return finish(SCOPE_MESSAGE, "out_of_scope")
            call = calls[0]
            if not isinstance(call, dict):
                return contact(outcome="safe_fallback", error_code="invalid_router_call")
            name, arguments = call.get("name"), call.get("args")
            metadata["proposed_tool"] = name if name in TOOL_NAMES else None
            emit("router_choice", tool_name=metadata["proposed_tool"],
                 query_source="original_redacted_question" if name == "search_support_knowledge" else None)
            result = call_tool(name, arguments, expected_query=safe_question)
            if name == "search_support_knowledge":
                metadata.update(embedding_called=result.get("query_embedding_attempted", False),
                                index_embedding_calls=result.get("index_embedding_calls", 0),
                                embedding_input_tokens=result.get("embedding_input_tokens", 0))
            if result["outcome"] in {"blocked", "tool_error"}:
                return contact(outcome="service_error" if result["outcome"] == "tool_error" else "safe_fallback",
                               error_code=result["error_code"])
            if name == "get_human_support":
                data = result["data"]
                if (data.get("handoff_sent") is not False or data.get("url") != CONTACT_URL
                        or not output_is_safe(data["message"], {CONTACT_URL})):
                    return finish(HELP_MESSAGE, "safe_fallback", "invalid_contact_result")
                metadata["contact_url"] = data["url"]
                return finish(data["message"], "human_support")
            if result["outcome"] not in {"candidates_only", "insufficient_relevance"}:
                return finish(result["data"]["message"], result["outcome"])
            emit("retrieval", **result["data"])
            context = RetrievedToolContext(result, safe_question)
            grounded = RAGAgent(self.settings, search=context,
                                checker=budget.wrap(self.checker, "support_check"),
                                answerer=budget.wrap(self.answerer, "answer_generation")).ask(question)
            for key in ("source_ids", "retrieved_ids", "selected_ids", "nearest_score", "checker_called",
                        "answerer_called", "decision_action", "decision_reason"):
                metadata[key] = getattr(grounded, key)
            metadata["retrieved_context_reused"] = True
            emit("source_check", decision=grounded.decision_action, reason_code=grounded.decision_reason,
                 selected_ids=grounded.selected_ids, generation_called=grounded.answerer_called)
            if grounded.outcome in {"human_support", "safe_fallback", "service_error"}:
                return contact(grounded.answer, grounded.outcome, grounded.error_code,
                               "missing_information" if grounded.outcome == "human_support" else "unresolved")
            return finish(grounded.answer, grounded.outcome, grounded.error_code)
        except APIError as error:
            code = "routing_api_error"
            for kind, name in ((AuthenticationError, "authentication_error"), (RateLimitError, "rate_limit_or_quota"),
                               (APITimeoutError, "timeout"), (APIConnectionError, "connection_error"),
                               (NotFoundError, "model_unavailable")):
                if isinstance(error, kind):
                    code = name
                    break
        except Exception:
            code = "invalid_routing_or_tool_result"
        # Stop after an exception. No further model/tool retries are attempted.
        return finish(ERROR_MESSAGE, "service_error", code)


def write_event(reply, agent, path=LOG_PATH):
    event = asdict(reply)
    event.pop("answer")
    event.update(timestamp_utc=datetime.now(timezone.utc).isoformat(), version=VERSION, model=agent.settings.model)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(event)+"\n")


def print_debug(stage, **details):
    print(f"\n[DEV · {stage}]\n"+json.dumps(details, indent=2, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", "-q")
    parser.add_argument("--debug", action="store_true", help="Show safe tool/source diagnostics in this terminal.")
    args = parser.parse_args()
    try:
        agent = ToolAgent(Settings.from_local_env())
    except (ConfigurationError, OSError, ValueError):
        print("Check your local key, prompts and support files. Do not share your API key.", file=sys.stderr)
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
        reply = agent.ask(question, print_debug if args.debug else None)
        print(f"\nOctakidz: {reply.answer}")
        if reply.contact_url:
            print(f"Contact: {reply.contact_url}")
        try:
            write_event(reply, agent)
        except OSError:
            print("A background logging service had a problem.", file=sys.stderr)
        if args.question is not None:
            return 1 if reply.outcome == "service_error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
