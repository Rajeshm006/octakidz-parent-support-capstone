"""Inspect one real RAG request from the developer's terminal."""

import argparse
from dataclasses import asdict
import json
import sys

from baseline_agent import CONTACT_URL, load_faq
from llm_agent import ConfigurationError, Settings, make_model, make_structured_model
from rag_agent import ContextDecision, MAX_SOURCES, RAGAgent, TOP_K
from semantic_search import KnowledgeSearch, MIN_SCORE, OpenAIEmbedder


VERSION = "developer_trace_v1"


def print_stage(stage, **details):
    print(f"\n[DEV · {stage}]", flush=True)
    print(json.dumps(details, indent=2, ensure_ascii=False), flush=True)


class TracedSearch:
    def __init__(self, wrapped, emit):
        self.wrapped = wrapped
        self.emit = emit
        self.attempts = 0

    def prepare(self, rebuild=False):
        self.attempts += 1
        # RAGAgent only reaches here after its own local preflight passes.
        self.emit("local_check", outcome="passed")
        self.emit("index", state="preparing")
        result = self.wrapped.prepare(rebuild=rebuild)
        self.emit("index_ready", embedding_requests=result["model_calls"],
                  embedding_input_tokens=result["input_tokens"],
                  cache_reused=result["model_calls"] == 0)
        return result

    def search(self, question, top_k):
        self.attempts += 1
        self.emit("search_started", maximum_candidates=top_k,
                  minimum_similarity=getattr(self.wrapped, "min_score", MIN_SCORE))
        result = self.wrapped.search(question, top_k=top_k)
        self.emit("retrieval", outcome=result.outcome, nearest_score=result.nearest_score,
                  candidates=[{key: item[key] for key in
                               ("id", "score", "question", "answer", "answer_status")}
                              for item in result.candidates],
                  note="Similarity ranks related text; it does not prove an answer is supported.")
        return result


class TracedModel:
    def __init__(self, wrapped, stage, emit, known_ids):
        self.wrapped = wrapped
        self.stage = stage
        self.emit = emit
        self.known_ids = known_ids

    def safe_ids(self, ids):
        return [source if source in self.known_ids else "[unrecognised source ID]" for source in ids]

    def invoke(self, messages, config=None):
        details = {"state": "started"}
        if self.stage == "generation":
            # Display only IDs from the public reference block, never the full prompt.
            marker = "PRODUCT REFERENCE DATA:\n"
            try:
                facts = json.loads(messages[0].content.split(marker, 1)[1])
                details["context_source_ids"] = self.safe_ids([item["id"] for item in facts])
            except (AttributeError, IndexError, KeyError, TypeError, ValueError):
                details["context_source_ids"] = "unavailable in this prompt format"
        self.emit(self.stage, **details)
        result = self.wrapped.invoke(messages, config=config)
        if self.stage == "support_check":
            decision = result.get("parsed")
            if isinstance(decision, ContextDecision) and not result.get("parsing_error"):
                self.emit("checker_proposal", action=decision.action, reason_code=decision.reason_code,
                          proposed_source_ids=self.safe_ids(decision.source_ids),
                          note="Python validates this proposal before allowing generation.")
            else:
                self.emit("checker_proposal", state="invalid_structured_decision")
        return result


def make_debug_agent(settings, emit=print_stage, *, search=None, checker=None, answerer=None,
                     rebuild_index=False):
    known_ids = {item["id"] for item in load_faq().values()}
    search = search if search is not None else KnowledgeSearch(OpenAIEmbedder(settings))
    checker = checker if checker is not None else make_structured_model(settings, ContextDecision, max_tokens=300)
    answerer = answerer if answerer is not None else make_model(settings)
    return RAGAgent(settings, search=TracedSearch(search, emit),
                    checker=TracedModel(checker, "support_check", emit, known_ids),
                    answerer=TracedModel(answerer, "generation", emit, known_ids),
                    rebuild_index=rebuild_index)


def inspect_question(agent, question, emit=print_stage):
    emit("start", mode=VERSION, model=agent.settings.model,
         maximum_candidates=TOP_K, maximum_selected_sources=MAX_SOURCES)
    search_attempts = agent.search.attempts
    reply = agent.ask(question)
    if agent.search.attempts == search_attempts and not reply.model_called and not reply.embedding_called:
        emit("local_check", outcome=reply.outcome, later_stages="skipped")
    if not reply.answerer_called:
        emit("generation", state="skipped")
    # The parent sees the separately printed checked answer. Do not dump raw model output.
    metadata = asdict(reply)
    metadata.pop("answer")
    emit("validated_result", **metadata)
    return reply


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", "-q", required=True)
    parser.add_argument("--rebuild-index", action="store_true")
    args = parser.parse_args()
    try:
        agent = make_debug_agent(Settings.from_local_env(), rebuild_index=args.rebuild_index)
    except (ConfigurationError, OSError, ValueError):
        print("Check your local key and support files. No credentials or raw provider errors are displayed.", file=sys.stderr)
        return 1
    print("Developer view: one question through the normal RAG pipeline; diagnostics appear in this terminal.")
    reply = inspect_question(agent, args.question)
    print(f"\nOctakidz: {reply.answer}")
    if reply.outcome in {"human_support", "safe_fallback", "service_error", "refused_private_data"}:
        print(f"Contact: {CONTACT_URL}")
    return 1 if reply.outcome == "service_error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
