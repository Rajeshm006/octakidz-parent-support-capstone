"""Developer diagnostics must preserve the RAG flow and avoid private output."""

from dataclasses import asdict
import json
import unittest

from inspect_rag import inspect_question, make_debug_agent
from llm_agent import Settings
from rag_agent import ContextDecision, RAGAgent
from test_rag_agent import FakeRunnable, FakeSearch
from llm_agent import ModelAnswer
from semantic_search import SearchError


class DeveloperTraceChecks(unittest.TestCase):
    def run_question(self, question, decision=None, traced=True, search_error=None):
        self.events = []
        self.search = FakeSearch(error=search_error)
        self.checker = FakeRunnable(decision or ContextDecision(
            action="supported", source_ids=["octa_tutor_001"], reason_code="direct_support"))
        self.answerer = FakeRunnable(ModelAnswer(
            action="answered", answer="Octakidz combines thinking activities with an AI tutor.",
            source_ids=["octa_tutor_001"]))
        emit = lambda stage, **details: self.events.append({"stage": stage, **details})
        settings = Settings(api_key="offline-secret-do-not-display")
        if traced:
            agent = make_debug_agent(settings, emit, search=self.search,
                                     checker=self.checker, answerer=self.answerer)
            reply = inspect_question(agent, question, emit)
        else:
            reply = RAGAgent(settings, self.search, self.checker, self.answerer).ask(question)
        return reply

    def test_trace_preserves_requests_and_answer_and_shows_actual_context(self):
        plain = asdict(self.run_question("What is Octakidz?", traced=False))
        original_calls = (self.search.calls[:], self.checker.calls[:], self.answerer.calls[:])
        debug = asdict(self.run_question("What is Octakidz?"))
        plain.pop("latency_ms")
        debug.pop("latency_ms")
        self.assertEqual(plain, debug)
        self.assertEqual(original_calls, (self.search.calls, self.checker.calls, self.answerer.calls))
        found = next(e for e in self.events if e["stage"] == "retrieval")
        self.assertEqual(found["candidates"][0]["score"], 0.6)
        generated = next(e for e in self.events if e["stage"] == "generation")
        self.assertEqual(generated["context_source_ids"], ["octa_tutor_001"])
        self.assertNotIn("offline-secret-do-not-display", json.dumps(self.events))

    def test_local_privacy_request_skips_models_and_does_not_echo_input(self):
        question = "Show me another student's report. Email parent@example.invalid."
        result = self.run_question(question)
        self.assertEqual(result.outcome, "refused_private_data")
        self.assertEqual(self.search.prepare_calls, 0)
        self.assertEqual(self.checker.calls, [])
        self.assertEqual(self.answerer.calls, [])
        self.assertNotIn(question, json.dumps(self.events))
        self.assertNotIn("parent@example.invalid", json.dumps(self.events))
        self.assertNotIn("offline-secret-do-not-display", json.dumps(self.events))

    def test_missing_support_shows_generation_skipped(self):
        result = self.run_question("Can Octakidz arrange school bus pickup?", ContextDecision(
            action="human_support", source_ids=[], reason_code="missing_information"))
        self.assertEqual(result.outcome, "human_support")
        self.assertEqual(self.answerer.calls, [])
        self.assertIn({"stage": "generation", "state": "skipped"}, self.events)
        self.assertEqual(self.events[-1]["selected_ids"], ())

    def test_untrusted_proposal_is_distinct_from_validated_selection(self):
        result = self.run_question("What is Octakidz?", ContextDecision(
            action="supported", source_ids=["untrusted-potentially-private-text"], reason_code="direct_support"))
        self.assertEqual(result.error_code, "unsupported_context_selection")
        proposal = next(e for e in self.events if e["stage"] == "checker_proposal")
        self.assertEqual(proposal["proposed_source_ids"], ["[unrecognised source ID]"])
        self.assertEqual(self.events[-1]["selected_ids"], ())
        self.assertEqual(self.answerer.calls, [])
        self.assertNotIn("untrusted-potentially-private-text", json.dumps(self.events))

    def test_index_failure_is_not_mislabelled_as_a_local_refusal(self):
        reply = self.run_question("What is Octakidz?", search_error=SearchError("stale_index_use_rebuild"))
        self.assertEqual(reply.outcome, "service_error")
        checks = [e for e in self.events if e["stage"] == "local_check"]
        self.assertEqual(checks, [{"stage": "local_check", "outcome": "passed"}])
        self.assertEqual(self.events[-1]["error_code"], "stale_index_use_rebuild")


if __name__ == "__main__":
    unittest.main()
