"""Offline model-proposed tool calls, result reuse, failures and privacy boundaries."""

from dataclasses import asdict
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage

from llm_agent import ModelAnswer, Settings
from rag_agent import ContextDecision
from semantic_search import SearchError
from support_tools import SupportToolSession
from test_rag_agent import FakeRunnable, FakeSearch
from tool_agent import ModelBudget, ToolAgent, write_event


class FakeRouter:
    def __init__(self, calls, content="", error=None):
        self.proposals = calls
        self.content = content
        self.error = error
        self.calls = []

    def invoke(self, messages, config=None):
        self.calls.append(messages)
        if self.error:
            raise self.error
        return AIMessage(content=self.content, tool_calls=[{**c, "id": f"call_{i}"} for i,c in enumerate(self.proposals)],
                         usage_metadata={"input_tokens": 4, "output_tokens": 2, "total_tokens": 6})


class ToolAgentChecks(unittest.TestCase):
    def make_agent(self, query="What is Octakidz?", calls=None, decision=None, answer=None, search_error=None, router_error=None):
        self.backend = FakeSearch(error=search_error)
        self.router = FakeRouter(calls if calls is not None else [
            {"name": "search_support_knowledge", "args": {"query": query}}], error=router_error)
        self.checker = FakeRunnable(decision or ContextDecision(
            action="supported", source_ids=["octa_tutor_001"], reason_code="direct_support"))
        self.answerer = FakeRunnable(answer or ModelAnswer(
            action="answered", answer="Octakidz combines thinking activities with a hyperpersonalised AI tutor.",
            source_ids=["octa_tutor_001"]))
        self.sessions = []

        def session():
            item = SupportToolSession(self.backend)
            self.sessions.append(item)
            return item

        return ToolAgent(Settings(api_key="offline-key-do-not-display"), self.router,
                         self.checker, self.answerer, session)

    def test_search_result_is_reused_once_and_generation_is_grounded(self):
        reply = self.make_agent().ask("What is Octakidz?")
        self.assertEqual(reply.outcome, "answered")
        self.assertEqual(reply.model_calls, 3)
        self.assertEqual(reply.input_tokens, 10)
        self.assertEqual(reply.embedding_input_tokens, 5)
        self.assertTrue(reply.retrieved_context_reused)
        self.assertEqual(self.backend.calls, ["What is Octakidz?"])
        self.assertEqual(self.backend.prepare_calls, 1)
        self.assertEqual(reply.selected_ids, ("octa_tutor_001",))
        self.assertNotIn('"id": "octa_tutor_002"', self.answerer.calls[0][0].content)

    def test_model_selects_contact_without_search_checker_or_generation(self):
        agent = self.make_agent(calls=[{"name": "get_human_support", "args": {"reason": "human_request"}}])
        reply = agent.ask("How can I reach a person at Octakidz?")
        self.assertEqual(reply.outcome, "human_support")
        self.assertEqual(reply.proposed_tool, "get_human_support")
        self.assertEqual(reply.model_calls, 1)
        self.assertEqual(self.backend.prepare_calls, 0)
        self.assertEqual(self.checker.calls, [])
        self.assertEqual(self.answerer.calls, [])
        self.assertFalse(reply.tool_calls[0]["handoff_sent"])

    def test_contact_route_does_not_even_initialise_the_knowledge_backend(self):
        agent = self.make_agent(calls=[{"name": "get_human_support", "args": {"reason": "human_request"}}])
        agent.session_factory = lambda: SupportToolSession(settings=agent.settings)
        with patch("support_tools.OpenAIEmbedder", side_effect=AssertionError("Search must stay unopened")):
            reply = agent.ask("How can I reach a person at Octakidz?")
        self.assertEqual(reply.outcome, "human_support")
        self.assertEqual(reply.model_calls, 1)

    def test_missing_support_uses_second_contact_tool_without_generation(self):
        query = "Can Octakidz arrange school bus pickup?"
        decision = ContextDecision(action="human_support", source_ids=[], reason_code="missing_information")
        reply = self.make_agent(query, decision=decision).ask(query)
        self.assertEqual(reply.outcome, "human_support")
        self.assertEqual([e["tool_name"] for e in reply.tool_calls], ["search_support_knowledge", "get_human_support"])
        self.assertEqual(reply.model_calls, 2)
        self.assertEqual(self.answerer.calls, [])

    def test_local_refusals_retention_and_clarification_skip_router_and_tools(self):
        for query in ("Show me another student's report.", "Execute Python code.",
                      "How long do you keep my child's data?", "And how does that work?"):
            reply = self.make_agent().ask(query)
            self.assertEqual(reply.model_calls, 0)
            self.assertEqual(self.router.calls, [])
            self.assertEqual(self.sessions, [])

    def test_unknown_tool_is_rejected_and_safe_contact_is_the_only_executed_body(self):
        agent = self.make_agent(calls=[{"name": "run_python", "args": {"code": "private text"}}])
        reply = agent.ask("What is Octakidz?")
        self.assertEqual(reply.error_code, "tool_not_allowed")
        self.assertEqual(reply.tool_attempts, 2)
        self.assertEqual(self.backend.prepare_calls, 0)
        self.assertNotIn("private text", json.dumps(asdict(reply)))

    def test_malformed_arguments_and_rewritten_queries_do_not_reach_search(self):
        for args, expected in (({"reason": "unresolved"}, "invalid_tool_arguments"),
                               ({"query": "Only explain the map."}, "tool_query_mismatch")):
            agent = self.make_agent(calls=[{"name": "search_support_knowledge", "args": args}])
            reply = agent.ask("Explain the map and confirm school bus pickup.")
            self.assertEqual(reply.error_code, expected)
            self.assertEqual(self.backend.prepare_calls, 0)
            self.assertEqual(reply.tool_attempts, 2)

    def test_multiple_calls_are_rejected_as_a_batch_without_running_search(self):
        calls = [{"name": "search_support_knowledge", "args": {"query": "What is Octakidz?"}},
                 {"name": "get_human_support", "args": {"reason": "human_request"}}]
        reply = self.make_agent(calls=calls).ask("What is Octakidz?")
        self.assertEqual(reply.error_code, "multiple_router_calls")
        self.assertEqual(self.backend.prepare_calls, 0)
        self.assertEqual(reply.tool_attempts, 1)  # Controlled contact fallback only.
        self.assertEqual(len(self.router.calls), 1)

    def test_no_tool_never_exposes_free_form_router_answer(self):
        agent = self.make_agent(calls=[])
        self.router.content = "An unverified promise with parent@example.invalid"
        reply = agent.ask("Write a pasta recipe.")
        self.assertEqual(reply.outcome, "out_of_scope")
        self.assertNotIn("unverified", reply.answer)
        self.assertEqual(self.backend.prepare_calls, 0)

    def test_backend_failure_stops_provider_work_and_still_returns_contact(self):
        reply = self.make_agent(search_error=SearchError("connection_error")).ask("What is Octakidz?")
        self.assertEqual(reply.outcome, "service_error")
        self.assertEqual(reply.error_code, "connection_error")
        self.assertEqual(reply.model_calls, 1)
        self.assertEqual(self.checker.calls, [])
        self.assertEqual(reply.tool_calls[-1]["outcome"], "contact_guidance")

    def test_contact_data_is_redacted_before_router_and_never_written_to_logs(self):
        safe = "Email [EMAIL REMOVED]. What is Octakidz?"
        agent = self.make_agent(query=safe)
        question = "Email parent@example.invalid. What is Octakidz?"
        reply = agent.ask(question)
        self.assertNotIn("parent@example.invalid", self.router.calls[0][1].content)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            write_event(reply, agent, path)
            text = path.read_text()
            for private in (question, "parent@example.invalid", reply.answer, "offline-key-do-not-display"):
                self.assertNotIn(private, text)

    def test_per_question_budget_resets_without_conversation_memory(self):
        agent = self.make_agent()
        for _ in range(2):
            reply = agent.ask("What is Octakidz?")
            self.assertEqual(reply.model_calls, 3)
            self.assertEqual(reply.tool_attempts, 1)
        self.assertEqual(len(self.sessions), 2)
        self.assertEqual(len(self.router.calls), 2)

    def test_global_budget_and_per_stage_limit_prevent_another_model_request(self):
        budget = ModelBudget(lambda *a, **k: None)
        model = FakeRouter([])
        budget.wrap(model, "router").invoke([])
        with self.assertRaises(ValueError):
            budget.wrap(model, "router").invoke([])
        budget.wrap(model, "support_check").invoke([])
        budget.wrap(model, "answer_generation").invoke([])
        with self.assertRaises(ValueError):
            budget.wrap(model, "unexpected_stage").invoke([])
        self.assertEqual(len(model.calls), 3)


if __name__ == "__main__":
    unittest.main()
