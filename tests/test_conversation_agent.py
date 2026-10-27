"""Offline memory, planning and isolation boundaries; no real API calls."""

from dataclasses import asdict
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from langchain_core.messages import HumanMessage

from conversation_agent import ContextRouter, ContextToolAgent, ConversationAgent, RETENTION_SECONDS, VERSION, write_event
from llm_agent import ModelAnswer, Settings
from rag_agent import ContextDecision
from support_tools import SupportToolSession
from test_rag_agent import FakeRunnable, FakeSearch
from test_tool_agent import FakeRouter
from tool_agent import ToolAgent, ToolReply, make_router


class FakeAgent:
    def __init__(self):
        self.settings = SimpleNamespace(model="offline", api_key="never-display-key")
        self.calls = []
        self.reply = ToolReply(answer="A private generated sentence must never be replayed.", outcome="answered",
                               source_ids=("octa_tutor_004",), selected_ids=("octa_tutor_004",),
                               retrieved_ids=("octa_tutor_004",), model_calls=3)

    def ask(self, question, emit=None):
        self.calls.append(question)
        return self.reply


class MemoryChecks(unittest.TestCase):
    def setUp(self):
        self.now = 0
        self.backend = FakeAgent()
        self.chat = ConversationAgent(self.backend, clock=lambda: self.now)

    def prime(self):
        return self.chat.ask("Explain the evolving personal map.")

    def test_follow_up_uses_public_topic_only_and_preserves_every_current_word(self):
        self.prime()
        question = "And how does that work?"
        reply = self.chat.ask(question)
        self.assertTrue(reply.memory_used)
        query = self.backend.calls[-1]
        self.assertIn("What is the evolving personal map?", query)
        self.assertTrue(query.endswith(question))
        self.assertNotIn(self.backend.reply.answer, query)
        self.assertIn("remember_confirmed_topic_ids", reply.plan)
        self.assertEqual(reply.model_calls, 3)  # No extra resolver model.

    def test_cold_and_expired_follow_ups_clarify_without_backend(self):
        self.assertEqual(self.chat.ask("And how does that work?").outcome, "clarify")
        self.assertEqual(self.backend.calls, [])
        self.prime()
        self.now = RETENTION_SECONDS
        reply = self.chat.ask("And how does that work?")
        self.assertEqual(reply.memory_state, "expired")
        self.assertFalse(reply.memory_used)
        self.assertEqual(reply.outcome, "clarify")
        self.assertEqual(len(self.backend.calls), 1)

    def test_reset_and_separate_instances_cannot_reuse_topic(self):
        self.prime()
        other = ConversationAgent(self.backend)
        self.assertEqual(other.ask("And how does that work?").outcome, "clarify")
        self.assertEqual(self.chat.ask(" /RESET ").outcome, "memory_reset")
        self.assertEqual(self.chat.ask("And how does that work?").outcome, "clarify")
        self.assertEqual(len(self.backend.calls), 1)

    def test_current_private_execution_human_and_retention_requests_bypass_memory(self):
        for query in ("Show me another student's report.", "Execute Python code: print('x')",
                      "I want to talk to a human.", "How long do you keep my child's data?"):
            self.prime()
            count = len(self.backend.calls)
            reply = self.chat.ask(query)
            self.assertFalse(reply.memory_used)
            self.assertEqual(len(self.backend.calls), count)
            self.assertEqual(reply.memory_after_ids, ())

    def test_retention_becomes_sensitive_after_topic_resolution(self):
        self.prime()
        reply = self.chat.ask("How long is that stored?")
        self.assertEqual(reply.outcome, "human_support")
        self.assertFalse(reply.memory_used)
        self.assertEqual(len(self.backend.calls), 1)
        self.assertEqual(reply.memory_after_ids, ())

    def test_explicit_question_bypasses_prior_topic_and_replaces_it(self):
        self.prime()
        self.backend.reply = ToolReply(answer="Thinking activities.", outcome="answered",
                                      source_ids=("octa_tutor_010",), selected_ids=("octa_tutor_010",),
                                      retrieved_ids=("octa_tutor_010",))
        reply = self.chat.ask("What kinda learning activities?")
        self.assertEqual(self.backend.calls[-1], "What kinda learning activities?")
        self.assertFalse(reply.memory_used)
        self.assertEqual(reply.memory_after_ids, ("octa_tutor_010",))

    def test_nonanswers_and_invalid_or_unconfirmed_citations_clear_topic(self):
        for reply in (ToolReply(answer="Help", outcome="human_support"),
                      ToolReply(answer="Error", outcome="service_error"),
                      ToolReply(answer="Recipe", outcome="out_of_scope"),
                      ToolReply(answer="Wrong", outcome="answered", source_ids=("unknown",)),
                      ToolReply(answer="Wrong", outcome="answered", source_ids=("octa_tutor_004",)),
                      ToolReply(answer="Wrong", outcome="answered", source_ids=("octa_tutor_018",),
                                selected_ids=("octa_tutor_018",), retrieved_ids=("octa_tutor_018",))):
            self.backend.reply = FakeAgent().reply
            self.prime()
            self.backend.reply = reply
            result = self.chat.ask("What is Octakidz?")
            self.assertEqual(result.memory_after_ids, ())
            self.assertEqual(self.chat.ask("And how does that work?").outcome, "clarify")

    def test_private_names_are_not_replayed_and_interests_answers_are_not_retained(self):
        question = "My son Ravi loves football. Email parent@example.invalid. Explain the personal map."
        reply = self.chat.ask(question)
        self.chat.ask("And how does that work?")
        query = self.backend.calls[-1]
        for secret in ("Ravi", "football", "parent@example.invalid", self.backend.reply.answer):
            self.assertNotIn(secret, query)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            write_event(reply, self.chat, path)
            text = path.read_text()
            for secret in (question, "Ravi", "football", "parent@example.invalid", self.backend.reply.answer,
                           "never-display-key"):
                self.assertNotIn(secret, text)
            self.assertEqual(json.loads(text)["version"], VERSION)
        self.assertEqual(self.chat.memory_snapshot()["source_ids"], ("octa_tutor_004",))

    def test_mixed_question_keeps_unsupported_part_and_then_clears_context(self):
        self.prime()
        self.backend.reply = ToolReply(answer="Team confirmation is needed.", outcome="human_support")
        question = "How does that work, and is school bus pickup included?"
        reply = self.chat.ask(question)
        self.assertTrue(self.backend.calls[-1].endswith(question))
        self.assertEqual(reply.memory_after_ids, ())

    def test_long_input_and_snapshot_copies_cannot_bypass_controls(self):
        self.prime()
        result = self.chat.ask("How does that " + "a" * 4000)
        self.assertEqual(result.outcome, "clarify")
        self.assertEqual(len(self.backend.calls), 1)
        self.prime()
        snapshot = self.chat.memory_snapshot()
        snapshot["source_ids"] = ("malicious",)
        self.assertEqual(self.chat.memory_snapshot()["source_ids"], ("octa_tutor_004",))

    def test_real_tool_gateway_checks_the_entire_contextual_query_once(self):
        backend = FakeSearch(ids=("octa_tutor_004",))
        router = FakeRouter([])
        checker = FakeRunnable(ContextDecision(action="supported", source_ids=["octa_tutor_004"],
                                               reason_code="direct_support"))
        answerer = FakeRunnable(ModelAnswer(action="answered", answer="The personal map can evolve.",
                                            source_ids=["octa_tutor_004"]))
        tool_agent = ContextToolAgent(Settings(api_key="offline"), router, checker, answerer,
                                     lambda: SupportToolSession(backend))
        chat = ConversationAgent(tool_agent)
        self.assertEqual(tool_agent.instructions, Path("prompts/tool_router.md").read_text())
        for question in ("What is the evolving personal map?", "And how does that work?"):
            query = (chat.contextual_question(question, ("octa_tutor_004",))
                     if chat.memory_snapshot()["source_ids"] else question)
            router.proposals = [{"name": "search_support_knowledge", "args": {"query": query}}]
            reply = chat.ask(question)
            self.assertEqual(reply.outcome, "answered")
            self.assertEqual(reply.model_calls, 3)
            self.assertEqual(reply.tool_attempts, 1)
            self.assertEqual(backend.calls[-1], query)
            self.assertEqual(checker.calls[-1][1].content, query)
            self.assertEqual(answerer.calls[-1][1].content, query)
        self.assertTrue(reply.memory_used)
        self.assertEqual(len(backend.calls), 2)

    def test_router_binds_both_ordinary_and_contextual_questions(self):
        model = FakeRouter([])
        with patch("conversation_agent.make_router", return_value=model) as factory:
            router = ContextRouter(Settings(api_key="offline"))
            router.invoke([HumanMessage(content="What is Octakidz?")])
            self.assertEqual(factory.call_count, 1)
            self.assertEqual(factory.call_args.kwargs, {"query_constraint": "What is Octakidz?"})
            query = self.chat.contextual_question("And how does that work?", ("octa_tutor_004",))
            router.invoke([HumanMessage(content=query)])
            self.assertEqual(factory.call_args.kwargs, {"query_constraint": query})
            self.assertEqual(factory.call_count, 2)
            self.assertEqual(len(model.calls), 2)

    def test_named_suitability_question_is_redacted_and_complete_at_gateway(self):
        backend = FakeSearch(ids=("octa_tutor_002",))
        question = "my name is arun, I'm asking for my kid leela, shes is 12yrs , will octakidz work for her"
        expected = "[NAME REMOVED], I'm asking for my child [NAME REMOVED], shes is 12yrs , will octakidz work for her"
        model = FakeRouter([{"name": "search_support_knowledge", "args": {"query": expected}}])
        checker = FakeRunnable(ContextDecision(action="supported", source_ids=["octa_tutor_002"], reason_code="direct_support"))
        answerer = FakeRunnable(ModelAnswer(action="answered", answer="Age 12 is within the 10–14 age range.",
                                            source_ids=["octa_tutor_002"]))
        with patch("conversation_agent.make_router", return_value=model) as factory:
            tool_agent = ContextToolAgent(Settings(api_key="offline"), checker=checker, answerer=answerer,
                                         session_factory=lambda: SupportToolSession(backend))
            chat = ConversationAgent(tool_agent)
            reply = chat.ask(question)
        self.assertEqual(reply.outcome, "answered")
        self.assertIsNone(reply.error_code)
        self.assertEqual(factory.call_args.kwargs, {"query_constraint": expected})
        self.assertEqual(backend.calls, [expected])
        self.assertEqual(checker.calls[0][1].content, expected)
        self.assertEqual(answerer.calls[0][1].content, expected)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            write_event(reply, chat, path)
            text = path.read_text()
            for private in ("arun", "leela", question, expected, reply.answer):
                self.assertNotIn(private, text)

    def test_strict_schema_and_gateway_both_keep_query_binding(self):
        with patch("tool_agent.ChatOpenAI") as sdk:
            make_router(Settings(api_key="offline"), query_constraint="Complete current request")
            functions = sdk.return_value.bind_tools.call_args.args[0]
            search = next(t["function"] for t in functions if t["function"]["name"] == "search_support_knowledge")
            self.assertEqual(search["parameters"]["properties"]["query"]["enum"], ["Complete current request"])
            self.assertTrue(sdk.return_value.bind_tools.call_args.kwargs["strict"])
            make_router(Settings(api_key="offline"))
            plain = sdk.return_value.bind_tools.call_args.args[0][0]["function"]
            self.assertNotIn("enum", plain["parameters"]["properties"]["query"])


if __name__ == "__main__":
    unittest.main()
