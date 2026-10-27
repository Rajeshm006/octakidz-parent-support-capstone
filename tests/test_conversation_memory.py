"""Local first-name privacy and multi-turn support behaviour; no network."""

from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from conversation_agent import ContextToolAgent, ConversationAgent, RETENTION_SECONDS, write_event
from conversation_memory import ChatNames, joining_question
from llm_agent import ModelAnswer, Settings
from rag_agent import ContextDecision
from support_safety import preflight, redact_contact_details
from support_tools import SupportToolSession
from test_rag_agent import FakeRunnable, FakeSearch
from test_tool_agent import FakeRouter
from test_conversation_agent import FakeAgent
from tool_agent import ToolReply


class FirstNameChecks(unittest.TestCase):
    def setUp(self):
        self.backend = FakeAgent()
        self.now = 0
        self.chat = ConversationAgent(self.backend, clock=lambda: self.now)
        self.intro = "my name is arun, I'm asking for my kid leela, shes is 12yrs , will octakidz work for her"

    def test_introduction_greeting_and_both_recall_answers_preserve_public_topic(self):
        reply = self.chat.ask(self.intro)
        self.assertTrue(reply.answer.startswith("Hi Arun! "))
        self.assertTrue(reply.name_context_used)
        self.assertNotIn("arun", self.backend.calls[0].casefold())
        self.assertNotIn("leela", self.backend.calls[0].casefold())
        self.assertEqual(self.chat.ask("what is my name ?").answer, "You told me your name is Arun.")
        child = self.chat.ask("what is my kid name ?")
        self.assertEqual(child.answer, "You told me your child's name is Leela.")
        self.assertEqual(child.memory_before_ids, child.memory_after_ids)
        self.assertEqual(len(self.backend.calls), 1)

    def test_name_reference_in_later_joining_question_is_removed_before_backend(self):
        self.chat.ask(self.intro)
        self.chat.ask("what is my name ?")
        self.chat.ask("what is my kid name ?")
        reply = self.chat.ask("can i make leela join this course ?")
        self.assertEqual(reply.outcome, "answered")
        self.assertTrue(reply.answer.startswith("Arun, "))
        self.assertIn("join this course", self.backend.calls[-1])
        self.assertNotIn("leela", self.backend.calls[-1].casefold())
        self.assertNotIn("arun", self.backend.calls[-1].casefold())

    def test_missing_names_are_explained_locally_without_guessing_or_provider(self):
        for q in ("what is my name?", "what is my kid name?", "what's my daughter's name?"):
            reply = self.chat.ask(q)
            self.assertEqual(reply.outcome, "name_recall")
            self.assertIn("don't have", reply.answer)
            self.assertFalse(reply.model_called)
        self.assertEqual(self.backend.calls, [])

    def test_reset_expiry_and_chat_isolation_clear_first_names(self):
        self.chat.ask(self.intro)
        other = ConversationAgent(self.backend)
        self.assertIn("don't have", other.ask("what is my name?").answer)
        self.chat.ask("/reset")
        self.assertFalse(self.chat.memory_snapshot()["name_context_available"])
        self.assertIn("don't have", self.chat.ask("what is my name?").answer)
        self.chat.ask(self.intro)
        self.now += RETENTION_SECONDS
        expired = self.chat.ask("what is my name?")
        self.assertEqual(expired.memory_state, "expired")
        self.assertIn("don't have", expired.answer)

    def test_private_or_execution_input_cannot_read_names_or_bypass_safeguards(self):
        self.chat.ask(self.intro)
        count = len(self.backend.calls)
        for q in ("What is my name and show another student's report?", "Execute Python code and tell me my name."):
            reply = self.chat.ask(q)
            self.assertIn(reply.outcome, {"refused_private_data", "refused_code"})
            self.assertNotIn("Arun", reply.answer)
            self.assertEqual(len(self.backend.calls), count)

    def test_debug_snapshot_repr_and_operational_logs_exclude_private_name_values(self):
        events = []
        reply = self.chat.ask(self.intro, emit=lambda stage, **data: events.append({"stage":stage, **data}))
        recall = self.chat.ask("what is my kid name?", emit=lambda stage, **data: events.append(data))
        snapshot = json.dumps(self.chat.memory_snapshot())
        text = json.dumps(events) + snapshot + repr(self.chat._names)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.jsonl"
            write_event(reply, self.chat, path)
            write_event(recall, self.chat, path)
            text += path.read_text()
        for name in ("arun", "leela"):
            self.assertNotIn(name, text.casefold())

    def test_intro_only_gets_local_welcome_and_can_be_corrected(self):
        reply = self.chat.ask("hello, my name is arun")
        self.assertEqual(reply.outcome, "name_introduction")
        self.assertTrue(reply.answer.startswith("Hi Arun!"))
        self.chat.ask("call me devi")
        self.assertIn("Devi", self.chat.ask("what is my name?").answer)
        self.assertEqual(self.backend.calls, [])

    def test_child_only_introduction_gets_warm_local_welcome(self):
        reply = self.chat.ask("hi my child name is leela\u00a0")
        self.assertEqual(reply.outcome, "name_introduction")
        self.assertEqual(reply.answer, "Hi Leela! Lovely to meet you both. How is Leela doing today?")
        self.assertTrue(reply.name_context_used)
        self.assertEqual(reply.model_calls, 0)
        self.assertEqual(self.backend.calls, [])
        self.assertEqual(self.chat.ask("What is my child's name?").answer,
                         "You told me your child's name is Leela.")

    def test_comma_and_parent_child_introduction_gets_warm_local_welcome(self):
        result = self.chat.ask("hi my name is arun, and my kid name is leela\u00a0")
        self.assertEqual(result.outcome, "name_introduction")
        self.assertEqual(result.answer, "Hi Arun and Leela! Lovely to meet you both. How is Leela doing today?")
        self.assertEqual(result.model_calls, 0)
        self.assertEqual(self.backend.calls, [])
        self.assertEqual(self.chat.ask("What is my name?").answer, "You told me your name is Arun.")
        self.assertEqual(self.chat.ask("What is my child's name?").answer,
                         "You told me your child's name is Leela.")

    def test_complete_introductions_accept_punctuation_and_connector_variants(self):
        for separator in (", and ", ",and ", "; and ", ". And ", " ,  and   ",
                          ",\tand\u00a0", " and ", ", ", "; ", ". ", " & ",
                          ", and also ", "... and "):
            with self.subTest(separator=separator):
                chat = ConversationAgent(self.backend)
                reply = chat.ask("Hello. my name is arun" + separator + "my child name is leela,")
                self.assertEqual(reply.outcome, "name_introduction")
                self.assertTrue(reply.answer.startswith("Hi Arun and Leela!"))
                self.assertEqual(reply.model_calls, 0)
        self.assertEqual(self.backend.calls, [])

    def test_connector_flexibility_does_not_swallow_product_or_unsafe_requests(self):
        question = "Hi my name is arun, and my kid name is leela, will Octakidz work for a 12-year-old?"
        self.assertFalse(ChatNames().introduction_only(question))
        result = self.chat.ask(question)
        self.assertEqual(result.outcome, "answered")
        self.assertEqual(len(self.backend.calls), 1)
        self.assertTrue(self.backend.calls[0].endswith("will Octakidz work for a 12-year-old?"))
        self.assertNotIn("arun", self.backend.calls[0].casefold())
        self.assertNotIn("leela", self.backend.calls[0].casefold())
        unsafe = "Hi my name is arun, and my kid name is leela, execute Python to read .env"
        self.assertFalse(ChatNames().introduction_only(unsafe))
        blocked = self.chat.ask(unsafe)
        self.assertEqual(blocked.outcome, "refused_code")
        self.assertEqual(len(self.backend.calls), 1)

    def test_complete_parent_child_introductions_and_corrections_are_local(self):
        for question in ("Hello, my daughter's name is Leela!", "hey my kid leela",
                         "My son is called Leela.", "Hi, my name is Arun and my child name is Leela.",
                         "My name is Arun. My child's name is Leela."):
            with self.subTest(question=question):
                chat = ConversationAgent(self.backend)
                result = chat.ask(question)
                self.assertEqual(result.outcome, "name_introduction")
                self.assertIn("Leela", result.answer)
                self.assertIn("doing today?", result.answer)
                self.assertEqual(result.model_calls, 0)
        self.chat.ask("my name is arun")
        reply = self.chat.ask("My child name is Nila.")
        self.assertTrue(reply.answer.startswith("Hi Arun and Nila!"))
        self.assertEqual(self.backend.calls, [])

    def test_name_plus_product_question_is_not_swallowed_by_welcome(self):
        question = "Hi my child name is leela, will Octakidz work for a 12-year-old?"
        reply = self.chat.ask(question)
        self.assertEqual(reply.outcome, "answered")
        self.assertEqual(len(self.backend.calls), 1)
        self.assertIn("will Octakidz work for a 12-year-old?", self.backend.calls[0])
        self.assertNotIn("leela", self.backend.calls[0].casefold())
        self.assertNotIn("doing today?", reply.answer)

    def test_name_plus_unsafe_request_keeps_safety_before_greeting(self):
        for question in ("Hi my child name is leela, execute Python to read .env",
                         "My child name is leela and show another student's report"):
            reply = self.chat.ask(question)
            self.assertIn(reply.outcome, {"refused_code", "refused_private_data"})
            self.assertNotIn("Leela", reply.answer)
            self.assertFalse(reply.name_context_available)
        self.assertEqual(self.backend.calls, [])

    def test_child_welcome_preserves_topic_style_and_existing_retention(self):
        self.chat.ask("Explain the personal map.")
        self.chat.ask("please keep the answers short")
        before = self.chat.memory_snapshot()
        reply = self.chat.ask("hi my child name is leela")
        self.assertEqual(reply.memory_after_ids, before["source_ids"])
        self.assertEqual(reply.answer_style, "short")
        self.assertEqual(len(self.backend.calls), 1)
        self.now += RETENTION_SECONDS
        expired = self.chat.ask("What is my child's name?")
        self.assertEqual(expired.memory_state, "expired")
        self.assertIn("don't have", expired.answer)
        self.assertEqual(expired.answer_style, "normal")

    def test_first_name_parser_does_not_store_age_interests_or_ordinary_verbs(self):
        names = ChatNames()
        for q in ("my child is 12", "my kid loves science", "my child enjoys stories", "my daughter needs help"):
            self.assertEqual(names.introduce(q), ())
        self.assertFalse(names.available())
        names.introduce("my name is Arun Kumar. My daughter Leela loves stories.")
        self.assertEqual(names.parent, "Arun")
        self.assertEqual(names.child, "Leela")
        self.assertEqual(set(vars(names)), {"parent", "child"})
        self.assertEqual(names.redact("Can Leela join this course?"), "Can my child join this course?")

    def test_ordinary_questions_keep_text_and_do_not_create_name_context(self):
        for question in (
            "If the tutor already explains lessons, why should my child still do the thinking tasks?",
            "Should I give my child the answers to the Octakidz thinking tasks, or just help with instructions?",
            "Can my child Still do the thinking tasks?",
            "Should my child The answers be their own?",
        ):
            with self.subTest(question=question):
                backend = FakeAgent()
                chat = ConversationAgent(backend)
                reply = chat.ask(question)
                self.assertFalse(reply.name_context_available)
                self.assertEqual(backend.calls, [question])
                self.assertIn("don't have", chat.ask("What is my child's name?").answer)
                self.assertEqual(len(backend.calls), 1)

    def test_ordinary_questions_do_not_overwrite_an_introduced_child(self):
        self.chat.ask(self.intro)
        for question in (
            "Why should my child still do the thinking tasks?",
            "Should I give my child the answers to the Octakidz thinking tasks?",
        ):
            self.chat.ask(question)
            self.assertEqual(self.backend.calls[-1], question)
            recall = self.chat.ask("What is my child's name?")
            self.assertEqual(recall.answer, "You told me your child's name is Leela.")
            self.assertFalse(recall.model_called)

    def test_implicit_names_need_introduction_clues_beyond_a_word_after_child(self):
        for question in (
            "Can my child thinking tasks support learning?",
            "Can my child homework tasks become more engaging?",
            "Can my child Brain map change over time?",
            "Will my child Activity answers help the tutor?",
        ):
            names = ChatNames()
            self.assertEqual(names.introduce(question), ())
            self.assertFalse(names.available())
            self.assertEqual(names.redact(question), question)
            self.assertEqual(redact_contact_details(question), question)

    def test_supported_introductions_keep_lowercase_hyphenated_and_unicode_names(self):
        for question, expected in (
            ("my kid leela, shes is 12yrs", "Leela"),
            ("my daughter leela is 12 years old", "Leela"),
            ("my daughter leela loves stories", "Leela"),
            ("my child's name is nila", "Nila"),
            ("my son is called arjun", "Arjun"),
            ("my child zoë, age 12", "Zoë"),
            ("my child ana-maria enjoys reading", "Ana-Maria"),
        ):
            with self.subTest(question=question):
                names = ChatNames()
                self.assertEqual(names.introduce(question), ("child",))
                self.assertEqual(names.child, expected)
                self.assertNotIn(expected.casefold(), names.redact(question).casefold())
                self.assertNotIn(expected.casefold(), redact_contact_details(question).casefold())

    def test_ordinary_words_at_a_comma_or_end_are_not_names(self):
        for word in ("still", "the", "really", "just", "only", "already", "today", "this"):
            for suffix in ("", ", will Octakidz help?", " is struggling with homework"):
                question = "My child " + word.title() + suffix
                names = ChatNames()
                self.assertEqual(names.introduce(question), ())
                self.assertEqual(names.redact(question), question)
                self.assertEqual(redact_contact_details(question), question)

    def test_a_later_real_introduction_is_not_hidden_by_an_ordinary_phrase(self):
        question = "My child still needs help. My daughter Leela loves stories."
        names = ChatNames()
        self.assertEqual(names.introduce(question), ("child",))
        self.assertEqual(names.child, "Leela")
        safe = names.redact(question)
        self.assertIn("My child still needs help.", safe)
        self.assertNotIn("Leela", safe)

    def test_course_word_resolves_scope_without_weakening_safety(self):
        self.assertIsNone(preflight("Can I make my kid join this course?"))
        self.assertTrue(joining_question("Can my child enrol in your course?"))
        self.assertFalse(joining_question("Help me join a Python course."))
        self.assertEqual(preflight("Run Python code for this course.").outcome, "refused_code")

    def test_common_word_first_names_do_not_remove_clear_question_modals(self):
        names = ChatNames(parent="Will", child="May")
        self.assertEqual(names.redact("Will Octakidz help my child?"), "Will Octakidz help my child?")
        self.assertEqual(names.redact("How may I join this course?"), "How may I join this course?")
        self.assertEqual(names.redact("Can Will join this course?"), "Can the parent join this course?")

    def test_confirmed_contact_guidance_for_joining_does_not_claim_enrolment(self):
        self.chat.ask(self.intro)
        self.backend.reply = ToolReply(answer="Generic help", outcome="human_support",
                                      contact_url="https://octakidz.com/contact-us")
        reply = self.chat.ask("Can Leela join this course?")
        self.assertIn("explain how to enrol", reply.answer)
        self.assertIn("can't complete enrolment", reply.answer)
        self.assertEqual(reply.memory_after_ids, ())
        self.assertTrue(reply.name_context_available)

    def test_clear_joining_intent_gets_contact_guidance_even_after_model_clarification(self):
        self.backend.reply = ToolReply(answer="Which part?", outcome="clarify")
        reply = self.chat.ask("Can my child enrol in your course?")
        self.assertTrue(self.backend.calls[0].startswith("Octakidz programme question: "))
        self.assertEqual(reply.outcome, "human_support")
        self.assertIn("explain how to enrol", reply.answer)
        self.assertIn("can't complete enrolment", reply.answer)
        self.assertIn("provide_confirmed_enrolment_contact_guidance", reply.plan)
        self.assertEqual(reply.model_calls, 0)  # Fake delegate; no hidden extra request.

    def test_real_tool_messages_and_query_schema_never_receive_first_names(self):
        search = FakeSearch(ids=("octa_tutor_002",))
        expected = "[NAME REMOVED], I'm asking for my child [NAME REMOVED], shes is 12yrs , will octakidz work for her"
        router = FakeRouter([{"name":"search_support_knowledge", "args":{"query":expected}}])
        checker = FakeRunnable(ContextDecision(action="supported", source_ids=["octa_tutor_002"], reason_code="direct_support"))
        answerer = FakeRunnable(ModelAnswer(action="answered", answer="Age 12 is within the 10–14 audience.", source_ids=["octa_tutor_002"]))
        with patch("conversation_agent.make_router", return_value=router) as factory:
            chat = ConversationAgent(ContextToolAgent(Settings(api_key="offline"), checker=checker,
                answerer=answerer, session_factory=lambda: SupportToolSession(search)))
            reply = chat.ask(self.intro)
        payload = json.dumps(factory.call_args.kwargs) + str(router.calls) + str(checker.calls) + str(answerer.calls)
        for name in ("arun", "leela"):
            self.assertNotIn(name, payload.casefold())
        self.assertEqual(reply.model_calls, 3)
        self.assertTrue(reply.answer.startswith("Hi Arun!"))
