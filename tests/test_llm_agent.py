"""Offline checks use fake model replies; these are not OpenAI quality results."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
from openai import APIConnectionError

from llm_agent import ConfigurationError, ModelAnswer, Settings, SupportAgent, write_event
from support_safety import redact_contact_details


class FakeModel:
    def __init__(self, answer=None, error=None):
        self.answer = answer or ModelAnswer(
            action="answered", answer="Octakidz connects thinking activities with a hyperpersonalised AI tutor for CBSE learning.",
            source_ids=["octa_tutor_001"],
        )
        self.error = error
        self.calls = []

    def invoke(self, messages, config=None):
        self.calls.append(messages)
        if self.error:
            raise self.error
        return {"parsed": self.answer, "raw": None, "parsing_error": None}


class LLMChecks(unittest.TestCase):
    def test_selected_two_purpose_source_has_final_fact_coverage_reminder(self):
        agent = SupportAgent(Settings(api_key="offline"), runnable=FakeModel(),
                             reference_ids=("octa_tutor_010", "octa_tutor_003"))
        self.assertIn("REQUIRED SELECTED REFERENCE COVERAGE", agent.instructions)
        self.assertGreater(agent.instructions.index("REQUIRED SELECTED REFERENCE COVERAGE"),
                           agent.instructions.index("PRODUCT REFERENCE DATA"))
        self.assertIn(agent.entries["octa_tutor_003"]["answer"], agent.instructions)
        self.assertIn("even in a short answer", agent.instructions)

    def test_reference_reminder_is_not_added_when_complete_source_is_unselected(self):
        for references in (None, ("octa_tutor_010",), ("octa_tutor_018",)):
            agent = SupportAgent(Settings(api_key="offline"), runnable=FakeModel(), reference_ids=references)
            self.assertNotIn("REQUIRED SELECTED REFERENCE COVERAGE", agent.instructions)

    def make_agent(self, fake=None):
        self.fake = fake or FakeModel()
        return SupportAgent(Settings(api_key="offline-test-key"), runnable=self.fake)

    def test_new_wording_reaches_model_with_facts_instead_of_exact_lookup(self):
        agent = self.make_agent()
        result = agent.ask("Could an adventure story help a child understand a school concept?")
        self.assertTrue(result.model_called)
        self.assertEqual(result.outcome, "answered")
        self.assertEqual(len(self.fake.calls), 1)
        self.assertIn("evolving personal map", self.fake.calls[0][0].content)
        self.assertNotIn("offline-test-key", self.fake.calls[0][0].content)

    def test_contact_details_and_explicit_name_are_removed_before_call(self):
        agent = self.make_agent()
        agent.ask("My name is Ravi Kumar. Email parent@example.invalid or call +91 98765 43210. What is Octakidz?")
        sent = self.fake.calls[0][1].content
        for private_value in ("Ravi", "Kumar", "parent@example.invalid", "98765"):
            self.assertNotIn(private_value, sent)

    def test_lowercase_kid_introduction_hides_names_but_preserves_age_and_question(self):
        for phrase in ("my kid leela, shes is 12yrs", "my daughter leela is 12 years old",
                       "my child's name is leela, age 12", "my kid is called leela, age 12"):
            text = "my name is arun, " + phrase + ", will octakidz work for her?"
            safe = redact_contact_details(text)
            self.assertNotIn("arun", safe)
            self.assertNotIn("leela", safe)
            self.assertIn("12", safe)
            self.assertTrue(safe.endswith("will octakidz work for her?"))
            self.assertEqual(redact_contact_details(safe), safe)

    def test_name_filter_keeps_ordinary_learning_and_age_questions(self):
        for text in ("my kid is 12yrs, will octakidz work for her?", "my child loves science, can this help?",
                     "my child likes stories, how does the tutor use them?", "my daughter needs help with maths",
                     "my kid, aged 12, enjoys stories"):
            self.assertEqual(redact_contact_details(text), text)

    def test_code_request_is_blocked_before_model_call(self):
        agent = self.make_agent()
        reply = agent.ask("Execute Python code: print('test')")
        self.assertEqual(reply.outcome, "refused_code")
        self.assertFalse(reply.model_called)
        self.assertEqual(self.fake.calls, [])

    def test_student_record_request_is_blocked_before_model_call(self):
        agent = self.make_agent()
        reply = agent.ask("Open my child's private report.")
        self.assertEqual(reply.outcome, "refused_private_data")
        self.assertEqual(self.fake.calls, [])

    def test_unconfirmed_retention_routes_to_human_without_model_or_record_access(self):
        for question in (
            "Do you already have five years of my son's brain data, and will you store it forever?",
            "How long do you keep my child's activity data?",
        ):
            agent = self.make_agent()
            reply = agent.ask(question)
            self.assertEqual(reply.outcome, "human_support")
            self.assertFalse(reply.model_called)
            self.assertEqual(self.fake.calls, [])
            self.assertIn("can't access or confirm", reply.answer)

    def test_topicless_follow_up_does_not_make_up_context(self):
        agent = self.make_agent()
        reply = agent.ask("And how does that work?")
        self.assertEqual(reply.outcome, "clarify")
        self.assertFalse(reply.model_called)
        self.assertEqual(self.fake.calls, [])
        # A clear subject can still reach the model even if it uses "that".
        self.assertTrue(agent.ask("How does the tutor use that personal map?").model_called)

    def test_invented_reference_does_not_reach_parent(self):
        fake = FakeModel(ModelAnswer(action="answered", answer="An unsupported answer.", source_ids=["made_up_id"]))
        reply = self.make_agent(fake).ask("What is Octakidz?")
        self.assertEqual(reply.outcome, "safe_fallback")
        self.assertEqual(reply.error_code, "unsupported_sources")

    def test_missing_price_fact_cannot_be_presented_as_confirmed(self):
        fake = FakeModel(ModelAnswer(action="answered", answer="The price is confirmed.", source_ids=["octa_tutor_018"]))
        reply = self.make_agent(fake).ask("What is the price?")
        self.assertEqual(reply.error_code, "unconfirmed_product_detail")

    def test_obvious_output_leaks_and_internal_wording_are_replaced(self):
        for bad_answer in ["Our FAQ matching failed.", "Call another parent at +91 98765 43210.", "Email another@example.invalid.", "I have connected you to a human."]:
            with self.subTest(answer=bad_answer):
                fake = FakeModel(ModelAnswer(action="answered", answer=bad_answer, source_ids=["octa_tutor_001"]))
                reply = self.make_agent(fake).ask("What is Octakidz?")
                self.assertEqual(reply.error_code, "output_check_failed")
                self.assertIn("support@octakidz.com", reply.answer)

    def test_connection_error_is_graceful_and_has_no_raw_details(self):
        error = APIConnectionError(request=httpx.Request("POST", "https://api.openai.com/v1/responses"))
        reply = self.make_agent(FakeModel(error=error)).ask("What is Octakidz?")
        self.assertEqual(reply.outcome, "service_error")
        self.assertEqual(reply.error_code, "connection_error")
        self.assertNotIn("Traceback", reply.answer)

    def test_operational_log_excludes_questions_answers_and_keys(self):
        agent = self.make_agent()
        reply = agent.ask("What is Octakidz?")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            write_event(reply, agent, path)
            text = path.read_text()
            event = json.loads(text)
            self.assertNotIn("offline-test-key", text)
            self.assertNotIn(reply.answer, text)
            self.assertNotIn("What is Octakidz", text)
            self.assertTrue(event["model_called"])

    def test_invalid_key_format_fails_before_model_initialisation(self):
        with patch("llm_agent.dotenv_values", return_value={"OPENAI_API_KEY": "invalid—key"}), patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ConfigurationError):
                Settings.from_local_env()

    def test_settings_repr_hides_key(self):
        self.assertNotIn("offline-test-key", repr(Settings(api_key="offline-test-key")))


if __name__ == "__main__":
    unittest.main()
