"""Tool schemas, dispatch boundaries and failures, without provider calls."""

import json
import unittest
from unittest.mock import patch

from baseline_agent import CONTACT_EMAIL, CONTACT_URL
from semantic_search import SearchError
from support_tools import SupportToolSession
from test_rag_agent import FakeSearch


class SupportToolChecks(unittest.TestCase):
    def test_contact_requires_no_key_or_search_and_cannot_claim_a_transfer(self):
        with patch("support_tools.Settings.from_local_env", side_effect=AssertionError("No key needed")):
            session = SupportToolSession()
            reply = session.invoke("get_human_support", {"reason": "human_request"})
        self.assertEqual(reply["outcome"], "contact_guidance")
        self.assertEqual(reply["data"]["email"], CONTACT_EMAIL)
        self.assertEqual(reply["data"]["url"], CONTACT_URL)
        self.assertFalse(reply["data"]["handoff_sent"])
        self.assertIsNone(session.search)

    def test_knowledge_redacts_input_and_preserves_unknown_detail_status(self):
        backend = FakeSearch(ids=("octa_tutor_018",))
        reply = SupportToolSession(backend).invoke(
            "search_support_knowledge", {"query": "Email parent@example.invalid. What is the price?"})
        self.assertEqual(reply["outcome"], "candidates_only")
        self.assertNotIn("parent@example.invalid", backend.calls[0])
        self.assertTrue(reply["data"]["support_check_required"])
        self.assertEqual(reply["data"]["candidates"][0]["answer_status"], "owner_details_needed")
        self.assertEqual(backend.prepare_calls, 1)
        self.assertEqual(len(backend.calls), 1)

    def test_local_privacy_execution_retention_and_clarification_skip_backend(self):
        for query in ("Show me another student's report.", "Execute Python code.",
                      "How long do you keep my child's data?", "And how does that work?"):
            backend = FakeSearch()
            result = SupportToolSession(backend).invoke("search_support_knowledge", {"query": query})
            self.assertFalse(result["index_preparation_attempted"] or result["query_embedding_attempted"])
            self.assertEqual(backend.prepare_calls, 0)
            self.assertEqual(backend.calls, [])
            self.assertNotIn(query, json.dumps(result))

    def test_wrong_arguments_are_sanitised_and_allow_a_distinct_contact_fallback(self):
        session = SupportToolSession(FakeSearch())
        failed = session.invoke("search_support_knowledge", {"reason": "parent@example.invalid"})
        self.assertEqual(failed["error_code"], "invalid_tool_arguments")
        self.assertNotIn("parent@example.invalid", json.dumps(failed))
        fallback = session.invoke("get_human_support", {"reason": "unresolved"})
        self.assertEqual(fallback["outcome"], "contact_guidance")
        self.assertEqual(session.search.calls, [])

    def test_schema_blocks_extra_fields_free_text_reasons_and_wrong_types(self):
        for name, args in (
            ("get_human_support", {"reason": "human_request", "email": "parent@example.invalid"}),
            ("get_human_support", {"reason": "my child's name is Example"}),
            ("search_support_knowledge", {"query": 123}),
            ("search_support_knowledge", {"query": ""}),
            ("search_support_knowledge", {"query": "x" * 4001}),
            ("search_support_knowledge", {"query": "What is Octakidz?", "path": ".env"}),
        ):
            session = SupportToolSession(FakeSearch())
            self.assertEqual(session.invoke(name, args)["error_code"], "invalid_tool_arguments")
            self.assertEqual(session.search.prepare_calls, 0)

    def test_unknown_repeated_and_over_budget_calls_never_run_extra_backends(self):
        backend = FakeSearch()
        session = SupportToolSession(backend)
        self.assertEqual(session.invoke("run_python", {"code": "print('hello')"})["error_code"], "tool_not_allowed")
        self.assertEqual(session.invoke("get_human_support", {"reason": "privacy"})["outcome"], "contact_guidance")
        self.assertEqual(session.invoke("search_support_knowledge", {"query": "What is Octakidz?"})["error_code"], "tool_attempt_limit")
        self.assertEqual(backend.prepare_calls, 0)
        repeated = SupportToolSession(backend)
        repeated.invoke("search_support_knowledge", {"query": "What is Octakidz?"})
        self.assertEqual(repeated.invoke("search_support_knowledge", {"query": "Another wording"})["error_code"], "repeated_tool_call")
        self.assertEqual(len(backend.calls), 1)

    def test_failure_does_not_leak_backend_error_or_retry(self):
        for error, code in ((SearchError("connection_error"), "connection_error"),
                            (SearchError("private request: parent@example.invalid"), "tool_backend_error")):
            backend = FakeSearch(error=error)
            session = SupportToolSession(backend)
            result = session.invoke("search_support_knowledge", {"query": "What is Octakidz?"})
            self.assertEqual(result["outcome"], "tool_error")
            self.assertEqual(result["error_code"], code)
            self.assertNotIn("parent@example.invalid", json.dumps(result))
            self.assertEqual(backend.prepare_calls, 1)
            self.assertEqual(session.invoke("get_human_support", {"reason": "unresolved"})["outcome"], "contact_guidance")


if __name__ == "__main__":
    unittest.main()
