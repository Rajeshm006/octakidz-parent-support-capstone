"""Offline flow/boundary tests; live retrieval quality is evaluated separately."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import httpx
from openai import APIConnectionError

from baseline_agent import load_faq
from llm_agent import ModelAnswer, Settings
from rag_agent import ContextDecision, RAGAgent, write_event
from semantic_search import SearchError, SearchResult


class FakeSearch:
    def __init__(self, ids=("octa_tutor_001", "octa_tutor_002"), error=None):
        entries = {item["id"]:item for item in load_faq().values()}
        self.candidates = [{**entries[source], "score":0.6} for source in ids]
        self.prepare_calls = 0
        self.calls = []
        self.error = error

    def prepare(self, rebuild=False):
        self.prepare_calls += 1
        if self.error:
            raise self.error
        return {"model_calls":0, "input_tokens":0}

    def search(self, question, top_k):
        self.calls.append(question)
        return SearchResult("candidates_only" if self.candidates else "insufficient_relevance", self.candidates,
                            model_called=True, input_tokens=5, nearest_score=0.6 if self.candidates else 0.1)


class FakeRunnable:
    def __init__(self, parsed, error=None):
        self.parsed = parsed
        self.error = error
        self.calls = []

    def invoke(self, messages, config=None):
        self.calls.append(messages)
        if self.error:
            raise self.error
        return {"parsed":self.parsed, "raw":SimpleNamespace(usage_metadata={"input_tokens":3,"output_tokens":1}), "parsing_error":None}


class RAGChecks(unittest.TestCase):
    def make_agent(self, decision=None, answer=None, ids=("octa_tutor_001", "octa_tutor_002"), search_error=None, check_error=None):
        self.search = FakeSearch(ids, error=search_error)
        self.checker = FakeRunnable(decision or ContextDecision(action="supported", source_ids=["octa_tutor_001"], reason_code="direct_support"), error=check_error)
        self.answerer = FakeRunnable(answer or ModelAnswer(action="answered", answer="Octakidz combines thinking activities with a hyperpersonalised AI tutor.", source_ids=["octa_tutor_001"]))
        return RAGAgent(Settings(api_key="offline-test-key"),self.search,self.checker,self.answerer)

    def test_selected_sources_limit_both_generation_context_and_citations(self):
        result = self.make_agent().ask("What is Octakidz?")
        self.assertEqual(result.outcome,"answered")
        self.assertEqual(result.retrieved_ids,("octa_tutor_001","octa_tutor_002"))
        self.assertEqual(result.selected_ids,("octa_tutor_001",))
        system = self.answerer.calls[0][0].content
        self.assertIn('"id": "octa_tutor_001"',system)
        self.assertNotIn('"id": "octa_tutor_002"',system)
        self.assertNotIn('"id": "octa_tutor_030"',self.checker.calls[0][0].content)
        self.assertTrue(result.checker_called and result.answerer_called)
        self.assertEqual(result.input_tokens,6)
        self.assertEqual(result.embedding_input_tokens,5)

    def test_strong_but_unsupported_context_stops_before_generation(self):
        decision = ContextDecision(action="human_support",source_ids=[],reason_code="missing_information")
        result = self.make_agent(decision=decision,ids=("octa_tutor_014","octa_tutor_002")).ask("Can Octakidz arrange school bus pickup?")
        self.assertEqual(result.outcome,"human_support")
        self.assertFalse(result.answerer_called)
        self.assertEqual(self.answerer.calls,[])
        self.assertNotIn("bus pickup is available",result.answer)

    def test_partial_multi_question_context_does_not_authorise_generation(self):
        decision = ContextDecision(action="human_support",source_ids=["octa_tutor_001"],reason_code="partial_support")
        result = self.make_agent(decision=decision).ask("Explain Octakidz and confirm bus pickup.")
        self.assertEqual(result.decision_reason,"partial_support")
        self.assertFalse(result.answerer_called)

    def test_unknown_price_cannot_be_approved_even_if_checker_makes_a_mistake(self):
        decision = ContextDecision(action="supported",source_ids=["octa_tutor_018"],reason_code="direct_support")
        result = self.make_agent(decision=decision,ids=("octa_tutor_018",)).ask("What is the exact price?")
        self.assertEqual(result.outcome,"human_support")
        self.assertEqual(result.error_code,"unknown_detail_blocked")
        self.assertEqual(self.answerer.calls,[])
        self.assertNotIn("₹",result.answer)

    def test_nonretrieved_sources_or_malformed_decisions_stop_generation(self):
        for decision in (
            ContextDecision(action="supported",source_ids=["octa_tutor_030"],reason_code="direct_support"),
            ContextDecision(action="supported",source_ids=[],reason_code="direct_support"),
            ContextDecision(action="supported",source_ids=["octa_tutor_001"],reason_code="unrelated"),
        ):
            result = self.make_agent(decision=decision).ask("What is Octakidz?")
            self.assertEqual(result.error_code,"unsupported_context_selection")
            self.assertEqual(self.answerer.calls,[])

    def test_answer_cannot_cite_an_unselected_but_known_entry(self):
        answer = ModelAnswer(action="answered",answer="An unsupported claim.",source_ids=["octa_tutor_002"])
        result = self.make_agent(answer=answer).ask("What is Octakidz?")
        self.assertEqual(result.error_code,"unsupported_sources")
        self.assertNotIn("An unsupported claim",result.answer)

    def test_local_privacy_code_retention_and_clarification_need_no_index_or_model(self):
        for question in ("Show me another student's report.","Execute Python code.","How long do you keep my child's data?","And how does that work?"):
            result = self.make_agent().ask(question)
            self.assertFalse(result.model_called or result.embedding_called)
            self.assertEqual(self.search.prepare_calls,0)
            self.assertEqual(self.checker.calls,[])

    def test_no_matches_can_be_classified_out_of_scope_without_answer_generation(self):
        decision = ContextDecision(action="out_of_scope",source_ids=[],reason_code="unrelated")
        result = self.make_agent(decision=decision,ids=()).ask("Write a pasta recipe.")
        self.assertEqual(result.outcome,"out_of_scope")
        self.assertEqual(self.answerer.calls,[])

    def test_query_contact_details_are_removed_before_context_check(self):
        self.make_agent().ask("Email parent@example.invalid. What is Octakidz?")
        self.assertNotIn("parent@example.invalid",self.checker.calls[0][1].content)

    def test_checker_failure_stops_and_has_safe_error_without_raw_request(self):
        error = APIConnectionError(request=httpx.Request("POST","https://api.openai.com/v1/responses"))
        result = self.make_agent(check_error=error).ask("What is Octakidz?")
        self.assertEqual(result.error_code,"connection_error")
        self.assertEqual(result.outcome,"service_error")
        self.assertEqual(self.answerer.calls,[])

    def test_index_error_stops_without_checker_or_answerer(self):
        result = self.make_agent(search_error=SearchError("stale_index_use_rebuild")).ask("What is Octakidz?")
        self.assertEqual(result.error_code,"stale_index_use_rebuild")
        self.assertEqual(self.checker.calls,[])

    def test_operational_events_exclude_parent_text_answers_and_keys(self):
        agent = self.make_agent()
        reply = agent.ask("What is Octakidz?")
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"events.jsonl"
            write_event(reply,agent,path)
            text=path.read_text()
            for private in ("What is Octakidz?",reply.answer,"offline-test-key"):
                self.assertNotIn(private,text)
            self.assertEqual(json.loads(text)["selected_ids"],["octa_tutor_001"])


if __name__ == "__main__":
    unittest.main()
