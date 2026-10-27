"""Source-choice schema, retained fail-closed checks and privacy-safe roles."""

import json
import unittest

from langchain_core.messages import HumanMessage, SystemMessage

from conversation_memory import ChatNames
from grounded_checker import RetrievedSourceChecker, decision_schema
from llm_agent import ModelAnswer, Settings
from rag_agent import ContextDecision, RAGAgent
from test_rag_agent import FakeRunnable, FakeSearch
from tool_agent import ModelBudget


def messages(ids, question="How does Octakidz help my child?"):
    facts = [{"id":s,"question":"Public question", "answer":"Public answer", "answer_status":"grounded_draft"} for s in ids]
    return [SystemMessage(content="Base safety rules\n\nCANDIDATE REFERENCES:\n"+json.dumps(facts)),
            HumanMessage(content=question)]


class CheckerChecks(unittest.TestCase):
    def test_wire_schema_limits_ids_to_current_candidates_and_at_most_three(self):
        schema = decision_schema([{"id":"octa_tutor_014"},{"id":"octa_tutor_015"},{"id":"octa_tutor_014"}])
        prop = schema["properties"]["source_ids"]
        self.assertEqual(prop["items"]["enum"], ["octa_tutor_014","octa_tutor_015"])
        self.assertEqual(prop["maxItems"], 3)
        self.assertFalse(schema["additionalProperties"])
        self.assertNotIn("enum", ContextDecision.model_json_schema()["properties"]["source_ids"]["items"])

    def test_no_candidates_permits_only_empty_source_list(self):
        self.assertEqual(decision_schema([])["properties"]["source_ids"]["maxItems"],0)

    def test_one_request_per_turn_with_fresh_schema_and_dict_conversion(self):
        calls=[]
        class Model:
            def __init__(self,source):self.source=source
            def invoke(self,ms,config=None):
                return {"parsed":{"action":"supported","source_ids":[self.source],"reason_code":"direct_support"},
                        "raw":"usage-object", "parsing_error":None}
        def factory(settings,schema,max_tokens):
            calls.append(schema)
            self.assertEqual(max_tokens,300)
            return Model(schema["properties"]["source_ids"]["items"]["enum"][0])
        checker=RetrievedSourceChecker(Settings(api_key="offline"),factory)
        for source in ("octa_tutor_001","octa_tutor_014"):
            result=checker.invoke(messages([source]))
            self.assertIsInstance(result["parsed"],ContextDecision)
            self.assertEqual(result["parsed"].source_ids,[source])
            self.assertEqual(result["raw"],"usage-object")
        self.assertEqual(len(calls),2)
        self.assertNotIn("octa_tutor_001",json.dumps(calls[1]))

    def test_wording_guidance_preserves_entire_question_and_base_safety(self):
        captured=[]
        class Model:
            def invoke(self,ms,config=None):
                captured.extend(ms)
                return {"parsed":{"action":"human_support","source_ids":[],"reason_code":"partial_support"},"raw":None,"parsing_error":None}
        q="how will my child be benifitng of ocatkidz, and is bus pickup included?"
        checker=RetrievedSourceChecker(Settings(api_key="offline"),lambda *a,**k:Model())
        checker.invoke(messages(["octa_tutor_014"],q))
        self.assertIn("Base safety rules",captured[0].content)
        self.assertIn("spelling mistakes",captured[0].content)
        self.assertIn("Preserve every",captured[0].content)
        self.assertEqual(captured[1].content,q)

    def test_bad_candidate_payload_stops_before_provider(self):
        calls=[]
        checker=RetrievedSourceChecker(Settings(api_key="offline"),lambda *a,**k:calls.append(1))
        for payload in ("not JSON", "{}", '[{"answer":"no ID"}]'):
            with self.assertRaises(ValueError):
                checker.invoke([SystemMessage(content="Rules\n\nCANDIDATE REFERENCES:\n"+payload),HumanMessage(content="Q")])
        self.assertEqual(calls,[])

    def test_sdk_parse_failure_is_preserved_instead_of_retried(self):
        calls=[]
        class Model:
            def invoke(self,*a,**k):
                calls.append(1)
                return {"parsed":None,"raw":None,"parsing_error":"injected"}
        result=RetrievedSourceChecker(Settings(api_key="offline"),lambda *a,**k:Model()).invoke(messages([]))
        self.assertEqual(result["parsing_error"],"injected")
        self.assertEqual(len(calls),1)

    def test_independent_rag_guard_still_rejects_unretrieved_or_excess_sources(self):
        for ids in (["octa_tutor_030"],["octa_tutor_001","octa_tutor_002","octa_tutor_003","octa_tutor_004"],[]):
            class InvalidModel:
                def invoke(self,*a,**k):
                    return {"parsed":{"action":"supported","source_ids":ids,"reason_code":"direct_support"},"raw":None,"parsing_error":None}
            checker=RetrievedSourceChecker(Settings(api_key="offline"),lambda *a,**k:InvalidModel())
            answerer=FakeRunnable(ModelAnswer(action="answered",answer="Should not run",source_ids=["octa_tutor_001"]))
            reply=RAGAgent(Settings(api_key="offline"),FakeSearch(),checker,answerer).ask("What is Octakidz?")
            self.assertEqual(reply.error_code,"unsupported_context_selection")
            self.assertFalse(reply.answerer_called)
            self.assertEqual(answerer.calls,[])

    def test_known_names_become_roles_while_typos_and_extra_requests_are_preserved(self):
        names=ChatNames(parent="Arun",child="Leela")
        for original,expected in (
            ("will leela be helpful of the octakidz ?","will my child be helpful of the octakidz ?"),
            ("how will leela be benifitng of ocatkidz ?","how will my child be benifitng of ocatkidz ?"),
            ("How will arun pay, and is a bus included?","How will the parent pay, and is a bus included?")):
            self.assertEqual(names.redact(original),expected)

    def test_checker_debug_emits_only_public_id_shapes_not_arbitrary_model_text(self):
        events=[]
        budget=ModelBudget(lambda stage,**data:events.append({"stage":stage,**data}))
        model=FakeRunnable(ContextDecision(action="supported",reason_code="direct_support",
                                          source_ids=["octa_tutor_001","arun@example.invalid"]))
        budget.wrap(model,"support_check").invoke([])
        proposal=next(e for e in events if e['stage']=='checker_proposal')
        self.assertEqual(proposal['proposed_source_ids'],['octa_tutor_001'])
        self.assertEqual(proposal['invalid_id_format_count'],1)
        self.assertNotIn('arun',json.dumps(events))
