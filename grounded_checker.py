"""A source picker whose output IDs are limited to this turn's retrieved entries."""

import json

from langchain_core.messages import SystemMessage

from llm_agent import make_structured_model
from rag_agent import ContextDecision, MAX_SOURCES


WORDING_GUIDANCE = """Interpret ordinary parent wording generously: spelling mistakes,
informal English and awkward word order do not by themselves make a product question
unanswerable. A name replaced by 'my child' or 'the parent' identifies the role; it
does not require a private student record. When the intent is a general question
about how Octakidz helps a child, use supported learning benefits and their limits.
Do not require an exact reference question or perfect product spelling. Do not turn
general benefits into a personal assessment or guaranteed result. Preserve every
material part of the current question: missing policies, services, prices or other
unsupported details still require human_support. Truly different plausible intents
still require clarification. Choose only the IDs permitted by the response schema;
supported needs one to three directly useful sources, never an empty list.
"""


def decision_schema(candidates):
    ids = list(dict.fromkeys(item["id"] for item in candidates))
    schema = ContextDecision.model_json_schema()
    schema["title"] = "RetrievedSourceDecision"
    sources = schema["properties"]["source_ids"]
    sources["maxItems"] = MAX_SOURCES if ids else 0
    if ids:
        sources["items"]["enum"] = ids
    return schema


class RetrievedSourceChecker:
    """One existing checker request, using a per-turn strict response schema."""

    def __init__(self, settings, factory=make_structured_model):
        self.settings = settings
        self.factory = factory

    def invoke(self, messages, config=None):
        marker = "\n\nCANDIDATE REFERENCES:\n"
        instructions, text = messages[0].content.rsplit(marker, 1)
        candidates = json.loads(text)
        if (not isinstance(candidates, list)
                or any(not isinstance(item, dict) or not isinstance(item.get("id"), str) for item in candidates)):
            raise ValueError("invalid_checker_candidates")
        schema = decision_schema(candidates)
        revised = [SystemMessage(content=instructions + "\n\n" + WORDING_GUIDANCE + marker + text), *messages[1:]]
        result = self.factory(self.settings, schema, max_tokens=300).invoke(revised, config=config)
        if not result.get("parsing_error") and isinstance(result.get("parsed"), dict):
            # The JSON-schema SDK path returns a dict. Preserve the existing RAG
            # decision type and all of its independent post-model safeguards.
            result = {**result, "parsed": ContextDecision.model_validate(result["parsed"])}
        return result
