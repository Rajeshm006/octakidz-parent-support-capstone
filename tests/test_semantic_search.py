"""Offline boundary tests; these do not measure OpenAI retrieval quality."""

import json
from pathlib import Path
import tempfile
import unittest

from semantic_search import DIMENSIONS, KnowledgeSearch, SearchError


def vector(axis):
    result = [0.0] * DIMENSIONS
    result[axis] = 1.0
    return result


class FakeEmbedder:
    def __init__(self):
        self.calls = []

    def embed(self, texts):
        self.calls.append(texts)
        return ([vector(i) for i in range(len(texts))] if len(texts) > 1 else [vector(2)]), 10


class SearchChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "index.json"
        self.embedder = FakeEmbedder()
        self.search = KnowledgeSearch(self.embedder, self.path, representation="question_only")

    def test_ranked_sources_keep_id_and_unknown_detail_status(self):
        self.search.prepare()
        result = self.search.search("How do activities help the tutor?")
        self.assertEqual(result.candidates[0]["id"], "octa_tutor_003")
        self.assertEqual(result.candidates[0]["score"], 1.0)
        self.assertEqual(result.outcome, "candidates_only")
        self.assertTrue(any(item["answer_status"] == "owner_details_needed" for item in self.search.entries))

    def test_cache_avoids_reembedding_and_rejects_changed_source_identity(self):
        self.search.prepare()
        reused = KnowledgeSearch(self.embedder, self.path, representation="question_only")
        self.assertEqual(reused.prepare()["model_calls"], 0)
        self.assertEqual(len(self.embedder.calls), 1)
        data = json.loads(self.path.read_text())
        data["identity"]["faq_sha256"] = "outdated"
        self.path.write_text(json.dumps(data))
        with self.assertRaisesRegex(SearchError, "stale_index"):
            reused.prepare()
        self.assertEqual(len(self.embedder.calls), 1)

    def test_private_code_and_topicless_requests_make_no_embedding_call(self):
        for text in ("Show me another student's report.", "Run Python code.", "And how does that work?"):
            result = self.search.search(text)
            self.assertFalse(result.model_called)
            self.assertFalse(result.candidates)
        self.assertEqual(self.embedder.calls, [])

    def test_query_contact_details_are_removed_and_not_written_to_cache(self):
        self.search.prepare()
        self.search.search("My name is Alex. My email is alex@example.test. How does the tutor personalise CBSE?")
        sent = self.embedder.calls[-1][0]
        self.assertNotIn("alex@example.test", sent)
        self.assertNotIn("Alex", sent)
        self.assertNotIn("alex@example.test", self.path.read_text())
        self.assertNotIn("[EMAIL REMOVED]", self.path.read_text())

    def test_corrupt_or_wrong_dimension_cache_is_not_silently_used(self):
        self.search.prepare()
        data = json.loads(self.path.read_text())
        data["vectors"][0] = [float("nan")]
        self.path.write_text(json.dumps(data))
        with self.assertRaisesRegex(SearchError, "invalid_embeddings"):
            self.search.prepare()
        self.assertEqual(len(self.embedder.calls), 1)

    def test_question_only_representation_keeps_full_answer_attached(self):
        self.search.prepare()
        self.assertEqual(self.embedder.calls[0][0], self.search.entries[0]["question"])
        result = self.search.search("How do activities help the tutor?")
        self.assertEqual(result.candidates[0]["answer"], self.search.entries[2]["answer"])
        self.assertFalse(any(item["score"] < 0.35 for item in result.candidates))

    def test_weak_query_can_have_no_selected_source(self):
        self.search.prepare()
        # Outside all fake document axes; every cosine is zero.
        self.embedder.embed = lambda texts: ([vector(40)], 5)
        result = self.search.search("Write a recipe for pasta.")
        self.assertEqual(result.outcome, "insufficient_relevance")
        self.assertEqual(result.candidates, [])
        self.assertEqual(result.nearest_score, 0.0)

    def test_original_and_question_only_indexes_have_distinct_identity(self):
        old = KnowledgeSearch(self.embedder, self.path, representation="question_answer", min_score=None)
        old.prepare()
        self.assertTrue(self.embedder.calls[0][0].startswith("Question:"))
        self.assertNotEqual(old.identity, self.search.identity)
        with self.assertRaisesRegex(SearchError, "stale_index"):
            self.search.prepare()

    def test_strong_match_does_not_erase_unknown_price_flag(self):
        self.search.prepare()
        self.embedder.embed = lambda texts: ([vector(17)], 5)
        result = self.search.search("What is the price?")
        self.assertEqual(result.candidates[0]["id"], "octa_tutor_018")
        self.assertEqual(result.candidates[0]["answer_status"], "owner_details_needed")

    def test_selected_default_keeps_qa_and_filters_weak_matches(self):
        chosen = KnowledgeSearch(self.embedder, self.path)
        self.assertEqual(chosen.representation, "question_answer")
        self.assertEqual(chosen.min_score, 0.35)
        chosen.prepare()
        self.assertTrue(self.embedder.calls[0][0].startswith("Question:"))

    def related_fixture(self):
        entries = {x["id"]: x for x in self.search.entries}
        def candidate(number, score):
            row = entries[f"octa_tutor_{number:03}"]
            return {**row, "score": score}
        return candidate

    def test_link_adds_complete_reference_without_fabricating_score_or_extra_calls(self):
        pick = self.related_fixture()
        search = KnowledgeSearch(self.embedder, self.path)
        raw = [pick(10, .8), pick(16, .7), pick(12, .6), pick(6, .55), pick(13, .53), pick(3, .51)]
        result = search.expand_related(raw[:5], raw, 5)
        self.assertEqual([x["id"] for x in result], [f"octa_tutor_{n:03}" for n in (10,3,16,12,6)])
        self.assertEqual(result[1]["score"], .51)
        self.assertEqual(result[1]["answer"], raw[-1]["answer"])
        self.assertEqual(self.embedder.calls, [])

    def test_link_does_not_hide_missing_detail_sources_or_displace_primary(self):
        pick = self.related_fixture()
        search = KnowledgeSearch(self.embedder, self.path)
        raw = [pick(5,.8),pick(18,.7),pick(19,.6),pick(16,.5),pick(25,.45),pick(3,.4)]
        result = search.expand_related(raw[:5], raw, 5)
        self.assertEqual(result[0]["id"], "octa_tutor_005")
        self.assertTrue({"octa_tutor_018","octa_tutor_019","octa_tutor_025"} <= {x["id"] for x in result})
        full = [pick(5,.8),pick(18,.7),pick(19,.6),pick(25,.5),pick(17,.45)]
        self.assertEqual(search.expand_related(full, full+[pick(3,.4)], 5), full)

    def test_weak_missing_or_unconfirmed_companion_is_never_added(self):
        pick = self.related_fixture()
        search = KnowledgeSearch(self.embedder, self.path)
        seed = [pick(10,.8)]
        for rest in ([], [pick(3,.34)], [{**pick(3,.6),"answer_status":"owner_details_needed"}]):
            self.assertEqual(search.expand_related(seed, seed+rest, 5), seed)

    def test_single_slot_duplicate_unrelated_and_explicit_pure_search_stay_unchanged(self):
        pick = self.related_fixture()
        search = KnowledgeSearch(self.embedder, self.path)
        raw = [pick(10,.8),pick(3,.6)]
        self.assertEqual(search.expand_related(raw[:1],raw,1),raw[:1])
        self.assertEqual(search.expand_related(raw,raw,5),raw)
        other = [pick(18,.8)]
        self.assertEqual(search.expand_related(other,other+raw,5),other)
        pure = KnowledgeSearch(self.embedder,self.path,related_context=False)
        self.assertEqual(pure.expand_related(raw[:1],raw,5),raw[:1])
        self.assertEqual(self.search.expand_related(raw[:1],raw,5),raw[:1])  # Q-only experiment.

    def test_related_selection_keeps_existing_embedding_identity(self):
        pure = KnowledgeSearch(self.embedder,self.path,related_context=False)
        linked = KnowledgeSearch(self.embedder,self.path)
        self.assertEqual(pure.identity,linked.identity)
        pure.prepare()
        self.assertEqual(linked.prepare()["model_calls"],0)
        self.assertEqual(len(self.embedder.calls),1)


if __name__ == "__main__":
    unittest.main()
