"""Phase 4 search preview: compare representations and reject very weak matches."""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from time import perf_counter

from openai import APIConnectionError, APIError, APITimeoutError, AuthenticationError, OpenAI, RateLimitError

from baseline_agent import FAQ_PATH, PROJECT_DIR, load_faq
from llm_agent import ConfigurationError, Settings
from support_safety import preflight, redact_contact_details


VERSION = "phase9_search_v1"
EMBEDDING_MODEL = "text-embedding-3-small"
DIMENSIONS = 1536
REPRESENTATIONS = ("question_answer", "question_only")
DEFAULT_REPRESENTATION = "question_answer"
MIN_SCORE = 0.35
CACHE_PATH = PROJECT_DIR / ".cache" / "faq_embeddings.json"
# Authored links between existing public FAQs, not matches against test questions.
# The complete activities-to-tutor explanation supplements narrower map/activity
# entries when one is the strongest semantic hit. Similarity and support checks
# still apply; no product fact or document is invented by following a link.
RELATED_SUPPORT_SOURCES = {
    "octa_tutor_004": "octa_tutor_003",
    "octa_tutor_005": "octa_tutor_003",
    "octa_tutor_010": "octa_tutor_003",
}


class SearchError(Exception):
    """A safe error code; provider exceptions and input text stay private."""


def validate_vectors(vectors, count):
    if not isinstance(vectors, list) or len(vectors) != count:
        raise SearchError("invalid_embeddings")
    for vector in vectors:
        if (not isinstance(vector, list) or len(vector) != DIMENSIONS
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in vector)
                or not any(vector)):
            raise SearchError("invalid_embeddings")


def cosine(left, right):
    return sum(a * b for a, b in zip(left, right, strict=True)) / (
        math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    )


class OpenAIEmbedder:
    def __init__(self, settings):
        self.client = OpenAI(api_key=settings.api_key, base_url="https://api.openai.com/v1", timeout=30, max_retries=0)

    def embed(self, texts):
        try:
            response = self.client.embeddings.create(
                model=EMBEDDING_MODEL, dimensions=DIMENSIONS, input=texts, encoding_format="float",
            )
            rows = sorted(response.data, key=lambda row: row.index)
            if [row.index for row in rows] != list(range(len(texts))):
                raise SearchError("invalid_embeddings")
            vectors = [row.embedding for row in rows]
            validate_vectors(vectors, len(texts))
            return vectors, response.usage.total_tokens
        except AuthenticationError:
            raise SearchError("authentication_error") from None
        except RateLimitError:
            raise SearchError("rate_limit_or_quota") from None
        except APITimeoutError:
            raise SearchError("timeout") from None
        except APIConnectionError:
            raise SearchError("connection_error") from None
        except APIError:
            raise SearchError("api_error") from None


@dataclass(frozen=True)
class SearchResult:
    outcome: str
    candidates: list[dict]
    local_answer: str | None = None
    model_called: bool = False
    input_tokens: int = 0
    latency_ms: float = 0
    nearest_score: float | None = None


class KnowledgeSearch:
    def __init__(self, embedder, cache_path=None, representation=DEFAULT_REPRESENTATION, min_score=MIN_SCORE,
                 related_context=True):
        if representation not in REPRESENTATIONS:
            raise ValueError("Choose a known search representation")
        if min_score is not None and (not math.isfinite(min_score) or not -1 <= min_score <= 1):
            raise ValueError("min_score must be a cosine similarity or None")
        self.embedder = embedder
        self.representation = representation
        self.min_score = min_score
        self.related_context = related_context and representation == "question_answer"
        filename = "faq_embeddings.json" if representation == "question_answer" else "faq_question_embeddings.json"
        self.cache_path = Path(cache_path) if cache_path is not None else PROJECT_DIR / ".cache" / filename
        self.entries = sorted({item["id"]: item for item in load_faq().values()}.values(), key=lambda item: item["id"])
        # Q+A remains the returned chunk. Only its search representation changes.
        # Author-provided paraphrases are still excluded.
        self.texts = [
            item["question"] if representation == "question_only" else f"Question: {item['question']}\nAnswer: {item['answer']}"
            for item in self.entries
        ]
        self.identity = {
            # Index versions describe embedded bytes, independently of query policy.
            # The original Q+A cache remains valid and is never overwritten by Q-only.
            "version": "phase4_search_v1" if representation == "question_answer" else "phase4_question_index_v1",
            "model": EMBEDDING_MODEL, "dimensions": DIMENSIONS,
            "faq_sha256": hashlib.sha256(FAQ_PATH.read_bytes()).hexdigest(),
            "chunk_sha256": hashlib.sha256(json.dumps(self.texts, ensure_ascii=False).encode()).hexdigest(),
            "entry_ids": [item["id"] for item in self.entries],
        }
        self.vectors = None

    def prepare(self, rebuild=False):
        if self.cache_path.exists() and not rebuild:
            try:
                stored = json.loads(self.cache_path.read_text(encoding="utf-8"))
                if not isinstance(stored, dict) or stored.get("identity") != self.identity:
                    raise SearchError("stale_index_use_rebuild")
                validate_vectors(stored.get("vectors"), len(self.entries))
                self.vectors = stored["vectors"]
                return {"source": "local_cache", "model_calls": 0, "input_tokens": 0}
            except (ValueError, TypeError, KeyError):
                raise SearchError("invalid_index_use_rebuild") from None
        vectors, tokens = self.embedder.embed(self.texts)
        validate_vectors(vectors, len(self.entries))
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.cache_path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"identity": self.identity, "vectors": vectors}), encoding="utf-8")
        temporary.replace(self.cache_path)
        self.vectors = vectors
        return {"source": "new_embeddings", "model_calls": 1, "input_tokens": tokens}

    def rank(self, query_vector, top_k=3):
        """Return raw neighbours for evidence, before applying the weak-match floor."""
        if not 1 <= top_k <= len(self.entries):
            raise ValueError("top_k is outside the knowledge size")
        if self.vectors is None:
            raise SearchError("prepare_index_first")
        validate_vectors([query_vector], 1)
        ranked = sorted(
            [(cosine(query_vector, vector), entry) for vector, entry in zip(self.vectors, self.entries, strict=True)],
            key=lambda pair: (-pair[0], pair[1]["id"]),
        )[:top_k]
        return [
            {"id": item["id"], "score": round(score, 6), "question": item["question"],
             "answer": item["answer"], "answer_status": item["answer_status"], "url": item["url"]}
            for score, item in ranked
        ]

    def select(self, candidates):
        """A conservative score filter, NOT proof that sources answer a question."""
        if self.min_score is None:
            return list(candidates)
        return [item for item in candidates if item["score"] >= self.min_score]

    def expand_related(self, candidates, raw_ranked, top_k):
        """Reserve at most one slot for a linked FAQ, retaining its real score."""
        chosen = list(candidates)
        if not self.related_context or not chosen or top_k < 2:
            return chosen
        identifier = RELATED_SUPPORT_SOURCES.get(chosen[0]["id"])
        if not identifier or any(item["id"] == identifier for item in chosen):
            return chosen
        companion = next((item for item in raw_ranked if item["id"] == identifier), None)
        if (companion is None or companion["answer_status"] != "grounded_draft"
                or not self.select([companion])):
            return chosen
        if len(chosen) >= top_k:
            # Never displace the best semantic hit or a known missing-detail
            # fallback. Those entries preserve uncertainty about price/policies.
            removable = next((i for i in range(len(chosen)-1, 0, -1)
                              if chosen[i]["answer_status"] == "grounded_draft"), None)
            if removable is None:
                return chosen
            chosen.pop(removable)
        chosen.insert(1, companion)
        return chosen

    def search(self, question, top_k=3):
        started = perf_counter()
        if not 1 <= top_k <= len(self.entries):
            raise ValueError("top_k is outside the knowledge size")
        local = preflight(question)
        if local:
            return SearchResult(local.outcome, [], local.answer, latency_ms=(perf_counter()-started)*1000)
        if self.vectors is None:
            raise SearchError("prepare_index_first")
        # Query embeddings are used in memory only; no parent-query cache or logs.
        vectors, tokens = self.embedder.embed([redact_contact_details(question)])
        validate_vectors(vectors, 1)
        ranked = self.rank(vectors[0], len(self.entries))
        candidates = self.expand_related(self.select(ranked[:top_k]), ranked, top_k)
        return SearchResult("candidates_only" if candidates else "insufficient_relevance", candidates,
                            model_called=True, input_tokens=tokens, nearest_score=ranked[0]["score"],
                            latency_ms=(perf_counter()-started)*1000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--question", "-q")
    group.add_argument("--evaluate", action="store_true", help="Search the same fixed 11 questions; save observed rankings.")
    parser.add_argument("--rebuild", action="store_true", help="Explicitly replace an outdated or invalid local FAQ index.")
    parser.add_argument("--representation", choices=REPRESENTATIONS, default=DEFAULT_REPRESENTATION)
    args = parser.parse_args()
    # Refusals/clarification need neither a key nor an index-build call.
    if args.question is not None and (local := preflight(args.question)):
        print(local.answer)
        return 0
    directory = None
    try:
        search = KnowledgeSearch(OpenAIEmbedder(Settings.from_local_env()), representation=args.representation)
        if args.evaluate:
            cases_path = PROJECT_DIR / "evaluation" / "phase_comparison_cases.json"
            cases = json.loads(cases_path.read_text(encoding="utf-8"))["cases"]
            stamp = datetime.now(timezone.utc)
            directory = PROJECT_DIR / "evidence" / "phase4" / stamp.strftime("search_%Y%m%dT%H%M%S%fZ")
            report = {"started_utc": stamp.isoformat(), "status": "running", "version": VERSION,
                      "stage": "Search preview only; no retrieval-connected answer generation yet.",
                      "data_origin": "Fixed authored support questions; no real student records.",
                      "model": EMBEDDING_MODEL, "dimensions": DIMENSIONS, "top_k": 3,
                      "representation": search.representation, "min_score": search.min_score,
                      "identity": search.identity, "test_set_sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
                      "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "rows": []}
            directory.mkdir(parents=True)
            def save():
                (directory / "results.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
            save()
        index_info = search.prepare(rebuild=args.rebuild)
        if not args.evaluate:
            reply = search.search(args.question)
            print("Developer search preview — these are candidate sources, not the parent's final answer.")
            if not reply.candidates:
                print("No candidate passed the experimental relevance floor. Do not generate a product answer from weak matches.")
            for item in reply.candidates:
                print(f"{item['id']} | similarity {item['score']:.3f} | {item['answer_status']} | {item['question']}")
            return 0
        report["index"] = index_info
        for case in cases:
            reply = search.search(case["question"])
            report["rows"].append({"case_id": case["id"], "question": case["question"],
                                   "expected_points": case["expected_points"], "search": asdict(reply), "review_notes": None})
            save()
            top = reply.candidates[0]["id"] if reply.candidates else reply.outcome
            print(f"{case['id']}: {top}", flush=True)
        report["status"] = "complete_rankings_need_review"
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()
        print(f"Saved search evidence in {directory.relative_to(PROJECT_DIR)}")
        return 0
    except (ConfigurationError, SearchError, OSError, ValueError) as error:
        code = str(error) if isinstance(error, SearchError) else "configuration_or_file_error"
        if directory is not None:
            report.update(status="stopped_on_error", error_code=code)
            try:
                save()
            except OSError:
                pass
        print(f"Search could not finish ({code}). Check your local setup; no provider details are displayed.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
