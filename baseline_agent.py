"""Phase 2: a small FAQ assistant using only Python's built-in libraries."""

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from time import perf_counter
import unicodedata


PROJECT_DIR = Path(__file__).resolve().parent
FAQ_PATH = PROJECT_DIR / "octakidz_ai_tutor_faq.json"
LOG_PATH = PROJECT_DIR / "logs" / "baseline_events.jsonl"
CONTACT_URL = "https://octakidz.com/contact-us"
CONTACT_EMAIL = "support@octakidz.com"
VERSION = "phase2_baseline_v2"


@dataclass(frozen=True)
class Reply:
    answer: str
    outcome: str
    faq_id: str | None = None
    source_url: str | None = None


def normalise(text: str) -> str:
    """Ignore capital letters, punctuation and repeated spaces, not meaning."""
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.sub(r"[^\w\s]", " ", text).split())


def load_faq(path: Path = FAQ_PATH) -> dict:
    """Load known question wording into a lookup table."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("questions"), list):
        raise ValueError("The FAQ must contain a questions list.")
    if not data["questions"] or data.get("total_questions") != len(data["questions"]):
        raise ValueError("The FAQ count must match its non-empty questions list.")

    lookup = {}
    seen_ids = set()
    for item in data["questions"]:
        if not isinstance(item, dict):
            raise ValueError("Each FAQ entry must be an object.")
        for field in ("id", "question", "answer", "url", "answer_status"):
            if not isinstance(item.get(field), str) or not item[field].strip():
                raise ValueError("An FAQ entry is missing a required text field.")
        if not re.fullmatch(r"octa_tutor_\d{3}", item["id"]) or item["id"] in seen_ids:
            raise ValueError("FAQ IDs must be unique tutor identifiers.")
        seen_ids.add(item["id"])
        if item["answer_status"] not in {"grounded_draft", "owner_details_needed"}:
            raise ValueError("Unrecognised answer status.")
        alternatives = item.get("paraphrases", [])
        if not isinstance(alternatives, list) or any(
            not isinstance(text, str) or not text.strip() for text in alternatives
        ):
            raise ValueError("Alternative questions must be a list of text.")

        for wording in [item["question"], *alternatives]:
            key = normalise(wording)
            if not key or (key in lookup and lookup[key]["id"] != item["id"]):
                raise ValueError("Question wording must identify one FAQ answer.")
            lookup[key] = item
    return lookup


def respond(question: str, lookup: dict) -> Reply:
    """Apply a few rules, then look up a saved answer. Never execute input."""
    wording = normalise(question)
    if not wording:
        return Reply("Please ask a question about Octakidz's activities or AI tutor.", "clarify")

    # Chat input stays text. There is no interpreter or command-running tool.
    if re.search(r"\b(run|execute)\b", wording) and re.search(
        r"\b(python|code|script|command|shell)\b", wording
    ):
        return Reply(
            "I can help with Octakidz questions, but I cannot run code or commands.",
            "refused_code",
        )

    other_student = any(
        phrase in wording
        for phrase in ("another student", "other student", "another child", "other child", "someone else")
    )
    private_record = any(
        word in wording.split()
        for word in ("report", "reports", "map", "profile", "records", "data", "details", "contact", "name")
    )
    if other_student and private_record:
        return Reply(
            "I cannot share another child's personal information. I can explain "
            "the programme in general, or you can contact the team for account support.",
            "refused_private_data",
            source_url=CONTACT_URL,
        )

    if any(
        phrase in wording
        for phrase in ("speak to a human", "talk to a human", "speak to your team", "talk to your team", "real person")
    ):
        return Reply(
            "You can reach the Octakidz team through the Contact Us page and tell them what you need help with.",
            "human_support",
            source_url=CONTACT_URL,
        )

    item = lookup.get(wording)
    if item:
        outcome = "needs_owner_details" if item["answer_status"] == "owner_details_needed" else "faq_match"
        return Reply(item["answer"], outcome, item["id"], item["url"])

    return Reply(
        "I'm not sure about that yet. Would you like to speak to our team? "
        f"You can email us at {CONTACT_EMAIL} or use the Contact Us page.",
        "no_match",
        source_url=CONTACT_URL,
    )


def write_event(path: Path, reply: Reply, latency_ms: float) -> None:
    """Record operational metadata only: no question or answer text."""
    event = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "version": VERSION,
        "outcome": reply.outcome,
        "faq_id": reply.faq_id,
        "latency_ms": round(latency_ms, 3),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event) + "\n")


def answer_once(question: str, lookup: dict, log_path: Path) -> None:
    started = perf_counter()
    reply = respond(question, lookup)
    latency_ms = (perf_counter() - started) * 1000
    print(f"\nOctakidz: {reply.answer}")
    if reply.source_url:
        print(f"More help: {reply.source_url}")
    try:
        write_event(log_path, reply, latency_ms)
    except OSError:
        print("The answer is ready, but its metadata log could not be saved.", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", "-q", help="Ask one question, then exit.")
    parser.add_argument("--faq-file", type=Path, default=FAQ_PATH)
    parser.add_argument("--log-file", type=Path, default=LOG_PATH)
    args = parser.parse_args()

    try:
        lookup = load_faq(args.faq_file)
    except (OSError, UnicodeError, ValueError):
        print(
            f"Support information could not be loaded. Please try again or contact {CONTACT_EMAIL}.",
            file=sys.stderr,
        )
        return 1

    if args.question is not None:
        answer_once(args.question, lookup, args.log_file)
        return 0

    print("Octakidz Parent Support. Type exit to finish.")
    while True:
        try:
            question = input("\nYou: ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            return 0
        if question.strip().casefold() in {"exit", "quit"}:
            print("Goodbye!")
            return 0
        answer_once(question, lookup, args.log_file)


if __name__ == "__main__":
    raise SystemExit(main())
