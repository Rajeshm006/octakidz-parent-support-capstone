"""Phase 7: temporary chat context and feedback about answer length."""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import re
import sys
from time import monotonic, perf_counter

from baseline_agent import CONTACT_EMAIL, CONTACT_URL, PROJECT_DIR, load_faq, normalise
from conversation_memory import ChatNames, joining_question, recall_subject
from grounded_checker import RetrievedSourceChecker
from llm_agent import ConfigurationError, Settings
from response_feedback import ShortAnswerer, requested_style
from support_safety import preflight, redact_contact_details
from tool_agent import ToolAgent, ToolReply, make_router, print_debug


VERSION = "phase9_answer_quality_v1"
RETENTION_SECONDS = 600
MAX_TOPIC_SOURCES = 3
LOG_PATH = PROJECT_DIR / "logs/conversation_events.jsonl"
CLARIFY_MESSAGE = "Could you tell me which part of Octakidz you'd like me to explain?"
# Deliberately narrow: explicit product topics still stand alone, just as in Phase 5.
EXPLICIT_TOPICS = {
    "octakidz", "tutor", "activity", "activities", "map", "cbse", "plan", "price",
    "cost", "pricing", "trial", "fractions", "report", "reports", "privacy",
    "intelligence", "intelligences", "course", "programme", "program",
}


class ContextRouter:
    """Bind every search to the complete redacted question prepared by the app."""

    def __init__(self, settings):
        self.settings = settings

    def invoke(self, messages, config=None):
        query = messages[-1].content
        if not isinstance(query, str):
            raise ValueError("invalid_router_question")
        # The model still selects search/contact/no tool. Ordinary questions can
        # also be shortened accidentally, so constrain those as well as follow-ups.
        # The Python gateway independently revalidates the complete argument.
        router = make_router(self.settings, query_constraint=query)
        return router.invoke(messages, config=config)


class ContextToolAgent(ToolAgent):
    """Keep Phase 5 instructions; bind search arguments for every current turn."""

    def __init__(self, settings, router=None, checker=None, answerer=None, session_factory=None):
        super().__init__(settings, router if router is not None else ContextRouter(settings),
                         checker if checker is not None else RetrievedSourceChecker(settings),
                         answerer, session_factory)

    def ask(self, question, emit=None, *, answer_style="normal"):
        if answer_style not in {"normal", "short"}:
            raise ValueError("invalid_answer_style")
        if answer_style == "normal":
            return super().ask(question, emit=emit)
        # A per-call composition keeps a shared backend safe across chats. No
        # model is created here: all three existing runnables are supplied.
        turn = ToolAgent(self.settings, self.router, self.checker,
                         ShortAnswerer(self.answerer), self.session_factory)
        return turn.ask(question, emit=emit)


def needs_topic(question):
    text = normalise(question)
    words = set(text.split())
    return (len(text.split()) <= 16
            and bool(words & {"that", "it", "this", "those", "these"})
            and not words & EXPLICIT_TOPICS
            and bool(re.match(r"^(?:(?:and|but) )?(?:how|what|why|can|could|does|do|will|is|are)\b", text)))


@dataclass(frozen=True)
class ConversationReply(ToolReply):
    memory_used: bool = False
    memory_before_ids: tuple[str, ...] = ()
    memory_after_ids: tuple[str, ...] = ()
    memory_state: str = "empty"
    plan: tuple[str, ...] = ()
    name_context_used: bool = False
    name_context_available: bool = False
    answer_style: str = "normal"
    feedback_changed: bool = False


class ConversationAgent:
    """One instance per chat: public topics, local names and one style enum."""

    def __init__(self, agent, clock=monotonic):
        self.agent = agent
        self.clock = clock
        self._sources = {row["id"]: row for row in load_faq().values()}
        self._topic_ids = ()
        self._last_activity = None
        self._names = ChatNames()
        self._answer_style = "normal"

    def reset(self):
        self._topic_ids = ()
        self._last_activity = None
        self._names.clear()
        self._answer_style = "normal"

    def clear_topic(self):
        self._topic_ids = ()
        if not self._names.available() and self._answer_style == "normal":
            self._last_activity = None

    def memory_snapshot(self):
        """Lazy expiry: expired IDs are cleared before any read or reuse."""
        expired = (self._last_activity is not None
                   and self.clock() - self._last_activity >= RETENTION_SECONDS)
        if expired:
            self.reset()
        return {"source_ids": self._topic_ids, "state": "expired" if expired else (
            "available" if self._topic_ids or self._names.available() or self._answer_style == "short" else "empty"),
                "name_context_available": self._names.available(), "retention_seconds": RETENTION_SECONDS,
                "answer_style": self._answer_style}

    def contextual_question(self, question, source_ids):
        # Every title comes from the curated file, never from earlier parent/model text.
        titles = [self._sources[source]["question"] for source in source_ids]
        return ("Previous Octakidz support topic (titles only, not evidence for an answer): "
                + json.dumps(titles, ensure_ascii=False)
                + "\nCurrent parent question: " + redact_contact_details(question))

    def ask(self, question, emit=None):
        started = perf_counter()
        emit = emit or (lambda *args, **kwargs: None)
        snapshot = self.memory_snapshot()
        before = snapshot["source_ids"]
        plan = ["check_current_input"]
        used = False
        names_used = False
        updated = ()
        feedback_changed = False

        def finish(reply):
            result = ConversationReply(**{
                **asdict(reply), "latency_ms": (perf_counter() - started) * 1000,
                "memory_used": used, "memory_before_ids": before,
                "memory_after_ids": self._topic_ids, "memory_state": snapshot["state"],
                "plan": tuple(plan), "name_context_used": names_used,
                "name_context_available": self._names.available(),
                "answer_style": self._answer_style, "feedback_changed": feedback_changed,
            })
            emit("conversation_result", **{k: v for k, v in asdict(result).items() if k != "answer"})
            return result

        if question.strip().casefold() == "/reset":
            self.reset()
            snapshot["state"] = "reset"
            plan.append("clear_topic")
            return finish(ToolReply(answer="I've cleared this chat's names, topic and answer preference. What would you like to know about Octakidz?",
                                    outcome="memory_reset"))

        if (self._names.available() or self._answer_style == "short") and len(question) <= 4000:
            self._last_activity = self.clock()

        local = preflight(question)
        style = requested_style(question)
        # Exact complete-message recognition runs locally. No feedback text is
        # stored or promoted into instructions; mixed requests still face safety.
        if style is not None:
            feedback_changed = style != self._answer_style
            self._answer_style = style
            self._last_activity = self.clock()
            plan.append("remember_answer_length_feedback")
            message = ("Got it. I'll keep future answers brief in this chat, while keeping important details."
                       if style == "short" else "Got it. I'll use the usual level of detail in this chat.")
            return finish(ToolReply(answer=message, outcome="feedback_saved"))
        follow_up = needs_topic(question)
        # Only the absent-topic clarification may be resolved with memory. All other
        # local safeguards run on the untouched input before any context is added.
        if local and not (follow_up and local.outcome == "clarify" and len(question) <= 4000):
            self.clear_topic()
            plan.append("local_reply_and_clear_topic")
            return finish(ToolReply(answer=local.answer, outcome=local.outcome,
                                    contact_url=CONTACT_URL if local.outcome in {
                                        "human_support", "refused_private_data"} else None))

        # Only first names are retained. Neither the raw question nor ages,
        # interests, a generated answer, or a real student record are stored.
        updated = self._names.introduce(question)
        if self._names.available():
            self._last_activity = self.clock()
        subject = recall_subject(question)
        if subject:
            names_used = bool(getattr(self._names, subject))
            plan.append("reply_from_local_first_name_context")
            return finish(ToolReply(answer=self._names.recall(subject), outcome="name_recall"))

        if updated and self._names.introduction_only(question):
            names_used = True
            plan.append("welcome_with_local_first_name")
            if "child" in updated:
                people = (f"{self._names.parent} and {self._names.child}" if self._names.parent
                          else self._names.child)
                message = (f"Hi {people}! Lovely to meet you both. "
                           f"How is {self._names.child} doing today?")
            else:
                message = (f"Hi {self._names.parent}! I can help with Octakidz's activities, "
                           "AI tutor and joining the programme. What would you like to know?")
            return finish(ToolReply(answer=message, outcome="name_introduction"))

        query = self._names.redact(question)
        joining = joining_question(question)
        if joining:
            # In this assistant, 'this/your course' refers to Octakidz. Preserve
            # every current word, while supplying the missing product name.
            query = "Octakidz programme question: " + query
        if follow_up:
            plan.append("resolve_last_confirmed_topic")
            if not before:
                self.clear_topic()
                plan.append("ask_for_topic")
                return finish(ToolReply(answer=CLARIFY_MESSAGE, outcome="clarify"))
            query = self.contextual_question(query, before)
            # Recheck the complete request: e.g. 'How long is that stored?' may
            # only reveal a retention question once the map topic is supplied.
            contextual_check = preflight(query)
            if contextual_check:
                self.clear_topic()
                plan.append("context_safety_reply_and_clear_topic")
                return finish(ToolReply(answer=contextual_check.answer, outcome=contextual_check.outcome,
                                        contact_url=CONTACT_URL if contextual_check.outcome in {
                                            "human_support", "refused_private_data"} else None))
            used = True

        plan.append("route_and_check_current_knowledge")
        emit("conversation_plan", steps=tuple(plan), memory_used=used,
             topic_source_ids=before if used else (), retention_seconds=RETENTION_SECONDS)
        if self._answer_style == "short":
            plan.append("apply_short_answer_preference")
            reply = self.agent.ask(query, emit=emit, answer_style="short")
        else:
            reply = self.agent.ask(query, emit=emit)
        if joining and (reply.outcome == "clarify" or (
                reply.outcome == "human_support" and reply.contact_url == CONTACT_URL)):
            # Existing confirmed team route; no enrolment, price or availability
            # is promised. Keep this unresolved request out of answer generation.
            plan.append("provide_confirmed_enrolment_contact_guidance")
            reply = ToolReply(**{**asdict(reply), "outcome": "human_support", "contact_url": CONTACT_URL, "answer":
                "Our team can help you check suitability and choose a plan, then explain how to enrol. "
                f"You can email {CONTACT_EMAIL} or use the Contact Us page. "
                "I can't complete enrolment or take payment in this chat."})
        if self._names.parent and reply.outcome in {"answered", "human_support"}:
            names_used = True
            greeting = f"Hi {self._names.parent}! " if "parent" in updated else f"{self._names.parent}, "
            answer = reply.answer
            if "parent" not in updated and answer.startswith(("Our ", "The ", "This ", "Your ")):
                answer = answer[0].lower() + answer[1:]
            reply = ToolReply(**{**asdict(reply), "answer": greeting + answer})

        # A generated answer is never copied into memory. Require valid public
        # citations from this turn's retrieved/selected, answerable sources.
        ids = tuple(dict.fromkeys(reply.source_ids))
        valid = (reply.outcome == "answered" and 0 < len(ids) <= MAX_TOPIC_SOURCES
                 and set(ids) <= set(reply.selected_ids) <= set(reply.retrieved_ids)
                 and all(source in self._sources and self._sources[source]["answer_status"] == "grounded_draft"
                         for source in ids))
        self.clear_topic()
        if valid:
            self._topic_ids = ids
            self._last_activity = self.clock()
            plan.append("remember_confirmed_topic_ids")
        else:
            # Prevent stale topics surviving a switch, unsupported request or failure.
            plan.append("clear_topic")
        return finish(reply)


def write_event(reply, agent, path=LOG_PATH):
    event = asdict(reply)
    event.pop("answer")
    event.update(timestamp_utc=datetime.now(timezone.utc).isoformat(), version=VERSION,
                 model=agent.agent.settings.model)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(event) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--debug", action="store_true", help="Show public topic IDs and executed plan stages.")
    args = parser.parse_args()
    try:
        agent = ConversationAgent(ContextToolAgent(Settings.from_local_env()))
    except (ConfigurationError, OSError, ValueError):
        print("Check your local key and support files. Do not share your API key.", file=sys.stderr)
        return 1
    print("Octakidz Parent Support. Type /short or /normal for answer length, /reset to clear this chat, or exit to finish.")
    try:
        while True:
            try:
                question = input("\nYou: ")
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                return 0
            if question.strip().casefold() in {"exit", "quit"}:
                return 0
            reply = agent.ask(question, print_debug if args.debug else None)
            print(f"\nOctakidz: {reply.answer}")
            if reply.contact_url:
                print(f"Contact: {reply.contact_url}")
            try:
                write_event(reply, agent)
            except OSError:
                print("A background logging service had a problem.", file=sys.stderr)
    finally:
        agent.reset()


if __name__ == "__main__":
    raise SystemExit(main())
