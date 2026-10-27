"""Narrow session feedback: a style choice, never raw user instructions."""

import re

from langchain_core.messages import SystemMessage


SHORT_ANSWER_INSTRUCTION = (
    "SESSION ANSWER STYLE: short. Write the parent-facing answer in 2–3 short sentences, "
    "aiming for at most 55 words. Use plain language and omit repetition. "
    "Keep every material part of the question, essential qualifications, uncertainty and "
    "human-help guidance. If those need more words, preserve them rather than force the target. "
    "All existing source, privacy, safety, scope and structured-output rules still apply. "
    "This preference changes presentation only; it adds no product facts or permissions."
)


def requested_style(question):
    """Recognise a complete feedback message; never swallow a second request."""
    if len(question) > 4000:
        return None
    text = re.sub(r"\s+", " ", question.strip().casefold()).rstrip(".!?")
    short_request = re.fullmatch(
        r"(?:please,? )?(?:keep|make) (?:(?:the|your) )?answers? (?:short|shorter)(?:,? please)?",
        text,
    )
    if short_request or text in {"/short", "shorter answers please", "too long", "that was too long"}:
        return "short"
    if text in {"/normal", "normal answers please", "please use normal answers",
                "use normal answers", "usual detail please"}:
        return "normal"
    return None


class ShortAnswerer:
    """Add trusted static wording to one generation call without mutating messages."""

    def __init__(self, delegate):
        self.delegate = delegate

    def invoke(self, messages, config=None):
        # Append to the existing trusted instruction message. Reference facts,
        # the current question and the structured response schema stay intact.
        if not messages or not isinstance(messages[0], SystemMessage) or not isinstance(messages[0].content, str):
            raise ValueError("invalid_answer_messages")
        revised = [messages[0].model_copy(update={
            "content": messages[0].content + "\n\n" + SHORT_ANSWER_INSTRUCTION
        }), *messages[1:]]
        return self.delegate.invoke(revised, config=config)
