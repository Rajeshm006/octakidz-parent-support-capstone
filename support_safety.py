"""Small local checks used before and after a model call; not a complete PII detector."""

import re

from baseline_agent import CONTACT_EMAIL, CONTACT_URL, Reply, normalise, respond


EMAIL_PATTERN = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
PHONE_PATTERN = re.compile(r"(?<!\w)(?:\+?\d[\s().-]?){9,14}\d(?!\w)")
FIRST_NAME = r"[^\W\d_]+(?:[-'’][^\W\d_]+)?"
NOT_NAMES = {"is", "was", "will", "would", "can", "could", "has", "had", "does", "did", "name",
             "likes", "loves", "enjoys", "needs", "wants", "learns", "studies", "prefers", "gets",
             "feels", "finds", "doesnt", "doesn't", "aged", "age", "very", "not", "also",
             "a", "an", "the", "this", "that", "these", "those", "my", "our", "your", "their",
             "his", "her", "still", "just", "only", "really", "already", "always", "never",
             "often", "sometimes", "today", "tomorrow", "yesterday"}
CHILD_INTRODUCTION = re.compile(
    r"\bmy (?:kid|child|son|daughter)(?P<introduction>(?:'s|’s)? name is| is called)?\s+"
    r"(?P<name>" + FIRST_NAME + r")\b", re.IGNORECASE)


def child_name_matches(text):
    """Recognise limited introduction grammar; not a general name/PII detector."""
    for match in CHILD_INTRODUCTION.finditer(text):
        name = match.group("name")
        if len(name) > 32 or name.casefold() in NOT_NAMES:
            continue
        # A named introduction needs more evidence than an arbitrary word after
        # 'my child'. Keep lowercase comma/age/predicate introductions supported.
        tail = text[match.end():]
        if match.group("introduction") or re.match(
                r"\s*(?:[,;.!?]|$)|\s+(?:is|aged|who|she|he|likes|loves|enjoys|needs|wants|"
                r"learns|studies|prefers|gets|feels|finds|has|can|will)\b", tail, re.IGNORECASE):
            yield match


def preflight(question: str) -> Reply | None:
    """Reuse baseline refusals, without using exact FAQ matches as AI answers."""
    local = respond(question, {})
    if local.outcome != "no_match":
        return local
    text = normalise(question)
    if len(question) > 4000:
        return Reply("Please ask one short question about Octakidz so I can help.", "clarify")
    # There is no conversation memory yet. Do not guess an absent topic.
    words = set(text.split())
    record_terms = {"data", "information", "record", "records", "map", "profile", "brain", "uploads", "activity", "activities"}
    retention_request = bool(words & {"retention", "retain", "retained", "forever", "permanently"}) or (
        bool(words & {"store", "stored", "keep", "kept", "save", "saved"})
        and ("how long" in text or "years" in words)
    )
    if words & record_terms and retention_request:
        return Reply(
            "I can't access or confirm a child's records here, and I don't have a confirmed retention period to share. "
            "Please contact our team through the Contact Us page so they can explain how information is stored and for how long.",
            "human_support", source_url=CONTACT_URL,
        )
    topics = {"octakidz", "tutor", "activity", "activities", "map", "cbse", "plan", "price", "cost", "pricing", "trial", "fractions", "report", "reports", "privacy", "intelligence", "intelligences", "course", "programme", "program"}
    if len(text.split()) <= 16 and words & {"that", "it", "this", "those", "these"} and not words & topics:
        return Reply("Could you tell me which part of Octakidz you'd like me to explain?", "clarify")
    if re.search(r"\b(eval|exec|subprocess|os system)\b", text) or "```" in question:
        return Reply("I can help with Octakidz questions, but I cannot run code or commands.", "refused_code")
    if any(phrase in text for phrase in ("my child", "my son", "my daughter")) and (
        re.search(r"\b(open|fetch|download|access)\b", text)
        and re.search(r"\b(report|profile|record|account|map)\b", text)
    ):
        return Reply(
            "I cannot open private student records here. Please contact our team through the verified account-support process.",
            "refused_private_data", source_url=CONTACT_URL,
        )
    return None


def redact_contact_details(text: str) -> str:
    """Remove common contact details and explicitly introduced names/addresses."""
    text = EMAIL_PATTERN.sub("[EMAIL REMOVED]", text)
    text = PHONE_PATTERN.sub("[PHONE REMOVED]", text)
    name_intro = r"(?i:\b(?:my name is|(?:my )?(?:child|son|daughter|kid)(?:'s|’s)? name is|(?:my )?(?:child|son|daughter|kid) is called)\s+)"
    text = re.sub(name_intro + FIRST_NAME + r"(?:[ -][A-Z][a-z]+){0,2}", "[NAME REMOVED]", text)
    # Apply the same child-introduction rule used by local first-name memory.
    # Work backwards so replacing a span does not move an earlier match.
    for match in reversed(list(child_name_matches(text))):
        text = text[:match.start()] + "my child [NAME REMOVED]" + text[match.end():]
    text = re.sub(r"(?i:\b(?:my (?:home )?address is|i live at)\s+)[^\n.!?]+", "[ADDRESS REMOVED]", text)
    return text


def output_is_safe(answer: str, allowed_urls: set[str]) -> bool:
    """Reject obvious leaks, internal wording and invented executable actions."""
    if not answer.strip() or len(answer) > 2500 or len(answer.split("\n\n")) > 2:
        return False
    if "```" in answer or re.search(r"\bsk-[A-Za-z0-9_-]{10,}", answer):
        return False
    if re.search(r"\b(faq|question bank|retrieval scores?|system prompt|sample answers?)\b", answer, re.IGNORECASE):
        return False
    if any(email.casefold() != CONTACT_EMAIL for email in EMAIL_PATTERN.findall(answer)):
        return False
    for phone in PHONE_PATTERN.findall(answer):
        if re.sub(r"\D", "", phone) not in {"9500110931", "919500110931"}:
            return False
    for url in re.findall(r"https?://[^\s)\]>]+", answer):
        if url.rstrip(".,;!?") not in allowed_urls:
            return False
    if re.search(r"\bI (?:have |already )?(?:sent|connected|processed|issued|changed|deleted)\b", answer, re.IGNORECASE):
        return False
    # No confirmed commercial prices exist in the current support knowledge.
    if re.search(r"[₹$]|\b\d+\s*(?:rupees|INR)\b", answer, re.IGNORECASE):
        return False
    return True
