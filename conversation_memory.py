"""Local first-name context. Never send these values to models or log them."""

from dataclasses import dataclass, field
import re

from baseline_agent import normalise
from support_safety import FIRST_NAME, NOT_NAMES, child_name_matches, redact_contact_details


def recall_subject(question):
    text = normalise(question)
    if text in {"what is my name", "what s my name", "do you remember my name", "can you tell me my name"}:
        return "parent"
    if re.fullmatch(r"(?:what is|what s|do you remember|can you tell me) my "
                    r"(?:kid|child|son|daughter)(?: s)? name", text):
        return "child"
    return None


def joining_question(question):
    text = normalise(question)
    return (bool(re.search(r"\b(?:join|enrol|enroll|enrolment|enrollment|register|sign up)\b", text))
            and ("octakidz" in text.split() or bool(re.search(
                r"\b(?:this|your|the) (?:course|programme|program)\b", text))))


@dataclass(repr=False)
class ChatNames:
    parent: str | None = field(default=None, repr=False)
    child: str | None = field(default=None, repr=False)

    def clear(self):
        self.parent = self.child = None

    def available(self):
        return bool(self.parent or self.child)

    def introduce(self, question):
        matches = {
            "parent": re.search(r"\b(?:my name is|call me)\s+(?P<name>" + FIRST_NAME + r")\b",
                                question, re.IGNORECASE),
            "child": next(child_name_matches(question), None),
        }
        updated = []
        for subject, match in matches.items():
            if match:
                name = match.group("name")
                if len(name) <= 32 and name.casefold() not in NOT_NAMES:
                    setattr(self, subject, name.title() if name.islower() or name.isupper() else name)
                    updated.append(subject)
        return tuple(updated)

    def redact(self, question):
        safe = redact_contact_details(question)
        names = [re.escape(n) for n in (self.parent, self.child) if n]
        if names:
            # Existing removal markers are skipped, even if someone introduces
            # themselves as 'Name'. Known name mentions elsewhere are removed too.
            pattern = r"\[[A-Z ]+REMOVED\]|(?<!\w)(?:" + "|".join(names) + r")(?!\w)"
            def hide_known_name(match):
                if match.group(0).startswith("["):
                    return match.group(0)
                # Names such as Will/May can also be ordinary question words.
                # Keep clear modal uses; this is still a limited heuristic.
                if match.group(0).casefold() in {"will", "may"} and re.match(
                        r"\s+(?:i|we|you|my|our|octakidz|this|the|he|she|they)\b",
                        safe[match.end():], re.IGNORECASE):
                    return match.group(0)
                return "my child" if self.child and match.group(0).casefold() == self.child.casefold() else "the parent"

            safe = re.sub(pattern, hide_known_name, safe, flags=re.IGNORECASE)
        return safe

    def recall(self, subject):
        name = getattr(self, subject)
        if name:
            return ("You told me your name is " if subject == "parent" else
                    "You told me your child's name is ") + name + "."
        return ("I don't have your name in this chat. You can tell me your first name if you'd like."
                if subject == "parent" else
                "I don't have your child's name in this chat. You can share a first name if you'd like.")

    def introduction_only(self, question):
        prefix = r"(?:my name is|call me|my (?:kid|child|son|daughter)(?:(?:'s|’s)? name is| is called)?)"
        introduction = prefix + r"\s+" + FIRST_NAME
        # Consume the whole message: additional product questions or unsafe
        # requests must still follow the usual source/safety path.
        connector = r"(?:\band(?:\s+also)?\b|&)"
        separator = r"(?:[,;.]+\s*(?:" + connector + r"\s*)?|" + connector + r"\s*)"
        pattern = (r"\s*(?:(?:hi|hello|hey)[,!.\s]*)?" + introduction
                   + r"(?:\s*" + separator + introduction + r")*[,;.!\s]*")
        if not re.fullmatch(pattern, question, re.IGNORECASE):
            return False
        names = re.finditer(prefix + r"\s+(?P<name>" + FIRST_NAME + r")", question, re.IGNORECASE)
        return all(len(m.group("name")) <= 32 and m.group("name").casefold() not in NOT_NAMES
                   for m in names)
