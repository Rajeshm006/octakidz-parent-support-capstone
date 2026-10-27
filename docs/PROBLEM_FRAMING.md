# Problem framing

**Author:** Rajesh Manoharan. **Project:** Independent parent-support agent. **Scenario:** Customer Support. **Track:** A, LangChain. **Date:** 27 October 2026.

## User, problem and value

The primary user is a parent of a child aged 10-14, generally Grades 6-9, exploring Octakidz. The parent wants to understand the thinking activities, their connection to the AI tutor, suitability, purchase details and how to get help. Questions arrive with spelling mistakes, informal wording and follow-ups. A fixed question bank can miss their meaning, forget the topic or give an irrelevant answer.

The support agent explains approved public information, acknowledges uncertainty and offers the configured human-contact route. Its business purpose is to support an informed decision through understanding, without pressure, invented policies or guaranteed outcomes. It does not perform academic tutoring, evaluate a real child or implement the proprietary product.

## Product explanation and workflow

Activities have two purposes. Responses build an evolving personal map of interests, likes, dislikes and familiar stories, which guides interest-based CBSE explanations by the academic tutor. Each activity also practises thinking through 2-3 of the programme's eight Natural Intelligences. These are intended learning benefits, not measured improvement, a diagnosis or a fixed learning type.

The parent asks a question; local checks handle recognised unsafe requests or conversation commands. Otherwise the model chooses a permitted support tool. Search retrieves related approved facts, a separate check selects actual support, and generation answers only from those facts. Missing or partial information leads to human guidance; unclear context leads to clarification. The parent may follow up, request shorter answers, reset, or choose to contact the team. Contact guidance does not send a message or perform a transfer.

## Inputs, outputs and constraints

Inputs are the parent question, minimised temporary context and 30 curated public support entries. Outputs are a concise grounded answer, clarification, refusal or configured support guidance. Source identifiers and safe counters support developer inspection; implementation details are absent from ordinary parent replies.

The independent Python/LangChain/OpenAI implementation has two allowlisted tools, at most three model calls and two tool attempts per question. It has no code interpreter, student-record access, payment/account action or real handoff tool. Only recognised first names, up to three public topic IDs and an answer-length flag are retained temporarily; reset/exit clear state and unused state expires lazily after 600 seconds. Logs exclude raw conversations and identifying values. Heuristic filtering has limits.

## Success criteria and evaluation

Before the quality audit, a fixed checklist defined usefulness as addressing the question with all essential supported points, safe behaviour, clear language and correct conversation state. The proposed draft targets are at least 9/10 supported cases, 5/5 missing-information cases, 5/5 safety cases and no observed fixture leak at inspected boundaries. These are development checks, not course-grade predictions or population accuracy.

The evaluation uses 20 core cases, the fixed legacy eleven, six additional repeat turns and eight fictional conversation turns: 45 replies. Three anchors receive three attempts each. Prompt comparison uses the same 20 historical cases across three variants. Record actual answers, selected sources, latency, calls, explicit semantic judgments and failures; do not treat action agreement as accuracy. The latest recorded core results are 10/10, 5/5 and 5/5; broader independent review remains future work. No latency pass threshold was specified.

## Five example parent questions

1. My child is 12. What would they do with Octakidz?
2. How do activity answers help the AI tutor teach my child?
3. What would a familiar story-based maths example look like?
4. What is included in a plan and what is the exact price?
5. That still does not answer my concern. Can I speak to the team?

## Anticipated failures and limits

Related text can lack the requested fact: require actual source support, preserve unknown-policy flags and offer human help. A correct source can still produce an incomplete answer: preserve essential distinctions and evaluate actual text. Follow-ups can lose context: retain bounded public topics and clarify after reset. Claimed parent identity cannot authorise private records. Refuse code execution and boundary overrides. Provider/setup failures use safe guidance and metadata-only error codes.

The capstone assumes the tutor and support experience launch together; it makes no new coverage, price, trial, retention, marks or career guarantee. Local deployment is sufficient for demonstration. The public submission contains independent capstone code and approved support facts, plus UI documentation/screenshots only; original business/UI implementation, private tutor, actual student data and credentials are excluded. Production hosting and private product integration remain separate work.
