# Engineering and product justification

## Problem and selected architecture

A parent needs accurate public product explanations, follow-up context and an honest route to the team. A single exact-match FAQ cannot handle the variety of parent wording. A bounded support pipeline makes useful decisions while keeping private account/tutor functions outside its reach.

Track A uses **LangChain**: `ChatOpenAI`, structured model outputs and native bound tool schemas. Python owns validation, state and request budgets. The product is an independent support agent, not the academic tutor or a private student-map engine. The public UI is documented with screenshots, with its implementation excluded from this source release.

The request path is: local safety/commands and minimisation; bounded topic context; one model tool-choice call; Python tool validation; semantic search; one source-support check; one grounded answer; independent output/source guards; local name presentation; metadata-only logging. Branches may clarify, refuse or return human guidance earlier. This is explicit conditional planning, not an unbounded autonomous loop or a claim that hidden model reasoning is exposed.

## Decisions and tradeoffs

| Choice | Reason and practical limit |
| --- | --- |
| 30 public Q+A chunks | Small, reviewable knowledge scope; no proprietary code or student records. Internal provenance locations are removed from the public copy, while runtime facts stay the same. |
| OpenAI embeddings plus exhaustive cosine search | Easy to inspect at this scale; 1536 dimensions, Q+A representation. A new vector database adds little for 30 entries. This is not a scalability claim. |
| Experimental 0.35 floor and related-reference links | Filter weak matches and preserve a fuller account within the five-candidate pool. The score is not confidence or factual support; the floor still needs wider calibration. |
| At most five candidates, three selected sources | Limits context and cost while permitting multi-source answers. An independent checker decision must use supplied IDs and honest support status. It may still omit a necessary source. |
| Separate source checking and answering | Related vocabulary cannot confirm a policy or capability. The answerer only sees selected support. Two stages cost more latency/tokens and both remain model-based; Python validates their shapes/IDs, not complete semantic entailment. |
| Two typed tools | `search_support_knowledge` and `get_human_support`; no executor, records or transactional tool. Contact provides a configured route and never claims a completed transfer. |
| Bounded tool/model budget | One router, at most one checker and one answerer, at most two tool attempts. Invalid attempts count; repeats/parallel calls and query rewriting are rejected. Provider retries are disabled. |
| Temporary minimal state | Up to three public topic IDs, parent/child first names and a style enum; no transcript, age, interests or real personal map. Explicit reset/exit and 600-second lazy expiry. Heuristics miss some introductions and identifiers. |
| Local style feedback | Whole-message commands become a trusted normal/short enum. No raw feedback is promoted to instructions or stored as training data. Brevity is a soft goal, with essential qualifications preserved. |
| Configurable gpt-4.1-mini | Existing tested cost/latency role retained; explicit instructions and structured schemas support small-model operation. No claim it is the best/latest model. Aliased provider models and network timing can vary. |

## Safety and observability

Allowlisted schemas, original-question checks, source/status validation, output checks and fixed contact guidance complement prompt instructions. Local refusals handle recognised execution/private-record requests before provider calls. Unknown policies, price/trial terms and sensitive cases escalate without fabrication. Parent/model strings never become shell commands.

The provider key comes from an untracked local environment file or process environment. No credential or environment file is published. Provider requests use the explicit official endpoint, a timeout, no retries, `store=False` and disabled external tracing. This API setting and minimal local state are not promises of zero provider retention.

Operational logs contain time/version, outcome, public IDs, call/token counts, latency, error codes and safe state flags. They exclude questions, answers, names, credentials and raw provider errors. Authored demo/evaluation transcripts are separately labelled fictional evidence, not production conversation logs. Source traces show public facts and decisions rather than private chain-of-thought.

## Deployment and product limits

The independently runnable terminal agent is the publication target. Exact Python pins and clean-start guidance make it reviewable. A clean checkout has no cache: its first authorised search creates the public FAQ index. Stale/corrupt cache fails with explicit rebuild guidance; provider errors offer safe contact help. Logging failure does not block an ordinary local answer.

The local homepage was a separate interface demonstration. Historical screenshots prove that recorded browser flow under its labelled backend, not a current hosted product or answer-quality score. UI source, original assets and private business integration are excluded here.

No cloud deployment is needed for the assignment's local option. A production website would require separate hosting decisions, authentication, abuse controls, session storage, dependency/secret scanning, confirmed privacy/policy processes and independent product validation. No automatic Vercel deployment or browser-visible provider key is configured.

The business goal is a parent's informed choice, not purchase pressure. Suitability and learning benefits must be explained as intended outcomes, with commercial terms confirmed by the team. The natural-intelligence/interest map must leave room for change rather than confine a child to a label. User trust, tested limits and transparent escalation take priority over apparent complexity.
