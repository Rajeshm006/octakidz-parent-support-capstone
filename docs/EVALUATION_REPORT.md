# Evaluation and failure analysis

27 October 2026. Current backend: `phase9_answer_quality_v1`; launcher: `phase8_launcher_v1`. Scores are explicit assistant judgments on authored development questions, not independent expert ratings, held-out reliability or a predicted grade.

## Design and results

The frozen scorecard defines fully acceptable as the expected action, all essential points, grounded product facts, safe behaviour, clear/useful wording and any declared state check. It does not equate a source ID, similarity score, keyword match or action label with correctness. Cases and criteria remain unchanged across the reported quality iterations.

| Group | Final saved acceptance | Earlier recorded comparison |
| --- | --- | --- |
| Supported core | 10/10 | Previous full audit 8/10; not a fresh full old-version arm. |
| Missing-information core | 5/5 | No invented price, trial, coverage or policy. |
| Safety/privacy core | 5/5 | No tested record disclosure, execution, prompt leak or invented action. |
| Legacy fixed questions | 11/11 | Same original questions retained. |
| Additional repeat turns | 6/6 | Three anchors each total three attempts including their first core attempt. |
| Fictional conversation | 8/8 | Names/topic, short preference and reset expectations. |

All three repeated anchors retain their essential-point checklists across three attempts. This is exploratory consistency, not a reliability estimate. The proposed draft 9/10 supported, 5/5 missing and 5/5 safety checks are met in this set. Do not pool duplicated turns into a unique-question sample. [Every answer and review](QUALITY_REPLIES.md), [saved data](../evidence/quality/results.json), [scorecard](../QUALITY_SCORECARD.md).

The final 45-reply audit used 86 LLM requests, 32 query embeddings and no index rebuild, with 121,252 LLM input / 4,236 output tokens and 596 embedding tokens. Median model-turn latency was 4.96 seconds with a warm index. No response-time release target or currency-cost estimate is claimed. Its 572 inspected fixture/key absence assertions pass; these correlated probes are not comprehensive privacy certification.

## Before/after root causes and fixes

| Failure | Evidence-supported cause | Fix and actual comparison |
| --- | --- | --- |
| Baseline misses a story-based paraphrase and cannot interpret an absent follow-up topic | Exact wording lookup has no semantic representation or conversation state. | Historical LLM comparison answers the paraphrase; local clarification prevents a stateless follow-up guessing a topic. [Recorded clarification fix](../evidence/clarification_fix.json). |
| Related retrieval omits a fuller activity account | Complete source 003 was below the five-candidate cutoff in diagnostic rankings, despite a related source being strong. Similarity is not coverage. | One authored related-reference link can reserve an existing slot when its real score passes the floor, protecting the strongest and unknown-detail entries. No extra embedding request. |
| Retrieved full account is left out, or selected details disappear | Traces separate source-picking omissions from generation omissions. | Checker prefers the most complete relevant account; parent instructions retain distinctions, clear illustrations and conditional team help. A selected-reference reminder repeats only approved facts. |
| Vague fractions, missing team guidance, unconfirmed science implication | Final text can omit support details or treat a plausible example as a new capability. | Fresh six-question normal/short comparison: fully acceptable 8/12 to 12/12; essential points 35/38 to 38/38. [Before](../evidence/quality/before_focus.json), [after](../evidence/quality/after_focus.json), [explicit judgments](../evidence/quality/focus_review.json). |
| Unavailable function or rewritten query | Authored injected calls bypass a trustworthy tool-choice assumption. | Python blocks unallowlisted names/altered query arguments, counts attempts, then returns safe contact guidance within limits. [Injected controls](../evidence/tool_controls.json). These are not real model errors. |

The two final prompts changed together, so the focus comparison cannot assign causality to either individually. Fresh pre-change controls already passed some previously failing phrasings: output variability is retained rather than attributed entirely to the fix. The public FAQ removes internal provenance metadata, while its 30 runtime Q+A/paraphrase/status/link records remain unchanged; a separate clean-package demo checks this publication candidate.

## Required same-set prompt comparison

The historical experiment uses the same 20 cases, same facts and shared boundaries across `v1_simple`, `v2_grounded` and `v3_parent`. It records 60 outputs: 51 actual OpenAI replies and nine local-rule replies. Strict routing is 17/20 for all variants, not answer accuracy. [Complete comparison table](PROMPT_COMPARISON.md) includes actual outputs and improved/worsened notes.

v1 is concise but loses activity detail; v2 preserves more map/CBSE detail but suggests an unconfirmed free-trial premise; v3 has the preferred parent-facing voice and more cautious trial wording but still loses details and guessed an absent topic. v3 was selected with these limits. These are historical `phase3_llm_v1`/28-entry results, not a rerun of current prompts or proof of statistical superiority. [Structured original comparison](../evidence/prompt_comparison.json).

## Retrieval, tools, memory and adaptation evidence

[Historical retrieval comparison](../evidence/retrieval_comparison.json) records actual full-context/RAG replies, selected sources and content notes. Strong similarity can pass for a nonexistent service; the support checker and unknown-detail handling are therefore separate. Core code uses LangChain model binding/structured outputs, not a claim of multi-agent autonomy.

The two typed support tools are search and configured human-contact guidance. Python validates names/arguments, original query integrity and budgets. Tool controls include failed calls, safe recovery and a backend-failure simulation. A simulated timeout is not a real provider outage.

The current saved conversation and final demo show local first-name recall, bounded public topic reuse and reset. Topic state never becomes a student's personal map. [Feedback evidence](../evidence/feedback.json) records actual earlier normal/short comparisons of 71 to 47, 87 to 43 and 64 to 45 words with the same selected sources. Style acknowledgment is local, and only a style enum affects later generation; no model weights or shared learning are updated.

## Safety, ethics and limitations

No actual student records, private tutor, account actions, execution, payments or email tools exist. Names are replaced with roles before provider/tool use and are added locally; inspected metadata excludes fixture identifiers. Redaction and introduction parsing are heuristic and incomplete. Do not invite families to provide sensitive details or claim universal privacy protection.

Marks, admission and career outcomes are not guaranteed. Activity-based interests may inform tentative exploration while respecting a child's wishes and changing preferences; they are not a clinical brain scan or reliable occupational prediction. Natural Intelligences are the owner's programme framework, not a diagnosis or demonstrated treatment effect. Parent trust requires cautious benefit wording and confirmed product facts, not persuasion through invented offers.

Known limits remain: awkward grouping in one football example, imperfect adherence to requested numeric example detail, soft word limits, model variability and occasional firm benefit phrasing. The final set was influenced by development. Independent parent/domain review, unseen paraphrases, broader prompt-injection/PII tests, threshold calibration and external-machine checks are next steps.

## Reproducibility and roadmap

Exact pins and 159 offline core checks accompany the source. The earlier fresh-venv run verified 39 pins on macOS arm64/Python 3.14.4, without copying a key/cache/log, followed by actual CLI replies; [summary](../evidence/clean_install_summary.json). The original workspace also had 14 UI tests, but that UI implementation is intentionally absent here. The candidate's fresh offline tests, startup and actual ten-turn/five-scenario demo are separate [package checks](PACKAGE_VERIFICATION.md).

Next priorities are unseen questions and independent review, then production authentication/rate limits/session infrastructure and policy-confirmed integration if desired. Local deployment satisfies this capstone demonstration. Public GitHub code is the independent educational support agent; private product/UI code and credentials remain excluded.
