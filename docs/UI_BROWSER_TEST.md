# Current UI verification with Playwright

27 October 2026. Interface **homepage_chat_v1**, backend **phase9_answer_quality_v1**. The owner requested running the existing private homepage/chat, updating the README and adding fresh sample screenshots to the submission. No UI or agent code change was needed. This tests the private local interface connected to the same independent support runtime; it does not deploy the public repository as a website.

**Story:** a parent opens the homepage, opens Ask Octakidz, sends a question to the local Python API, receives a source-checked support reply, and reads it in the chat panel.

| Boundary | Actual result |
| --- | --- |
| Homepage and assets | All five image instances loaded. Desktop 1440 x 1000 had no horizontal overflow. Full homepage screenshot captured after lazy images loaded. |
| UI to Python API | All eleven unchanged regression questions and eight fictional conversation turns returned HTTP 200. |
| API to agent to UI | **11/11 fixed actions**, **8/8 focused actions**, and **19/19 exact displayed-answer matches**. Every returned backend version is the current version above. |
| Product explanation | Activities answer includes thinking examples, 2-3 of eight, the response-built interest map and CBSE tutor connection. The adventure illustration does not add an unconfirmed science offering. |
| Temporary memory | Fictional Mira/Tara introduction gets a warm greeting; recall and contextual follow-up work. Clear chat removes the name before the next recall. |
| Short-answer feedback | Local acknowledgment sets short mode; the same map answer goes from **72 to 53 words**, retaining both activity purposes and the CBSE/count details. One paired example, not a word-limit guarantee. |
| Unknowns/human route | No exact price is invented; the configured contact route is shown. Explicit human help displays a contact link without claiming a transfer. |
| Safety | Private-record and execution probes are refused; unrelated tasks are redirected. This is not comprehensive attack coverage. |
| Mobile/keyboard | At 390 x 844 the panel fits the viewport, with no horizontal overflow. Escape closes it and restores focus. |
| Browser state/errors | localStorage/sessionStorage empty; no observed console warnings/errors, page errors or failed requests in the normal tested flow. Resources load from the local origin only. |
| Private routes | Requested `.env`, source and log paths and traversal variants return 404. Browser URL normalisation can affect traversal variants; the separate existing raw-HTTP tests cover that boundary too. |
| Direct index URL | `/index.html` returns 200 through the running Python service. Its relative `/api/chat` request can work there because the backend is present. |

The fixed question wording/expectations were reused unchanged, with Clear chat between cases. A legacy note about absent conversation memory is historical; in this run a fresh/reset follow-up still correctly needs clarification. The eight focused turns test actual current memory separately. [Structured actual replies/checks](../evidence/browser_current/results.json).

Action/render matches are not a new semantic-accuracy score. Manual inspection found the intended product details, no tested policy fabrication, and correct safety/contact behaviour; some benefit wording remains firmer than cautious aims. Existing parser/redaction/model limits still apply. This step did not capture new provider call/token counts or inspect provider payloads, and it does not relabel the earlier 45-reply Python audit as browser evidence.

Four selected fresh screenshots are included: [homepage](ui/current/homepage-desktop.png), [activities chat](ui/current/activities-chat.png), [privacy refusal](ui/current/privacy-refusal.png), [mobile support](ui/current/mobile-support.png). They were visually checked and show no editor, key, environment file or actual student record. The two earlier screenshots remain separately labelled with their older backend. No UI source, raw brand assets or complete evidence folder is copied.

Opening a file directly or hosting only its static files does not supply the Python chat API. For the owner’s existing UI folder, run the Python service as explained in the main README. For a public hosted experience, see [Vercel hosting notes](VERCEL_HOSTING.md). No public chat URL or Vercel deployment is created by this test.
