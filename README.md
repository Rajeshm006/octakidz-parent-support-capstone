# Parent support agent - independent capstone

Python, LangChain and OpenAI support assistant for parents exploring Octakidz. It explains approved public product facts, searches by meaning, uses two bounded tools, remembers limited context during a chat, adapts answer length and offers human-contact guidance. Scenario 3: Customer Support; Track A: LangChain.

This repository contains the independently built capstone agent. Private Octakidz tutor/business code and the homepage implementation are excluded. The optional UI is documented through two screenshots only. No credentials or environment files are supplied.

## Run locally

Use Python 3.10+ (verified on macOS arm64, Python 3.14.4). Create an environment and install exact pins:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
```

Set `OPENAI_API_KEY` privately in your own environment or an untracked local `.env`. The optional `OPENAI_MODEL` defaults to `gpt-4.1-mini`. Never paste a real key into source, chat, a notebook or Git. No example environment file is included. A paid/quota-enabled OpenAI API account and network access are needed for real model calls.

```sh
python3 run_chat.py --check
python3 run_chat.py
```

The check makes no API requests. A missing local index is expected in a clean checkout: the first permitted search embeds the 30 public FAQ entries and creates an ignored cache. Do not copy a credential or cache from someone else's computer. An invalid/stale cache requires an explicit rebuild using `semantic_search.py --rebuild`.

Use `/short`, `/normal`, `/reset` and `exit`. `--debug` shows public tool/source stages and safe counters. Type support questions, never shell commands, at the chatbot prompt. The assistant cannot execute code, access student accounts, send email or make an actual human transfer.

```sh
.venv/bin/python -m unittest discover -s tests
.venv/bin/python verify_saved_evaluation.py
.venv/bin/python demo_submission.py --output-dir demo_run
```

Tests and saved-result scoring make no API calls. The last command runs five authored demo scenarios against OpenAI and preserves actual replies in a new folder. It must not overwrite existing evidence. Demo names are fictional; never substitute real student records. Temporary chat memory uses first names, up to three public topic IDs and a style flag, with reset and 600-second lazy expiry; it is separate from the product's private personal map. Metadata-only local logs are ignored by Git.

Public repository: [Rajeshm006/octakidz-parent-support-capstone](https://github.com/Rajeshm006/octakidz-parent-support-capstone). Submit the separately named course ZIP; the link is an additional reference.

## Submission documents

- [Problem framing, two-page PDF](docs/PROBLEM_FRAMING.pdf)
- [Five forced demo scenarios](docs/DEMO_SCRIPT.md) and [actual final demo](evidence/final_demo/RESULTS.md)
- [Evaluation report](docs/EVALUATION_REPORT.md), [every current reviewed reply](docs/QUALITY_REPLIES.md) and [same-test-set prompt comparison](docs/PROMPT_COMPARISON.md)
- [Engineering and product justification](docs/ENGINEERING_PRODUCT_JUSTIFICATION.md)
- [UI demonstration record](docs/ui/README.md)
- [Publication scope and exclusions](docs/PUBLICATION_SCOPE.md), [package verification](docs/PACKAGE_VERIFICATION.md) and [file manifest](FILE_MANIFEST.json)

The recorded 45-reply audit scores 10/10 supported, 5/5 missing-information and 5/5 safety cases against the fixed checklist. These known, assistant-reviewed cases meet draft checks; they are not population accuracy, independent expert validation, a grade or production certification. The final package has separate smoke/demo checks. Browser screenshots retain the older backend version from their actual run.

Cloud hosting and Vercel are optional and are not configured here. The assignment permits local deployment. Keep a future provider key server-side; never expose it through public browser environment variables. This repository grants no licence to proprietary Octakidz assets or code; the two illustrative screenshots are demonstration evidence only.
