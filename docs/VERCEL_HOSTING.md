# Opening index.html and using the hosted demonstration

The HTML page is the front desk; a Python API runs the assistant. The browser sends chat requests to `/api/chat` on the served origin. Opening a file by itself does not start that API.

| How it is opened | What is available |
| --- | --- |
| Double-click a local index.html file | The design may display, but Python and `/api/chat` do not start. This is explained from the inspected request path, not claimed as a new file-mode browser test. |
| Serve the owner's private UI with the local Python service | Homepage plus the existing local chat API. The public repository's independent CLI instructions are separate. |
| View GitHub or deploy static HTML alone | GitHub/static hosting does not execute this Python agent. UI/server implementation is intentionally excluded from the public repository. |
| Open the complete protected hosted preview | Selected frontend, FastAPI backend, the same Python/LangChain agent, OpenAI and Redis temporary context. Actual browser flow is verified; authorised Vercel sign-in is required. |

## Working hosted preview

[Open the Octakidz preview](https://octakidz-parent-support-demo-2om0ih0ue-rajeshm006s-projects.vercel.app). It is protected by Vercel Authentication and is not an anonymous public chat. Reviewers should use the included [screenshots/recorded results](HOSTED_DEMO.md) or the independently runnable terminal agent if they cannot sign in. [All submission links](../SUBMISSION_LINKS.md).

The private hosting adapter exports a supported FastAPI application. Its server-only OpenAI Secret and Redis credentials are configured in the preview environment, never shipped in HTML/JavaScript, the repository or the ZIP. Redis holds only the validated temporary names/topic/style/count context with a 600-second idle TTL; reset deletes it. The approved public FAQ vectors are read-only, so normal requests do not rebuild the index or write deployed knowledge files. Exact-origin checks, secure cookies, static selection and bounded requests preserve the HTTP boundary. See the official [Vercel FastAPI guide](https://vercel.com/docs/frameworks/backend/fastapi) and [server environment-variable guide](https://vercel.com/docs/environment-variables).

The [actual hosted report](HOSTED_DEMO.md) distinguishes 192 private-workspace offline checks, 159 core tests in this public package, nineteen hosted browser replies and separate earlier local evidence. It records known limits and unsuccessful setup attempts. Two unusable setup deployments were removed; only the verified protected preview is retained. The hosted adapter was tested on Vercel Linux/Python 3.14. No business-production release, public access, comprehensive security certification or new general accuracy result is claimed.

A private repository can protect source history, but visitors can inspect browser-delivered HTML/CSS/JavaScript. Credentials and sensitive tutor/business logic belong on the backend. This independent support demonstration does not implement or access the private tutor/map/business application. Only selected screenshots, links and source-free hosted evidence are added to the submission; its UI/hosting code and original private product code stay excluded.
