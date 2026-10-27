# Can someone chat by opening index.html? Can Vercel help?

The HTML file is the front desk. The Python API and support agent do the work behind it. `chat.js` sends a relative request to `/api/chat`; the page needs that backend on the served origin.

| How it is opened | What is available |
| --- | --- |
| Double-click a local `index.html` file | The design may display, but the file does not start Python or provide `/api/chat`. This mode was explained from the inspected request path, not claimed as a new file-mode browser test. |
| Open `/index.html` through the running local Python service | Homepage plus the existing chat API; the served index and the real chat flow were verified with Playwright. |
| View the source on GitHub or deploy only static HTML | Source/static hosting alone does not execute this Python agent. The public repository deliberately omits the UI implementation. |
| Open a complete hosted application URL | Visitors can chat through its deployed backend, subject to the app's access settings. No such public URL is deployed in this step. |

## Vercel is an option

Vercel supports Python ASGI/WSGI applications, including FastAPI and Flask, and can combine a frontend with a Python backend. A supported application entrypoint is required. Our current `web_server.py` starts a local loopback `ThreadingHTTPServer`; it is not yet that deployment entrypoint. See the official [Python runtime guide](https://vercel.com/docs/functions/runtimes/python).

Set `OPENAI_API_KEY` as a **server-side Secret environment variable** in the Vercel project. Python can read it during function execution; the frontend should receive only replies, never the key. Do not put it in HTML/JavaScript, Git, build-time public variables or a `NEXT_PUBLIC_` value. Vercel's [environment-variable guide](https://vercel.com/docs/environment-variables) explains project/environment scoping and that changed values apply to new deployments. No key value is needed in this document or chat.

## Changes needed before hosting this version

1. Add an appropriate Python application adapter for the existing chat/reset routes and serve the selected frontend from the same origin. Keep the current support reasoning and safety gates.
2. Replace loopback-only Host/Origin checks with the exact permitted deployment origins.
3. Move bounded session state to a suitable shared store with expiry, or use another explicitly designed session mechanism. The current Python dictionary/locks are local to one process and are not a reliable multi-instance session design.
4. Package the public knowledge/index for read-only use, or use an appropriate external store. Put scratch work only in supported temporary storage. Ordinary local cache/log writes need an explicit hosting strategy: Vercel Functions have a read-only deployed filesystem and writable temporary space. [Runtime reference](https://vercel.com/docs/functions/runtimes).
5. Choose access and request limits for the demo, then verify browser/API/state behaviour on the actual deployment. A public chat can receive requests from visitors; hiding the provider key does not itself restrict who can use the backend.

This is a reviewed migration plan, not a deployment or a claim that setting one environment variable makes the current local server cloud-ready. We can keep the hosting project's repository private and deploy only the selected UI/support application. Original tutor, account, assessment and other business implementations remain excluded.

One visibility distinction matters: a private repository can protect its source history, but a browser must receive a page's HTML, CSS and JavaScript to display it. Visitors can inspect those frontend assets. API keys and sensitive business logic must stay on the backend; they cannot be hidden inside frontend code.
