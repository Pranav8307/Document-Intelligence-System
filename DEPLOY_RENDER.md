# Deploy to Render

This repository deploys as a FastAPI web service and a React static site. The
included `render.yaml` and `Dockerfile` build the embedding model into the API
image and build the frontend from `frontend/`.

## Before deploying

1. Create a private GitHub repository and push this project to it. Do not push
   the local `.env` file.
2. Create a Render account and connect it to GitHub.
3. Decide whether this is a demo or a real LLM application:
   - Demo: keep `USE_MOCK_LLM=true`; no AI-provider key is needed.
   - Real answers: set `USE_MOCK_LLM=false`, then set provider, model, base URL,
     and the matching API key as Render secrets.
4. Choose at least 2 GB of memory. This service loads PyTorch and a sentence
   embedding model, so a 512 MB instance is not an appropriate production size.

## Create the Render service

1. In the Render dashboard, select **New** then **Blueprint**.
2. Select the GitHub repository and branch.
3. Render finds `render.yaml`. Confirm the service name and select the region
   closest to most users.
4. Pick a compute plan with at least 2 GB RAM.
5. Create the Blueprint and watch the first deploy log. The Docker build
   downloads the embedding model once; subsequent starts use the image copy.
6. When Render marks the deploy live, open:
   - `https://<your-service>.onrender.com/health`
   - `https://<your-service>.onrender.com/docs`
   - `https://document-intelligence-web.onrender.com`

The frontend is optional: FastAPI Swagger at `/docs` remains available for
developer testing. The React site is the normal user interface.

The frontend is written entirely in JavaScript and JSX. There are no
TypeScript files or TypeScript build steps.

## Updating the hosted app

Render is configured with `autoDeploy: true` for both the API and frontend.
After changing code:

1. Run the local checks below.
2. Commit and push to the branch connected to Render, usually `main`.
3. Render automatically starts a new deploy for both services.
4. Watch the deploy logs, then test `/health`, `/docs`, and the frontend.

If you change `VITE_API_URL`, `FRONTEND_ORIGINS`, provider keys, or other
environment values, update them in Render's Environment tab and redeploy. Do
not put secrets in React or GitHub.

## Render environment variables for real answers

Set these in the service's **Environment** tab. Mark API keys as secrets.

| Variable | Example value |
| --- | --- |
| `USE_MOCK_LLM` | `false` |
| `LLM_PROVIDER` | `openai` or `groq` |
| `LLM_MODEL` | the model name from your AI provider |
| `LLM_BASE_URL` | provider API base URL when required |
| `OPENAI_API_KEY` | your OpenAI key when `LLM_PROVIDER=openai` |
| `GROQ_API_KEY` | your Groq key when `LLM_PROVIDER=groq` |

Never place a provider key in `render.yaml`, source code, browser JavaScript, or
GitHub. Set it only in Render's Environment tab.

## Verify after deployment

1. `GET /health` must return HTTP 200.
2. In `/docs`, call `POST /documents/upload` with a small PDF or TXT file and
   copy the returned `document_id`.
3. Call `POST /qa/ask` with that ID and a question about the document.
4. With real-LLM mode enabled, confirm the answer is not prefixed with
   `[MOCK ANSWER]`.

## Run the frontend locally

From the repository root:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. It uses `http://127.0.0.1:8000` by default. To
use another API, create `frontend/.env.local` with
`VITE_API_URL=https://your-api.onrender.com`.

## Important current limitation

Documents, vectors, cached answers, and metrics are stored only in the running
process. They disappear when the service restarts, redeploys, or scales beyond
one instance. This is suitable for a single-instance demo, not durable product
storage. A production version needs persistent object storage for uploaded files
and a database/vector store for document text, embeddings, and cache data.
