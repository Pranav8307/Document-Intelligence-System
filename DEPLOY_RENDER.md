# Deploy to Render

This repository deploys as a FastAPI web service and a React static site. The
`Dockerfile` installs the API with lightweight TF-IDF retrieval (no PyTorch or
embedding model). The included `render.yaml` can create both services and a free
PostgreSQL database; the manual steps below achieve the same result.

## Before deploying

1. Create a private GitHub repository and push this project to it. Do not push
   the local `.env` file.
2. Create a Render account and connect it to GitHub.
3. Decide whether this is a demo or a real LLM application:
   - Demo: keep `USE_MOCK_LLM=true`; no AI-provider key is needed.
   - Real answers: set `USE_MOCK_LLM=false`, then set provider, model, base URL,
     and the matching API key as Render secrets.
4. The API uses lightweight TF-IDF retrieval so it can run on Render's free
   memory tier without PyTorch, CUDA, or a sentence-transformer model.

## Create the API web service

1. In Render, select **New -> Web Service**.
2. Connect the GitHub repository and select the `main` branch.
3. Set the service name to `document-intelligence-api`.
4. Set **Language/Runtime** to **Docker**.
5. Set Dockerfile path to `./Dockerfile`.
6. Choose the Free plan. The API fits the free memory tier because it uses
   TF-IDF retrieval instead of PyTorch or SentenceTransformers.
7. Add health check path `/health`.
8. Create the service and wait for the Docker build to finish.

After deployment, copy the API URL and verify:

- `https://<api-service>.onrender.com/health`
- `https://<api-service>.onrender.com/docs`

History: the first deployed version used SentenceTransformers + FAISS. On the
free tier it failed with a FAISS/NumPy compatibility error and then exceeded the
memory limit (PyTorch + CUDA libraries), so retrieval was switched to TF-IDF on
10 Sep 2026. If you see `numpy.core.multiarray failed to import`, check that
`numpy==1.26.4` from `requirements.txt` is the version installed.

If Render reports that the service exceeded its memory limit, confirm the latest
Dockerfile and `requirements.txt` are deployed.

## Create the frontend static site

1. In Render, select **New -> Static Site**.
2. Select the same GitHub repository and `main` branch.
3. Set the site name to `document-intelligence-web`.
4. Set **Root Directory** to `frontend`.
5. Set build command to `npm install && npm run build`.
6. Set publish directory to `dist`.
7. Add this environment variable, replacing the value with your actual API URL:

```text
VITE_API_URL=https://<api-service>.onrender.com
```

8. Choose the Free plan and create the site.

The frontend URL will look like:

```text
https://<frontend-service>.onrender.com
```

## Connect the two services

Open the API service's **Environment** settings and add:

```text
FRONTEND_ORIGINS=https://<frontend-service>.onrender.com
```

Save and redeploy the API. Then open the frontend URL and test upload and Q&A.

The frontend is optional: FastAPI Swagger at `/docs` remains available for
developer testing. The React site is the normal user interface.

The frontend is written entirely in JavaScript and JSX. There are no
TypeScript files or TypeScript build steps.

## Updating the hosted app

Enable **Auto-Deploy: Yes** in each separate Render service. After changing
code:

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
| `LLM_PROVIDER` | `groq` (as deployed) or `openai` |
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

## Persistence and limitations

If `DATABASE_URL` is set (the included `render.yaml` wires a Render PostgreSQL
database), uploaded documents and chunks are stored in PostgreSQL and the TF-IDF
indexes are rebuilt from them on startup. Without `DATABASE_URL`, everything is
kept in process memory and disappears on restart.

Always in memory (reset on restart, not shared across instances): the answer
cache (5 minutes), the rate limiter, and the `/metrics` counters. There is no
authentication or per-user isolation. This is suitable for a single-instance
demo, not for sensitive documents.
