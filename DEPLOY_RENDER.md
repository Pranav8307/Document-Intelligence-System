# Deploy to Render

This repository deploys as a FastAPI web service and a React static site. The
included `Dockerfile` builds the embedding model into the API image. Deploy the
API and frontend as two separate Render services; `render.yaml` is not needed
for this manual workflow.

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

## Create the API web service

1. In Render, select **New -> Web Service**.
2. Connect the GitHub repository and select the `main` branch.
3. Set the service name to `document-intelligence-api`.
4. Set **Language/Runtime** to **Docker**.
5. Set Dockerfile path to `./Dockerfile`.
6. Choose the Free plan if available. The API may need more memory because it
   loads PyTorch, FAISS, and SentenceTransformers.
7. Add health check path `/health`.
8. Create the service and wait for the Docker build to finish.

After deployment, copy the API URL and verify:

- `https://<api-service>.onrender.com/health`
- `https://<api-service>.onrender.com/docs`

If the first Docker deploy fails with `AttributeError: _ARRAY_API not found`
or `numpy.core.multiarray failed to import`, push the latest requirements file
and redeploy. The project pins NumPy 1.26.4 because the selected FAISS version
is not compatible with NumPy 2.x.

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
