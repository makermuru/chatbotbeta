# Pseudo-AI Chatbot

`main.py` is the FastAPI backend for Render. `docs/` is the static GitHub Pages frontend.

## Deploy

1. Push the contents of this directory to a GitHub repository.
2. In Render, create a **Blueprint** from the repository. It reads `render.yaml`.
3. Set Render environment variable `ALLOWED_ORIGINS` to your exact Pages URL:
   `https://GITHUB_USERNAME.github.io/REPOSITORY_NAME`
4. Put your Render URL in `docs/app.js` as `API_BASE_URL`, then push again.
5. In GitHub **Settings > Pages**, choose **GitHub Actions** as the source.

## Database

The bundled SQLite files provide initial data. Render's default filesystem is temporary, so answers saved through `/teach` disappear after a restart or redeploy unless you attach a persistent disk or move to Postgres.

## Render free-instance mode

Render's free instance runs the lightweight SQLite keyword search by default.
Do not add PyTorch or `sentence-transformers` to `requirements.txt`. On a
larger server, install `requirements-embeddings.txt` and set
`ENABLE_EMBEDDINGS=true` to activate semantic search.
