# Requirements Assistant (Gradio + RAG)

Interactive requirements assistant built with Gradio. Upload a requirements/spec document, ask questions, and the app will retrieve context and either draft a Jira-style ticket or produce a compliance matrix using a Qwen model on OpenRouter. Local sentence-transformer embeddings plus ChromaDB keep everything in-memory for fast, lightweight retrieval.

## Features
- Gradio UI with multi-conversation history.
- File upload (.txt/.md/.json/.csv/.pdf) with on-the-fly chunking into an in-memory ChromaDB collection.
- Simple RAG summarizer that routes to one of two agents:
  - Jira ticket generator (JSON shape).
  - Compliance matrix generator (markdown table).
- OpenRouter model calls (defaults to `qwen/qwen3-4b:free`) with streaming responses.

## Prerequisites
- Python 3.10+ recommended.
- pip for installing dependencies.

## Setup
1. (Optional) Create/activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Provide an API key (preferred: OpenRouter):
   - Create a `.env` alongside `app.py` (auto-loaded on startup):
     ```
     OPENROUTER_API_KEY=sk-or-...
     ```
     `OPENAI_API_KEY` is also accepted as a fallback.

## Run the app
```bash
python app.py
```
Gradio will print a local URL. Open it, start a new conversation, optionally upload a file, and ask your question.

## Notes
- Embeddings use `zacCMU/miniLM2-ENG3` and store data in an in-memory Chroma collection; restart clears uploaded context.
- If you see 429 rate limits from OpenRouter’s free pool, add your own key or switch to a different model in `pipelines/requirements_pipe.py`.
