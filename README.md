# Adaptive RAG Chat — README

A Streamlit app that mixes **RAG over your documents**, **tool-augmented agents**, and **multimodal outputs** (images & charts). Upload PDFs/TXT, the app indexes them to Pinecone, auto-generates concise summaries, and lets you chat in three modes:

- **LangGraph Engine** (custom state graph: RAG + chart/image routing + in-text citations)
- **OpenAI Agent** (function tools: RAG, citations, charts, images)
- **LangChain Agent** (ReAct with structured tools)



## Features

- **Document ingestion**: Split PDFs/TXT into chunks and upsert to Pinecone, per-file namespaces.
- **Auto summaries**: Generates a short summary per uploaded file (stored in `summaries/`).
- **Three chat modes**:
  - **LangGraph** pipeline (routing: image/chart/RAG → retrieval → grading → re-query → final answer → add citations).
  - **OpenAI Agent** with function tools (RAG, `add_citations`, QuickChart URL builder, DALL·E image generation).
  - **LangChain Agent** using ReAct with the same toolset.
- **Citations**: Answers can be annotated with `[filename.ext]` after each paragraph.
- **Charts**: Builds **QuickChart** URLs from natural language; includes URL normalizer for base64/JSON configs.
- **Images**: Generates images locally via OpenAI Images; renders them in chat.
- **Multi-chat**: Create multiple chats, switch, and view tool traces/steps.

---

## Architecture

**Core flow (LangGraph Engine)**

1. Classify request → *needs image?* → *needs chart?* → else proceed.
2. If files are attached → decide **use RAG** from summaries → retrieve top-k chunks from Pinecone.
3. **Grade** chunks & summaries → either answer, or **transform query** and re-retrieve.
4. Generate final answer → **add in-text citations**.
5. If image or chart was requested, return that directly.

**Agents**

- **OpenAI Agent**: custom lightweight agent that calls function tools.
- **LangChain Agent**: ReAct agent via `langgraph.prebuilt.create_react_agent`.

---

## Project Structure

```
.
├─ streamlit_app.py            # UI, chat manager, file upload, chart URL normalization, render logic
├─ chat_engine.py              # LangGraph engine: nodes, routing, citations, DALL·E
├─ chat_engine_langchain.py    # LangChain ReAct agent + tools (RAG, chart, image, citations)
├─ chat_engine_openai.py       # OpenAI Agent (function tools) + tools
├─ create_index.py             # PDF/TXT loader, splitter, upsert to Pinecone
├─ pinecone_client.py          # Pinecone init + index factory
├─ rag_summary.py              # Simple vector store wrapper + summary RAG helper
├─ uploads/                    # (created at runtime) uploaded files
├─ summaries/                  # (created at runtime) auto-generated summaries per file
└─ generated_images/           # (created at runtime) images from DALL·E
```

---

## Prerequisites

- Python 3.10+
- A Pinecone account & API key
- An OpenAI API key
- (Optional) A virtual environment

---

## Installation

```bash
git clone <your-repo-url>
cd <your-repo-dir>

# (recommended) create venv
python -m venv .venv
# Windows:
. .venv/Scripts/activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

If you don’t have a `requirements.txt` yet, start with:

```txt
streamlit
python-dotenv
openai>=1.0.0
pinecone-client
langchain
langchain-community
langchain-openai
langgraph
pydantic
```

> Adjust versions as needed for your environment.

---

## Configuration

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=developer-quickstart-py
```

Notes:

- `pinecone_client.py` will create the index if missing (`developer-quickstart-py` by default) and configure the embedding field map for `chunk_text`.
- The app saves generated images to `generated_images/`, and summaries to `summaries/`.

---

## Running the App

```bash
streamlit run streamlit_app.py
```

Open the URL shown in your terminal (typically `http://localhost:8501`).

---

## Using the App

1. **Upload files**  
   Use the sidebar to upload **PDF** or **TXT**. Each file:
   - is ingested to Pinecone (its **namespace** is the filename),
   - gets an auto-generated summary saved into `summaries/<file>.txt`.

2. **Create or select a chat**  
   Click **New Chat** in the right column, then select it from the list.

3. **Choose a mode** (sidebar radio):
   - **LangGraph Engine**
   - **OpenAI Agent**
   - **LangChain Agent**

4. **Ask a question**  
   Optionally select one or more uploaded files to attach (enables RAG).  
   Ask your question in the input field and press **Send**.

5. **View results**  
   - If the answer is a **QuickChart** URL, the app will render the chart.
   - If the answer is a **local image path**, the app displays the image.
   - For RAG answers, expand **Intermediate Steps / Tool Calls** to inspect traces.

---

## How It Works (under the hood)

### Indexing & Summaries
- `create_index.py` loads & splits documents (recursive splitter with overlap) and upserts chunks:
  - `namespace = filename`
  - field `chunk_text` contains the chunk content
- `rag_summary.py` queries top-k chunks and asks a model to produce a concise summary, saved in `summaries/`.

### Retrieval & Grading
- `chat_engine.py` retrieves from Pinecone via the custom `pinecone_client.index.search`.
- Chunks are **graded** (kept/rejected) by an LLM to ensure only relevant context is used.

### Citations
- `add_intext_citations` inserts `[filename.ext]` per paragraph based on file summaries and attached namespaces.

### Chart Generation
- Tools produce **QuickChart** URLs.  
- `streamlit_app.py` includes `normalize_quickchart(...)` to:
  - accept both **URL-encoded JSON** (preferred) and **base64** configs,
  - set `encoding=base64` when needed,
  - sanitize/query-encode configs.

### Image Generation
- Images are created with OpenAI Images and stored under `generated_images/`, then displayed in chat.

---

## Troubleshooting

### QuickChart “error at character 23”
- Cause: using **raw** (not URL-encoded) Chart.js config with single quotes or spaces in the `c` param.
- Fix: always **URL-encode the JSON** config or return a proper `encoding=base64` param.  
  The agent tools are instructed to produce valid URLs, and `normalize_quickchart` helps sanitize incoming URLs.

### Image path returned but not shown
- Ensure the tool returns an **absolute path** to an existing file.
- The app displays:
  - any `https://...` image URL,
  - any existing local file path,
  - otherwise falls back to rendering text.

### `MediaFileStorageError: Error opening ''`
- Indicates an **empty** or invalid path was passed to `st.image`.
- Ensure your tool never returns an empty string. The included tools guard against this; double-check any custom tool code.

### Windows path issues
- Local paths are matched via regex `([A-Za-z]:\\S+)` and rendered with `st.image`.  
  Make sure the returned path exists and uses backslashes (or provide an absolute POSIX path when running WSL).

### Pinecone not found / wrong index
- Check `.env` for `PINECONE_API_KEY` and `PINECONE_INDEX_NAME`.
- The starter index name is `developer-quickstart-py`. Change in both `.env` and `pinecone_client.py` if needed.

---

## Development Notes

- **Models**: default chat LLM is `gpt-4o` (temperature 0); lightweight classification uses `gpt-4o-mini`.
- **Images**: generated with `gpt-image-1`, saved under `generated_images/`.
- **Namespaces**: per-file; attach files in the UI to enable **RAG** on those namespaces.
- **History**: per chat tab; internal traces are visible via expanders.
- **Extending tools**: add new function tools in `chat_engine_openai.py` / `chat_engine_langchain.py` and register them with the agent.

---

## Security Notes

- Never commit your `.env`. Use environment variables in production.
- Uploaded files, summaries, and generated images are stored locally—treat them as sensitive if your data is sensitive.
- Review any additional telemetry or logging before deploying.

---

## License

MIT (or your preferred license). Add a `LICENSE` file to the repo.

---

### Quick Start (TL;DR)

```bash
# 1) Install
pip install -r requirements.txt

# 2) Configure
echo "OPENAI_API_KEY=sk-..."            > .env
echo "PINECONE_API_KEY=..."            >> .env
echo "PINECONE_INDEX_NAME=developer-quickstart-py" >> .env

# 3) Run
streamlit run streamlit_app.py
```

Open the app, upload PDFs/TXT, create a chat, pick a mode, attach files (for RAG), and ask away 🚀
