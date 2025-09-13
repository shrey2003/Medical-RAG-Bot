# RAG ChatBot — LangChain + ChatGPT

A simple Retrieval-Augmented Generation (RAG) chatbot built with Streamlit, LangChain and OpenAI. Upload documents (PDF, DOCX, TXT, CSV, JSON), the app builds embeddings with a small HuggingFace model, stores vectors in a local Chroma vector store and lets you chat with your documents via a conversational retrieval chain.


<img width="1792" height="832" alt="image" src="https://github.com/user-attachments/assets/81e4d6de-84cb-4f5c-bd61-7a45824c46f9" />


---

## Demo video

https://drive.google.com/file/d/1FKtvLheESMe9KwJMrzPEltk5tPhWEF23/view?usp=sharing

## Key features

- Upload multiple documents (PDF, DOCX, TXT, CSV, JSON)
- Chunking and embedding using `sentence-transformers/all-MiniLM-L6-v2`
- Local Chroma vector store persisted under `chroma_store/`
- ConversationalRetrievalChain with a ChatOpenAI LLM (configurable model)
- Simple Streamlit UI for uploading, processing and chatting

Additional supported formats

- CSV: parsed row-by-row into plain text (comma-separated) and indexed as documents. If parsing fails (malformed CSV or encoding issues), the raw file text is used as a fallback.
- JSON: pretty-printed and indexed as text. If JSON parsing fails, the raw file text is used as a fallback.

---

## Files of interest

- `app.py` — main Streamlit app
- `requirements.txt` — Python dependencies
- `chroma_store/` — persisted vector store and related files (generated at runtime)
- `.env` — environment file (contains `OPENAI_API_KEY`) — **do not commit**

---

## Setup (Windows PowerShell)

1. Clone the repo and open in VS Code (or your editor)

2. Create and activate a virtual environment (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks scripts, run as admin and do:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

3. Install dependencies:

```powershell
pip install -r requirements.txt
```

4. Create a `.env` file at the project root containing your OpenAI key:

```text
OPENAI_API_KEY="sk-..."
```

> Important: rotate keys if you accidentally commit them.

---

## Run

From the project root (with venv active):

```powershell
streamlit run app.py
```

Open the URL printed by Streamlit (usually `http://localhost:8501`).

---




## How it works (high level)

1. Upload documents in the left column.
2. Files are temporarily written to disk and read via appropriate loaders or parsers:
   - PDFs: `PyPDFLoader`
   - DOC/DOCX: `Docx2txtLoader`
   - TXT: `TextLoader`
   - CSV: parsed into text rows and wrapped into LangChain `Document` objects (falls back to raw text on error)
   - JSON: pretty-printed JSON is used as document text (falls back to raw text on error)
3. Extracted text is split into chunks (768 chars, overlap 128) via `CharacterTextSplitter`.
4. Embeddings generated using `HuggingFaceEmbeddings` (MiniLM) and stored in Chroma.
5. A `ConversationalRetrievalChain` is created with a `ChatOpenAI` LLM and a conversation memory.
6. The chat UI sends queries to the chain; results are returned and stored in Streamlit session state.

---

## Design choices (short explanation)

- Local, small embedding model (`all-MiniLM-L6-v2`) — good accuracy vs cost tradeoff for local CPU inference.
- Chroma vector store — easy local persistence and fast retrieval for small/medium datasets.
- `ConversationalRetrievalChain` with memory — maintains chat context for follow-up questions.
- Streamlit UI — fast to iterate and share; minimal front-end code required.
- Kept splitting parameters conservative (768 chunk size) to balance context and retrieval quality.

---

## Security & housekeeping

- Remove `.env` from the repository and rotate the OpenAI key if it was committed.
- If you want a fresh vector store run, remove `chroma_store/` to force re-generation.
- `.gitignore` includes `chroma_store/` and `.env` to avoid committing them.

---

## Troubleshooting

- Missing dependency errors: ensure the venv is active and `pip install -r requirements.txt` completed.
- `PermissionError` when creating temp files: run VS Code/PowerShell with appropriate permissions or choose a different temp dir.
- Long processing times: embeddings are computed on CPU by default; switch to a GPU device in `HuggingFaceEmbeddings(model_kwargs={"device": "cuda"})` if available.
- Chat output empty or errors from OpenAI: verify `OPENAI_API_KEY` and network access.
- Chat output empty or errors from OpenAI: verify `OPENAI_API_KEY` and network access.

- CSV/JSON parsing issues: the app attempts to parse CSV rows and pretty-print JSON before indexing; if parsing fails due to malformed content or encoding problems, the raw file contents are indexed instead. For best results ensure CSV files are well-formed and JSON files are valid UTF-8.

---

## Next improvements (suggestions)

- Add ability to select LLM model and configure settings via UI.
- Add caching of embeddings per file hash to avoid recomputing embeddings for unchanged files.
- Add unit tests for document loaders and vector store creation.
- Add an admin panel to inspect the vector store (embeddings count, documents indexed).

---

## License

MIT — adapt as needed.

