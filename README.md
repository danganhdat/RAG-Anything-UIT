# RAG-Anything UIT

Multimodal RAG system built with [RAG-Anything](https://github.com/HKUDS/RAG-Anything), Milvus Lite, and Qwen models via OpenRouter.

Ingests PDFs (parsed by MinerU), builds a knowledge graph + vector index, and serves answers through a FastAPI backend + Streamlit chat UI.

## Project Structure

```
rag_app/                   # Main package
├── core/                  #   Config (Pydantic), exceptions, logging
├── adapters/              #   OpenRouter LLM + embedding clients
├── store/                 #   Milvus vector store
├── services/              #   RAG service, ingestion, DI container
└── api/                   #   FastAPI app factory, routes, deps

run_api.py                 # Start the API server
run_ui.py                  # Start the Streamlit chat UI
ingest_pdf.py              # CLI — ingest a PDF
query_system.py            # CLI — query the knowledge base
```

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -e .                    # or: pip install -r requirements.txt
cp .env.example .env                # fill in OPENROUTER_API_KEY
```

Get an API key at [openrouter.ai](https://openrouter.ai/).

## Usage

### 1. Ingest a PDF

```bash
python ingest_pdf.py                                          # default sample PDF
python ingest_pdf.py path/to/doc.pdf                          # custom PDF
python ingest_pdf.py path/to/doc.pdf --start-page 0 --end-page 5   # page range
```

### 2. Query (CLI)

```bash
python query_system.py "What are the admission requirements?"
```

### 3. API + Chat UI

```bash
# Terminal 1
python run_api.py

# Terminal 2
streamlit run run_ui.py
```

Open http://localhost:8501.

## Troubleshooting

**MinerU fails** — Ensure `raganything[all]` is installed. On Linux/WSL: `sudo apt install libmagic1`. Try `--end-page 2` first to test with fewer pages.

**Missing API key** — The app will fail immediately with a `ValidationError` if `OPENROUTER_API_KEY` is not set in `.env`.

**Embedding dimension mismatch** — If you change `EMBED_MODEL`, update `EMBED_DIM` to match and delete `milvus_lite.db/` to rebuild the index.
