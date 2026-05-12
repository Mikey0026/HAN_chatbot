# Local AI Internship Support Agent — HAN University

A fully local Retrieval-Augmented Generation (RAG) chatbot that answers
internship-related questions for HAN students, grounded in official HAN
documentation.

Built by Mike & Loukas as a school project, following the CRISP-DM methodology.

## Why local?

No student data leaves the machine. No external API keys. The full stack
LLM, embeddings, vector store and runs on your laptop.

## Stack

- **LLM**: Ollama (default `llama3.1:8b`, swappable via `config.yaml`)
- **Embeddings**: `nomic-embed-text` via Ollama
- **Vector store**: ChromaDB (persisted in `chroma_db/`)
- **Orchestration**: LangChain
- **Interface**: CLI (for now)

## Project layout

```
han_internship_agent/
├── config.yaml                 # Single source of truth for tunables
├── requirements.txt
├── data/
│   ├── raw/                    # Official HAN PDFs (source of truth)
│   │   ├── GI_manual_2025-2026.pdf
│   │   └── SOURCES.md
│   └── faq/                    # FAQ document(s)
├── src/
│   ├── ingestion/build_index.py   # Phase 3: chunk + embed + persist
│   ├── rag/chat.py                # Phase 4: retrieve + generate
│   └── utils/config.py
├── chroma_db/                  # Vector store (gitignored)
├── evaluation/                 # Phase 5: test questions + results
├── tests/
└── docs/
```

The folder layout mirrors CRISP-DM phases: `data/` for raw inputs,
`src/ingestion/` for Data Preparation, `src/rag/` for Modeling, `evaluation/`
for the Evaluation phase. Easy to explain in the report.

## Quick start

```bash
# 1. Install Ollama and pull the models (one-time)
ollama pull llama3.1:8b
ollama pull nomic-embed-text

# 2. Install Python deps
pip install -r requirements.txt

# 3. Build the vector index from the documents in data/
python -m src.ingestion.build_index

# 4. Chat
python -m src.rag.chat
```

## Sources

Currently ingested:
- **GI_manual_2025-2026.pdf** — Graduation Internship Manual for IB and CS
  students. 57 pages covering admission, acquisition, approval, deliverables,
  timeline, the 5 Performance Areas, CBI procedure, resit/retake rules, and
  per-specialisation appendices (O&C, M&S, Finance, SCM, CS).

Pending: a second HAN PDF and the FAQ.

See `data/raw/SOURCES.md` for details.

## Grounding policy

The bot answers **only** from the source documents. If retrieval doesn't
surface a relevant passage, the bot tells the student to contact the
internship coordinator rather than guess. This is enforced in the prompt
template, not just by hope.
