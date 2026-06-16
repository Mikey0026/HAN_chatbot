# Local AI Internship Support Agent
A fully local Retrieval-Augmented Generation (RAG) chatbot that answers
internship-related questions for HAN students, grounded in official HAN
documentation.

Built by Mike & Loukas as a school project, following the CRISP-DM methodology.

## Why local?

No student data leaves the machine. No external API keys. The full stack
LLM, embeddings, vector store and runs on your laptop.
This is done in order to keep private data private and most important: free to use.

## Stack

- **LLM**: Ollama (default `llama3.1:8b`, swappable via `config.yaml`)
- **Embeddings**: `nomic-embed-text` via Ollama
- **Vector store**: ChromaDB (persisted in `chroma_db/`)
- **Orchestration**: LangChain
- **Interface**: Streamlit

## Project layout

```
HAN_chatbot/
│
├── app.py
├── evaluate.py
├── config.yaml
├── requirements.txt
│
├── data/
│   ├── raw/
│   └── faq/
│
├── src/
│   ├── ingestion/
│   │   └── build_index.py
│   ├── rag/
│   │   └── chat.py
│   └── utils/
│       └── config.py
│
├── chroma_db/
├── evaluation/
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
streamlit run app.py
```

## Sources

Currently ingested:
- **GI_manual_2025-2026.pdf**
- **3rd year_internship_manual_Sem2_2025-2026**
- **Usage of AI**
- **Where to find the third year internship agreement & NDA & declaration letter**
- **Where to find the graduation internship agreement & NDA & declaration letter**
- **Where to find the graduation internship information on Brightspace**
- A possible FAQ

See `data/raw/SOURCES.md` for details.

## Grounding policy

The bot answers **ONLY** from the source documents. If retrieval doesn't
surface a relevant passage, the bot tells the student to contact the
internship advisor rather than guess. This is enforced in the prompt
template, not just by hope.
UPDATE: The bot refers more frequently to appendices instead of just referring the student to the advisor. This was requested by the advisor.
