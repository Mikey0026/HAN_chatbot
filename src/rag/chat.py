"""
RAG chat loop — Phase 4 of CRISP-DM (Modeling).

Retrieves the top-k most relevant chunks from ChromaDB for a student's
question, passes them to a local LLM via Ollama with a grounded prompt,
and prints the answer plus the sources used.

If the retriever returns nothing (or the LLM cannot find an answer in the
context), the bot tells the student to contact the internship coordinator
rather than guess. This is the single most important guardrail in the
system — enforced both by the prompt and by the empty-retrieval check.

Run with:
    python -m src.rag.chat

Type 'quit', 'exit', or hit Ctrl-D to leave.
"""
from __future__ import annotations

from typing import List, Tuple

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from src.utils.config import load_config, project_root


# The grounding prompt. Three rules: (1) answer only from context,
# (2) cite the source file and page, (3) refer to coordinator on miss.
PROMPT_TEMPLATE = """You are an assistant for HAN University students with questions about their graduation internship.

Answer the student's question using ONLY the information in the context below. The context is taken from official HAN documents.

Rules:
- If the context does not contain enough information to answer, reply exactly: "I don't have that information in the official documents. Please contact your internship coordinator."
- Do not invent details, dates, or rules that are not in the context.
- Keep the answer concise and practical.
- At the end of your answer, list the sources you used in the form: [source_file, page X].

Context:
{context}

PROMPT_TEMPLATE = """...
Conversation so far:
{history}

Question: {question}

Answer:"""


def format_context(docs: List[Document]) -> str:
    """Concatenate retrieved chunks into a single context block, prefixed
    with their source so the LLM can cite them."""
    parts = []
    for d in docs:
        src = d.metadata.get("source_file", "unknown")
        page = d.metadata.get("page", "?")
        # PyPDF pages are 0-indexed; show 1-indexed for human readability.
        page_human = page + 1 if isinstance(page, int) else page
        parts.append(f"[{src}, page {page_human}]\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


def format_sources(docs: List[Document]) -> str:
    """Build a deduplicated source list for display under the answer."""
    seen = set()
    lines = []
    for d in docs:
        src = d.metadata.get("source_file", "unknown")
        page = d.metadata.get("page", "?")
        page_human = page + 1 if isinstance(page, int) else page
        key = (src, page_human)
        if key not in seen:
            seen.add(key)
            lines.append(f"  - {src}, page {page_human}")
    return "\n".join(lines)

def format_history(history: List[Tuple[str, str]]) -> str:
    if not history:
        return "(no previous messages)"
    lines = []
    for user_msg, bot_msg in history:
        lines.append(f"Student: {user_msg}")
        lines.append(f"Bot: {bot_msg}")
    return "\n".join(lines)

def answer_question(
    question: str,
    vector_store: Chroma,
    llm: ChatOllama,
    k: int,
) -> Tuple[str, List[Document]]:
    """Retrieve, prompt, and return (answer_text, retrieved_docs)."""
    docs = vector_store.similarity_search(question, k=k)

    # Defensive empty-retrieval check — if the store is empty or nothing
    # is remotely similar, skip the LLM and respond directly.
    if not docs:
        return (
            "I don't have that information in the official documents. "
            "Please contact your internship coordinator.",
            [],
        )

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = prompt | llm
    response = chain.invoke({"context": format_context(docs), "question": question})
    return response.content, docs


def main() -> None:
    cfg = load_config()
    root = project_root()
    store_dir = root / cfg["vector_store_dir"]

    if not store_dir.exists() or not any(store_dir.iterdir()):
        print(
            "[chat] No vector store found. Run the ingestion first:\n"
            "       python -m src.ingestion.build_index"
        )
        return

    print(f"[chat] LLM: {cfg['llm_model']}  |  Embeddings: {cfg['embedding_model']}")
    print("[chat] Loading vector store ...")
    embeddings = OllamaEmbeddings(model=cfg["embedding_model"])
    vector_store = Chroma(
        collection_name=cfg["collection_name"],
        embedding_function=embeddings,
        persist_directory=str(store_dir),
    )

    llm = ChatOllama(model=cfg["llm_model"], temperature=cfg["temperature"])

    print("[chat] Ready. Ask a question, or type 'quit' to exit.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"quit", "exit"}:
            break

        answer, docs = answer_question(question, vector_store, llm, cfg["retrieval_k"], history)
        print(f"\nBot: {answer}\n")
        if docs:
            print("Sources used:")
            print(format_sources(docs))
        print()


if __name__ == "__main__":
    main()
