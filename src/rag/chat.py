"""
RAG chat loop:

Retrieves the top-k most relevant chunks from ChromaDB for a student's
question, passes them to a local LLM via Ollama with a grounded prompt,
and returns the answer plus the list of sources used.

Key behaviours (from the advisor meeting, May 2026):

  - Distinguishes between the third-year internship and the graduation
    internship. The student passes an `internship_type` value; retrieval is
    filtered by source file so the bot doesn't mix the two manuals.
    This is done by locking out the student's choice of internship type at the start of the conversation.

  - When the documents don't fully answer a question, the bot tries to
    point the student to the most relevant page or appendix in the manual
    before giving up. Only a complete blank pushes the "contact your
    coordinator" fallback. This was a key advisor request to avoid the bot giving up too early on.
    We did say that we need to keep the boundaries otherwise the bot gives hallucinations.

  - Appendix-aware: in the beginning the bot did not check appendices explicitly but now, the prompt explicitly tells the LLM that appendices
    contain assignment forms, appraisal forms, and the AI policy, and that
    students should be pointed to them by number when relevant.

  - Conversation history is maintained across turns so follow-ups like
    "what about for CS students?" can be resolved correctly.

  - Includes a small helper, `compute_portfolio_deadline()`, that returns
    the date "4 weeks before the internship end date" for the new portfolio
    deadline rule the advisor mentioned.
    This will be the new rule for both internships starting in the 2025-2026 academic year, so it's worth hardcoding this logic in a helper function rather than leaving it to the LLM.


Run with:
    python -m src.rag.chat

Type 'quit' or 'exit' to leave. Type 'clear' to wipe conversation history.
"""
from __future__ import annotations

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from src.utils.feedback import save_feedback

import re
from datetime import date, timedelta
from typing import List, Optional, Tuple

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from src.utils.config import load_config, project_root



# Internship-type routing
# Maps the value the UI passes in to (a) the human-readable label used in
# prompts and (b) a Chroma `where` filter that restricts retrieval to the
# right document(s). The filenames here must match what's in data/raw/
# exactly adjust if the actual filenames differ.

INTERNSHIP_GI = "graduation"
INTERNSHIP_Y3 = "third_year"

INTERNSHIP_LABELS = {
    INTERNSHIP_GI: "Graduation Internship",
    INTERNSHIP_Y3: "Third-Year Internship",
}

# Source-file filters. Filenames must match data/raw/ exactly,
# including spaces and capitalization.

# Files relevant to BOTH internship types, students need these regardless
# of which internship they picked.

SHARED_SOURCE_FILES_3RD = [
    "Usage of AI.pdf",
    "Where to find the third year internship agreement & NDA & declaration letter.pdf"
]
SHARED_SOURCE_FILES_GI = [
    "Usage of AI.pdf",
    "Where to find the graduation internship information on Brightspace.pdf",
    "Where to find the graduation internship agreement & NDA & declaration letter.pdf"
]

GI_SOURCE_FILES = ["GI manual 2025-2026.pdf"] + SHARED_SOURCE_FILES_GI
Y3_SOURCE_FILES = [
    "3rd year internship manual Sem2 2025-2026.pdf"
] + SHARED_SOURCE_FILES_3RD


def build_metadata_filter(internship_type: str) -> Optional[dict]:
    """Return a Chroma `where` dict that restricts retrieval to the
    documents relevant for the chosen internship type."""
    if internship_type == INTERNSHIP_GI:
        return {"source_file": {"$in": GI_SOURCE_FILES}}
    if internship_type == INTERNSHIP_Y3:
        return {"source_file": {"$in": Y3_SOURCE_FILES}}
    raise ValueError(
        f"Unknown internship_type: {internship_type!r}. "
        f"Expected one of: {list(INTERNSHIP_LABELS)}"
    )



# History length to keep for follow-up question rewriting. Longer history gives more.

HISTORY_TURNS = 5



# THE HOLY PROMPT.

PROMPT_TEMPLATE = """You are an assistant for HAN University students with questions about their {internship_label}.

Answer the student's question using ONLY the information in the context below. The context is taken from official HAN documents.

About the HAN internship documents:
- The manuals contain numbered appendices (e.g. "Appendix 1", "Appendix 2", "Appendix 6") with forms, assessment forms, 360-degree feedback forms, appraisal forms, and the AI usage policy. When relevant, point the student to the specific appendix by number.
- Brightspace is HAN's online learning environment where many forms and announcements are posted.

Rules:
- Use ONLY the facts in the context. Never invent details, dates, names, or rules.
- Use the conversation history ONLY to understand what the student is referring to (pronouns, follow-ups). Do NOT treat previous answers as a source of new facts, every claim must come from the context.
- When the context contains step-by-step instructions, navigation steps, or a procedure, give the full steps in your answer as a numbered list. Do NOT summarise them or just point to where they live, write them out.
- When the context contains a URL or link, include it in your answer exactly as written.
- If the context does not fully answer the question but DOES mention where the answer can be found (a page, section, appendix, or another document), point the student there. Example: "I don't have the exact details, but you can find this in Appendix 4 of the manual."
- If the context contains nothing relevant at all, reply exactly: "I don't have that information in the official documents. Please contact your internship coordinator."
- Keep answers focused. Use short bullet points or numbered steps when listing procedures.
- Do NOT include source citations, file names, or page numbers in your answer. Sources are shown to the student separately by the interface.
- If the retrieved context is not directly relevant to the question, do NOT answer using loosely related information.
- If no retrieved chunk directly answers the question, reply exactly:
  "I could not find a reliable answer in the retrieved documents."
- If the answer contains a list of requirements, steps, conditions, or criteria:
    - Put each bullet point on its own line
    - Do not place multiple bullet points on the same line
Conversation history:
{history}

Context:
{context}

Question: {question}

Answer:"""


# Matches inline citations like [GI_manual_2025-2026.pdf, page 9] in case
# the LLM still adds them despite the prompt rule.
_CITATION_PATTERN = re.compile(
    r"[\[\(][^\[\]\(\)]*?\.pdf[^\[\]\(\)]*?[\]\)]",
    re.IGNORECASE,
)


def strip_inline_citations(text: str) -> str:
    """Remove inline source citations the LLM may have added despite the
    prompt rule, and tidy the whitespace/punctuation left behind."""
    cleaned = _CITATION_PATTERN.sub("", text)
    cleaned = re.sub(r"\s+([.,;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


# Helpers for context / sources / history formatting

def format_context(docs: List[Document]) -> str:
    """Concatenate retrieved chunks into a single context block, prefixed
    with their source so the LLM knows which document each fact comes from
    (even though it won't cite them in its answer, it can be seen by the students in a dropdown)."""
    parts = []
    for d in docs:
        src = d.metadata.get("source_file", "unknown")
        page = d.metadata.get("page", "?")
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
        return "(no previous turns)"
    lines = []
    for q, a in history:
        lines.append(f"Student: {q}")
        lines.append(f"Assistant: {a}")
    return "\n".join(lines)


# Date calculation helper (advisor's "4 weeks before end date" rule)

def compute_portfolio_deadline(
    end_date: date, weeks_before: int = 4
) -> date:
    """Return the portfolio submission deadline under the new rule:
    `weeks_before` weeks before the internship end date."""
    return end_date - timedelta(weeks=weeks_before)


# Question rewriting (so retrieval works on follow-ups)

def rewrite_question_with_history(
    question: str,
    history: List[Tuple[str, str]],
    llm: ChatOllama,
) -> str:
    """Turn a follow-up like "what about CS students?" into a standalone
    question for retrieval. Vector search on bare follow-ups returns junk
    because there's no topical content to match."""
    if not history:
        return question

    rewrite_prompt = ChatPromptTemplate.from_template(
        """Given the conversation history and a follow-up question, rewrite the follow-up as a standalone question that can be understood without the history.

If the follow-up is already standalone, return it unchanged. Do not answer the question only rewrite it.

Conversation history:
{history}

Follow-up question: {question}

Standalone question:"""
    )
    chain = rewrite_prompt | llm
    response = chain.invoke({"history": format_history(history), "question": question})
    return response.content.strip()


# Main RAG entry point

def answer_question(
    question: str,
    history: List[Tuple[str, str]],
    vector_store: Chroma,
    llm: ChatOllama,
    k: int,
    internship_type: str,
) -> Tuple[str, List[Document]]:
    """Retrieve, prompt, and return (answer_text, retrieved_docs).

    Steps:
      1. Rewrite the follow-up into a standalone question (if there's history).
      2. Retrieve top-k chunks, filtered by internship_type.
      3. Generate an answer grounded in those chunks.
      4. Strip any stray inline citations.
    """
    search_query = rewrite_question_with_history(question, history, llm)

    metadata_filter = build_metadata_filter(internship_type)
    docs_and_scores = vector_store.similarity_search_with_score(
    search_query,
        k=k,
        filter=metadata_filter,
    )
    docs = [doc for doc, score in docs_and_scores]

    if not docs:
        return (
            "I don't have that information in the official documents. "
            "Please contact your internship coordinator.",
            [],
        )

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = prompt | llm
    response = chain.invoke(
        {
            "internship_label": INTERNSHIP_LABELS[internship_type],
            "history": format_history(history),
            "context": format_context(docs),
            "question": question,
        }
    )
    clean_answer = strip_inline_citations(response.content)
    return clean_answer, docs


# CLI entry point (the Streamlit app uses answer_question directly)

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

    # Ask the student which internship up front. No default, they must pick.
    print(
        "\nWhich internship are you asking about?\n"
        "  1) Graduation Internship\n"
        "  2) Third-Year Internship"
    )
    while True:
        choice = input("Choose 1 or 2: ").strip()
        if choice in {"1", "2"}:
            break
        print("Please enter 1 or 2.")
    internship_type = INTERNSHIP_GI if choice == "1" else INTERNSHIP_Y3
    print(f"[chat] Scope: {INTERNSHIP_LABELS[internship_type]}\n")

    history: List[Tuple[str, str]] = []
    print("[chat] Ready. Ask a question, 'clear' to reset history, 'quit' to exit.\n")
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
        if question.lower() == "clear":
            history.clear()
            print("[chat] History cleared.\n")
            continue

        answer, docs = answer_question(
            question, history, vector_store, llm, cfg["retrieval_k"],
            internship_type=internship_type,
        )
        print(f"\nBot: {answer}\n")
        if docs:
            print("Sources used:")
            print(format_sources(docs))
        print()

        history.append((question, answer))
        if len(history) > HISTORY_TURNS:
            history = history[-HISTORY_TURNS:]


if __name__ == "__main__":
    main()
