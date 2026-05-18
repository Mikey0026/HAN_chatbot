"""
Streamlit web UI for the HAN Internship Support Agent.

Wraps the existing RAG pipeline (src/rag/chat.py) in a browser-based chat
interface, styled to match HAN University's brand identity:
  - HAN Pink (#E6007E) as primary accent
  - Black (#000000) background with white text
  - HAN logo loaded locally from assets/han_logo.png
  - ChatGPT-style pill-shaped chat input with inline send button

Run with (from the project root):
    streamlit run app.py

The app opens automatically in your default browser at http://localhost:8501
"""
from __future__ import annotations

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from pathlib import Path

import streamlit as st

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama

from src.utils.config import load_config, project_root
from src.rag.chat import (
    answer_question,
    format_sources,
    HISTORY_TURNS,
)


# -----------------------------------------------------------------------------
# HAN brand palette (dark mode)
# -----------------------------------------------------------------------------
HAN_PINK = "#E6007E"        # primary accent
HAN_PINK_DARK = "#B8005F"   # hover / active states
HAN_BG = "#0E1117"          # main background
HAN_SURFACE = "#1A1A1A"     # cards, bubbles, sidebar
HAN_SURFACE_2 = "#2A2A2A"   # slightly lighter for contrast
HAN_TEXT = "#FFFFFF"
HAN_TEXT_DIM = "#B0B0B0"

# Logo lives in the project's assets/ folder, alongside app.py
HAN_LOGO_PATH = str(Path(__file__).resolve().parent / "assets" / "han_logo.png")


# -----------------------------------------------------------------------------
# Page setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="HAN Internship Assistant",
    page_icon="🎓",
    layout="centered",
)


# -----------------------------------------------------------------------------
# Custom CSS — dark theme with HAN accents
# -----------------------------------------------------------------------------
st.markdown(
    f"""
    <style>
    /* App background */
    .stApp {{
        background-color: {HAN_BG};
        color: {HAN_TEXT};
    }}

    /* All body text */
    .stApp, .stApp p, .stApp li, .stApp span, .stApp div {{
        color: {HAN_TEXT};
    }}

    /* Headings */
    h1, h2, h3, h4, h5, h6 {{
        color: {HAN_TEXT} !important;
        font-weight: 700;
    }}

    /* Title block, mirroring the HAN manual cover */
    .han-title {{
        color: {HAN_TEXT};
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }}
    .han-title-accent {{
        color: {HAN_PINK};
    }}
    .han-subtitle {{
        color: {HAN_TEXT_DIM};
        font-size: 1rem;
        border-bottom: 3px solid {HAN_PINK};
        padding-bottom: 0.75rem;
        margin-bottom: 1.5rem;
        display: inline-block;
    }}

    /* User chat bubble */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {{
        background-color: {HAN_SURFACE_2};
        border-left: 4px solid {HAN_PINK};
        border-radius: 6px;
        padding: 0.5rem 1rem;
    }}

    /* Assistant chat bubble */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {{
        background-color: {HAN_SURFACE_2};
        border-left: 4px solid {HAN_PINK};
        border-radius: 6px;
        padding: 0.5rem 1rem;
    }}

    /* ===== Chat input — ChatGPT-style ===== */

    /* Wrapper area at the bottom of the page */
    [data-testid="stChatInput"] {{
        background-color: {HAN_BG};
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
        padding: 1rem 0 1.5rem 0;
    }}

    /* The inner card that holds the textarea + send button.
       This is the part that becomes the pill shape. */
    [data-testid="stChatInput"] > div {{
        background-color: {HAN_SURFACE} !important;
        border: 1px solid {HAN_SURFACE_2} !important;
        border-radius: 28px !important;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.4) !important;
        outline: none !important;
        padding: 4px 8px !important;
        transition: border-color 0.15s, box-shadow 0.15s;
    }}
    [data-testid="stChatInput"] > div:focus-within {{
        border-color: {HAN_PINK} !important;
        box-shadow: 0 2px 16px rgba(230, 0, 126, 0.25) !important;
    }}

    /* The textarea itself — flush with the card, no internal border */
    [data-testid="stChatInput"] textarea {{
        background-color: transparent !important;
        color: {HAN_TEXT} !important;
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
        font-size: 1rem !important;
        padding: 12px 16px !important;
        min-height: 52px !important;
        resize: none !important;
    }}
    [data-testid="stChatInput"] textarea:focus {{
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
    }}
    [data-testid="stChatInput"] textarea::placeholder {{
        color: {HAN_TEXT_DIM} !important;
    }}

    /* Send button — circular, pink, sits inside the pill */
    [data-testid="stChatInput"] button {{
        background-color: {HAN_PINK} !important;
        border: none !important;
        border-radius: 50% !important;
        width: 36px !important;
        height: 36px !important;
        min-width: 36px !important;
        min-height: 36px !important;
        padding: 0 !important;
        margin: 6px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: background-color 0.15s;
    }}
    [data-testid="stChatInput"] button:hover {{
        background-color: {HAN_PINK_DARK} !important;
    }}
    [data-testid="stChatInput"] button:disabled {{
        background-color: {HAN_SURFACE_2} !important;
        opacity: 0.6;
    }}
    [data-testid="stChatInput"] button svg {{
        fill: white !important;
        color: white !important;
    }}

    /* ===== Sidebar buttons (Clear conversation) ===== */
    .stButton > button {{
        background-color: {HAN_PINK};
        color: white;
        border: none;
        border-radius: 4px;
        font-weight: 600;
        transition: background-color 0.15s;
    }}
    .stButton > button:hover {{
        background-color: {HAN_PINK_DARK};
        color: white;
    }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {HAN_SURFACE};
        border-right: 1px solid {HAN_SURFACE_2};
    }}
    [data-testid="stSidebar"] * {{
        color: {HAN_TEXT};
    }}
    [data-testid="stSidebar"] h2 {{
        color: {HAN_TEXT} !important;
        border-bottom: 2px solid {HAN_PINK};
        padding-bottom: 0.25rem;
    }}
    [data-testid="stSidebar"] code {{
        background-color: {HAN_SURFACE_2};
        color: {HAN_PINK};
        padding: 2px 6px;
        border-radius: 3px;
    }}

    /* Expander (the "Sources" dropdown) */
    [data-testid="stExpander"] {{
        background-color: {HAN_SURFACE};
        border: 1px solid {HAN_SURFACE_2};
        border-radius: 4px;
    }}
    [data-testid="stExpander"] summary {{
        color: {HAN_TEXT} !important;
    }}
    [data-testid="stExpander"] summary:hover {{
        color: {HAN_PINK} !important;
    }}

    /* Inline code in chat */
    .stApp code {{
        background-color: {HAN_SURFACE_2};
        color: {HAN_PINK};
        padding: 2px 6px;
        border-radius: 3px;
    }}

    /* Spinner colour */
    [data-testid="stSpinner"] > div > div {{
        border-top-color: {HAN_PINK} !important;
    }}

    /* Error / warning boxes should still be readable on black */
    [data-testid="stAlert"] {{
        background-color: {HAN_SURFACE};
        color: {HAN_TEXT};
    }}

    /* Hide Streamlit's default top bar branding */
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
    header[data-testid="stHeader"] {{
        background-color: {HAN_BG};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Header — HAN logo + title
# -----------------------------------------------------------------------------
header_left, header_right = st.columns([1, 4])
with header_left:
    if Path(HAN_LOGO_PATH).exists():
        st.image(HAN_LOGO_PATH, width=110)
    else:
        st.warning("Logo not found at assets/han_logo.png")
with header_right:
    st.markdown(
        '<div class="han-title">Internship <span class="han-title-accent">Assistant</span></div>'
        '<div class="han-subtitle">Ask anything about your graduation internship. The answers are from official HAN documents only.</div>',
        unsafe_allow_html=True,
    )

st.write("")  # vertical breathing room


# -----------------------------------------------------------------------------
# One-time resource loading
# Streamlit reruns the whole script on every interaction, so we cache the
# heavy stuff (vector store, LLM) to avoid reloading the model every message.
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading models and vector store...")
def load_resources():
    cfg = load_config()
    root = project_root()
    store_dir = root / cfg["vector_store_dir"]

    if not store_dir.exists() or not any(store_dir.iterdir()):
        st.error(
            "No vector store found. Run the ingestion first from a terminal:\n\n"
            "    python -m src.ingestion.build_index"
        )
        st.stop()

    embeddings = OllamaEmbeddings(model=cfg["embedding_model"])
    vector_store = Chroma(
        collection_name=cfg["collection_name"],
        embedding_function=embeddings,
        persist_directory=str(store_dir),
    )
    llm = ChatOllama(model=cfg["llm_model"], temperature=cfg["temperature"])
    return cfg, vector_store, llm


cfg, vector_store, llm = load_resources()


# -----------------------------------------------------------------------------
# Sidebar — info and controls
# -----------------------------------------------------------------------------
with st.sidebar:
    if Path(HAN_LOGO_PATH).exists():
        st.image(HAN_LOGO_PATH, width=140)
    st.markdown("## About")
    st.markdown(
        "This assistant answers questions about the **HAN 3rd year andGraduation "
        "Internships** based on official documents.\n\n"
        "All processing happens locally, no data leaves the machine." \
        " The data is stored for just this chat session and is cleared when you close the browser tab."
    )

    st.markdown("## Configuration")
    st.markdown(
        f"- **LLM**: `{cfg['llm_model']}`\n"
        f"- **Embeddings**: `{cfg['embedding_model']}`\n"
        f"- **Retrieval k**: {cfg['retrieval_k']}\n"
        f"- **History turns**: {HISTORY_TURNS}"
    )

    st.markdown("")
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.history = []
        st.rerun()


# -----------------------------------------------------------------------------
# Conversation state
# st.session_state persists across reruns of the script (i.e. between user
# messages) but resets when the browser tab is closed.
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    # Display-only list: each item is {"role": "user"|"assistant", "content": str, "sources": str?}
    st.session_state.messages = []

if "history" not in st.session_state:
    # RAG history: list of (question, answer) tuples used by the prompt.
    st.session_state.history = []


# -----------------------------------------------------------------------------
# Render the chat so far
# -----------------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 Sources"):
                st.markdown(msg["sources"])


# -----------------------------------------------------------------------------
# Handle a new question
# -----------------------------------------------------------------------------
if question := st.chat_input("Ask about your internship..."):
    # 1. Show the user's message immediately.
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # 2. Run the RAG pipeline and show the answer.
    with st.chat_message("assistant"):
        with st.spinner("Searching the documents..."):
            answer, docs = answer_question(
                question=question,
                history=st.session_state.history,
                vector_store=vector_store,
                llm=llm,
                k=cfg["retrieval_k"],
            )
        st.markdown(answer)

        sources_md = ""
        if docs:
            sources_md = format_sources(docs)
            with st.expander("📄 Sources"):
                st.markdown(sources_md)

    # 3. Persist to display + RAG history.
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources_md}
    )
    st.session_state.history.append((question, answer))
    if len(st.session_state.history) > HISTORY_TURNS:
        st.session_state.history = st.session_state.history[-HISTORY_TURNS:]
