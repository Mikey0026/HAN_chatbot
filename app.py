"""
UX additions from advisor feedback:
  - Students pick which internship they're asking about (Third-Year or
    Graduation), so retrieval doesn't mix the two manuals.
  - Sidebar widget that computes the portfolio submission deadline as
    4 weeks before the internship end date" (advisor's new rule).

Run(from the project root):
    streamlit run app.py
"""
from __future__ import annotations

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from datetime import date
from pathlib import Path
from src.utils.feedback import save_feedback

import streamlit as st

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama

from src.utils.config import load_config, project_root
from src.rag.chat import (
    answer_question,
    compute_portfolio_deadline,
    format_sources,
    HISTORY_TURNS,
    INTERNSHIP_GI,
    INTERNSHIP_LABELS,
    INTERNSHIP_Y3,
)


# HAN brand palette (dark mode)

HAN_PINK = "#E6007E"
HAN_PINK_DARK = "#B8005F"
HAN_BG = "#000000"
HAN_SURFACE = "#1A1A1A"
HAN_SURFACE_2 = "#2A2A2A"
HAN_TEXT = "#FFFFFF"
HAN_TEXT_DIM = "#B0B0B0"

HAN_LOGO_PATH = str(Path(__file__).resolve().parent / "assets" / "han_logo.png")


# Page setup

st.set_page_config(
    page_title="HAN Internship Assistant",
    page_icon=HAN_LOGO_PATH,
    layout="centered",
)


# Custom CSS dark theme with HAN accents, ChatGPT-style chat input

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: {HAN_BG};
        color: {HAN_TEXT};
    }}
    .stApp, .stApp p, .stApp li, .stApp span, .stApp div {{
        color: {HAN_TEXT};
    }}
    h1, h2, h3, h4, h5, h6 {{
        color: {HAN_TEXT} !important;
        font-weight: 700;
    }}

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

    /* Chat bubbles */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {{
        background-color: {HAN_SURFACE_2};
        border-left: 4px solid {HAN_PINK};
        border-radius: 6px;
        padding: 0.5rem 1rem;
    }}
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {{
        background-color: {HAN_SURFACE};
        border-left: 4px solid {HAN_PINK};
        border-radius: 6px;
        padding: 0.5rem 1rem;
    }}

    /* ===== Chat input ChatGPT-style pill ===== */
    [data-testid="stChatInput"] {{
        background-color: {HAN_BG};
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
        padding: 1rem 0 1.5rem 0;
    }}
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

    /* Sidebar */
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

    /* Radio buttons (internship-type selector) pink selected dot */
    [data-testid="stRadio"] label {{
        color: {HAN_TEXT} !important;
    }}
    [data-testid="stRadio"] [role="radio"][aria-checked="true"] {{
        background-color: {HAN_PINK} !important;
        border-color: {HAN_PINK} !important;
    }}

    /* Date input */
    [data-testid="stDateInput"] input {{
        background-color: {HAN_SURFACE_2} !important;
        color: {HAN_TEXT} !important;
        border: 1px solid {HAN_SURFACE_2} !important;
    }}

    /* Expander */
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

    .stApp code {{
        background-color: {HAN_SURFACE_2};
        color: {HAN_PINK};
        padding: 2px 6px;
        border-radius: 3px;
    }}
    [data-testid="stSpinner"] > div > div {{
        border-top-color: {HAN_PINK} !important;
    }}
    [data-testid="stAlert"] {{
        background-color: {HAN_SURFACE};
        color: {HAN_TEXT};
    }}

    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
    header[data-testid="stHeader"] {{
        background-color: {HAN_BG};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# Header

header_left, header_right = st.columns([1, 4])
with header_left:
    if Path(HAN_LOGO_PATH).exists():
        st.image(HAN_LOGO_PATH, width=110)
with header_right:
    st.markdown(
        '<div class="han-title">Internship <span class="han-title-accent">Assistant</span></div>'
        '<div class="han-subtitle">Ask anything about your internship, the answers are from official HAN documents ONLY.</div>',
        unsafe_allow_html=True,
    )

st.write("")


# Cached resource loading

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


# Internship-type selector
# Showed first to before students can ask questions

if "internship_type" not in st.session_state:
    st.session_state.internship_type = None

if st.session_state.internship_type is None:
    st.markdown("### Which internship are you asking about?")
    st.caption(
        "This helps me give you the right answer, the rules are different "
        "for the third-year internship and the graduation internship."
    )
    choice = st.radio(
        "Select one:",
        options=[INTERNSHIP_GI, INTERNSHIP_Y3],
        format_func=lambda v: INTERNSHIP_LABELS[v],
        index=None,
        label_visibility="collapsed",
    )
    if choice is not None:
        st.session_state.internship_type = choice
        st.rerun()
    st.stop()


# Sidebar
# With logo, internship type display + change button, about text, portfolio deadline calculator, config display, clear conversation button

with st.sidebar:
    if Path(HAN_LOGO_PATH).exists():
        st.image(HAN_LOGO_PATH, width=140)

    st.markdown("## Selected internship")
    st.markdown(
        f"**{INTERNSHIP_LABELS[st.session_state.internship_type]}**"
    )
    if st.button("Change internship", use_container_width=True):
        st.session_state.internship_type = None
        st.session_state.messages = []
        st.session_state.history = []
        st.rerun()

    st.markdown("## About")
    st.markdown(
        "This assistant answers questions about HAN internships based on "
        "official documents.\n\n"
        "All processing happens locally, no data leaves the machine."
    )

    # Portfolio deadline calculator (advisor's "4 weeks before end" rule)
    st.markdown("## Portfolio deadline")
    st.caption(
        "The portfolio is due 4 weeks before your internship end date. "
        "Pick your end date of your internship to see the deadline."
    )
    end_date_input = st.date_input(
        "Internship end date",
        value=None,
        min_value=date(2025, 1, 1),
        max_value=date(2030, 12, 31),
        label_visibility="collapsed",
    )
    if end_date_input is not None:
        deadline = compute_portfolio_deadline(end_date_input)
        st.markdown(
            f"<div style='background-color:{HAN_SURFACE_2}; padding:10px 12px; "
            f"border-left:3px solid {HAN_PINK}; border-radius:4px; margin-top:8px;'>"
            f"📅 <b>Portfolio deadline:</b><br>{deadline.strftime('%A, %d %B %Y')}"
            f"</div>",
            unsafe_allow_html=True,
        )
    debug_mode = st.sidebar.checkbox(
    "Debug retrieval",
    value=False
)
    
# This can be in there but I feel like it's not super important for the students and just adds to the "wall of text" in the sidebar.
    #st.markdown("## Configuration")
    #st.markdown(
        #f"- **LLM**: `{cfg['llm_model']}`\n"
        #f"- **Embeddings**: `{cfg['embedding_model']}`\n"
        #f"- **Retrieval k**: {cfg['retrieval_k']}\n"
        #f"- **History turns**: {HISTORY_TURNS}"
    #)

    st.markdown("")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.history = []
        st.rerun()


# Conversation state

if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []


# Render history

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 Sources"):
                st.markdown(msg["sources"])


# Handle a new question

placeholder_text = (
    f"Ask about your {INTERNSHIP_LABELS[st.session_state.internship_type].lower()}..."
)
if question := st.chat_input(placeholder_text):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Finding an answer..."):
            answer, docs = answer_question(
                question=question,
                history=st.session_state.history,
                vector_store=vector_store,
                llm=llm,
                k=cfg["retrieval_k"],
                internship_type=st.session_state.internship_type,
            )
        st.markdown(answer)

        sources_md = ""
        if docs:
            sources_md = format_sources(docs)
            with st.expander("📄 Sources"):
                st.markdown(sources_md)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources_md}
    )
    st.session_state.history.append((question, answer))
    st.session_state.last_question = question
    st.session_state.last_answer = answer
    if len(st.session_state.history) > HISTORY_TURNS:
        st.session_state.history = st.session_state.history[-HISTORY_TURNS:]
# Feedback buttons

if (
    "last_question" in st.session_state
    and "last_answer" in st.session_state
):

    st.divider()

    st.markdown("### Was this answer helpful?")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("👍 Helpful", key="feedback_up"):
            save_feedback(
                question=st.session_state.last_question,
                answer=st.session_state.last_answer,
                rating="up",
            )
            st.success("Thank you for your feedback!")

    with col2:
        if st.button("👎 Not Helpful", key="feedback_down"):
            save_feedback(
                question=st.session_state.last_question,
                answer=st.session_state.last_answer,
                rating="down",
            )
            st.success("Thank you for your feedback!")
