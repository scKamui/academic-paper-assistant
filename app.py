import hashlib
import tempfile
from pathlib import Path

import streamlit as st
from groq import APIError
from pypdf.errors import PdfReadError

from src.embeddings import load_embedding_model
from src.generate_answer import generate_answer
from src.process_document import process_document
from src.prompt import build_rag_prompt
from src.search import search_chunks
from src.structured_analysis import analyze_document


MAX_FILE_SIZE_MB = 20


def add_custom_styles():
    # make the default Streamlit layout feel more like a finished product
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1120px;
            padding-top: 2.5rem;
            padding-bottom: 4rem;
        }

        h1 {
            letter-spacing: -0.04em;
            font-size: clamp(2.8rem, 7vw, 4.8rem) !important;
            line-height: 1;
            margin-bottom: 0.4rem !important;
        }

        h2, h3 {
            letter-spacing: -0.02em;
        }

        .product-label {
            color: #5eead4;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.14em;
            margin-bottom: 0.8rem;
            text-transform: uppercase;
        }

        .hero-copy {
            color: #cbd5e1;
            font-size: 1.15rem;
            line-height: 1.7;
            max-width: 760px;
            margin-bottom: 1.4rem;
        }

        .trust-strip {
            background: rgba(20, 184, 166, 0.08);
            border: 1px solid rgba(45, 212, 191, 0.25);
            border-radius: 14px;
            color: #ccfbf1;
            margin: 1rem 0 2rem;
            padding: 0.9rem 1rem;
        }

        [data-testid="stFileUploader"] {
            background: rgba(15, 23, 42, 0.55);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 16px;
            padding: 0.5rem 1rem 1rem;
        }

        [data-testid="stMetric"] {
            background: rgba(15, 23, 42, 0.65);
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 14px;
            padding: 1rem;
        }

        .stButton > button, .stFormSubmitButton > button {
            border-radius: 10px;
            font-weight: 650;
            min-height: 2.7rem;
        }

        [data-testid="stExpander"] {
            border-color: rgba(148, 163, 184, 0.18);
            border-radius: 10px;
        }

        [data-testid="stSidebar"] {
            border-right: 1px solid rgba(148, 163, 184, 0.12);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def get_embedding_model():
    # load the model once and reuse it for every question in this session
    return load_embedding_model()


def process_uploaded_pdf(pdf_bytes, model):
    # pypdf needs a file path, so temporarily save the uploaded bytes
    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_file.write(pdf_bytes)
            temporary_path = Path(temp_file.name)

        return process_document(temporary_path, model=model)
    finally:
        # remove the uploaded paper as soon as its text has been processed
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def show_evidence(evidence):
    # place long supporting passages inside expandable sections
    for evidence_number, evidence_item in enumerate(evidence, start=1):
        page_number = evidence_item["page_number"]

        with st.expander(
            f"Evidence {evidence_number} — PDF page {page_number}"
        ):
            st.write(evidence_item["passage"])


def show_structured_analysis(analysis):
    # show the paper type first so students know what kind of document they uploaded
    document_type = analysis["document_type"]
    readable_type = document_type["type"].replace("_", " ").title()
    st.subheader(readable_type)
    st.write(document_type["explanation"])
    show_evidence(document_type["evidence"])

    st.divider()
    st.subheader("Hypothesis or objective")
    hypothesis = analysis["hypothesis"]

    if hypothesis["found"]:
        st.write(hypothesis["summary"])
        show_evidence(hypothesis["evidence"])
    else:
        st.info("No explicit hypothesis or objective was found in the evidence.")

    st.subheader("Methodology")
    methodology = analysis["methodology"]

    if methodology["found"]:
        st.write(methodology["summary"])
        show_evidence(methodology["evidence"])
    else:
        st.info("No methodology was found in the evidence.")

    st.subheader("Major findings")
    findings = analysis["findings"]

    if findings["found"]:
        for item_number, item in enumerate(findings["items"], start=1):
            st.markdown(f"**{item_number}. {item['claim']}**")
            show_evidence(item["evidence"])
    else:
        st.info("No major findings were found in the evidence.")

    st.subheader("Limitations stated by the authors")
    limitations = analysis["author_stated_limitations"]

    if limitations["found"]:
        for item_number, item in enumerate(limitations["items"], start=1):
            st.markdown(f"**{item_number}. {item['claim']}**")
            show_evidence(item["evidence"])
    else:
        st.info("No author-stated limitations were found in the evidence.")

    st.subheader("Possible limitations suggested by AI")
    ai_limitations = analysis["ai_suggested_limitations"]

    if ai_limitations:
        for item in ai_limitations:
            pages = ", ".join(str(page) for page in item["based_on_pages"])
            st.warning(
                f"{item['suggestion']}\n\n"
                f"Reason: {item['reason']}\n\n"
                f"Based on PDF page(s): {pages}"
            )
    else:
        st.info("No additional AI-suggested limitations were created.")


def reset_document_state(file_id, file_name):
    # clear results when the student uploads a different paper
    st.session_state["file_id"] = file_id
    st.session_state["file_name"] = file_name
    st.session_state.pop("document", None)
    st.session_state.pop("analysis", None)
    st.session_state["question_history"] = []


st.set_page_config(
    page_title="CiteBack | Evidence-first paper reading",
    page_icon="◈",
    layout="wide",
)

add_custom_styles()

st.markdown(
    '<div class="product-label">Evidence-first academic reading</div>',
    unsafe_allow_html=True,
)
st.title("CiteBack")
st.markdown(
    '<div class="hero-copy"><strong>Understand the paper. Check the evidence.</strong>'
    "<br>Ask questions and extract key research details without losing the "
    "original page-level context.</div>",
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="trust-strip">◈ Every generated claim is paired with source '
    "evidence. Important conclusions should still be checked against the "
    "original paper.</div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## ◈ CiteBack")
    st.caption("Evidence-first academic reading")
    st.divider()
    st.markdown("### How it works")
    st.write("**1. Upload** a text-based research PDF.")
    st.write("**2. Process** it into page-aware passages.")
    st.write("**3. Explore** with grounded questions or a structured analysis.")
    st.divider()
    st.markdown("### Privacy")
    st.caption(
        "The uploaded PDF is temporarily saved for text extraction and then "
        "deleted. Its processed text remains only in the current app session."
    )

uploaded_file = st.file_uploader(
    "Start with a research paper",
    type=["pdf"],
    help=f"Text-based PDF files up to {MAX_FILE_SIZE_MB} MB are supported.",
)

if uploaded_file is not None:
    pdf_bytes = uploaded_file.getvalue()
    file_size_mb = len(pdf_bytes) / (1024 * 1024)
    file_id = hashlib.sha256(pdf_bytes).hexdigest()

    if st.session_state.get("file_id") != file_id:
        reset_document_state(file_id, uploaded_file.name)

    st.caption(f"Selected: {uploaded_file.name} ({file_size_mb:.1f} MB)")

    if file_size_mb > MAX_FILE_SIZE_MB:
        st.error(f"Please upload a PDF smaller than {MAX_FILE_SIZE_MB} MB.")
    elif st.button("Process paper", type="primary"):
        try:
            with st.spinner("Extracting and preparing the paper..."):
                model = get_embedding_model()
                st.session_state["document"] = process_uploaded_pdf(
                    pdf_bytes,
                    model,
                )
                st.session_state["analysis"] = None
                st.session_state["question_history"] = []

            st.success("The paper is ready.")
        except (PdfReadError, OSError, ValueError) as error:
            st.error(f"The PDF could not be processed: {error}")

document = st.session_state.get("document")

if document is not None:
    page_count = len(document["pages"])
    searchable_page_count = len(document["searchable_pages"])
    chunk_count = len(document["chunks"])

    st.markdown(
        f"### Ready to explore: `{st.session_state['file_name']}`"
    )

    pages_column, searchable_column, chunks_column = st.columns(3)
    pages_column.metric("PDF pages", page_count)
    searchable_column.metric("Searchable pages", searchable_page_count)
    chunks_column.metric("Searchable passages", chunk_count)

    questions_tab, analysis_tab = st.tabs([
        "Ask questions",
        "Paper analysis",
    ])

    with questions_tab:
        st.markdown("### Ask the paper")
        st.caption(
            "CiteBack retrieves the most relevant passages before generating "
            "an answer. Each question is answered independently."
        )

        with st.form("question_form", clear_on_submit=True):
            question = st.text_input(
                "What would you like to understand?",
                placeholder="For example: What are the main limitations?",
            )
            ask_question = st.form_submit_button(
                "Ask CiteBack",
                type="primary",
            )

        if ask_question:
            try:
                with st.spinner("Searching the paper and preparing an answer..."):
                    model = get_embedding_model()
                    search_results = search_chunks(
                        query=question,
                        chunks=document["chunks"],
                        embeddings=document["embeddings"],
                        model=model,
                        top_k=3,
                    )
                    prompt = build_rag_prompt(question, search_results)
                    answer = generate_answer(prompt)

                    st.session_state["question_history"].append({
                        "question": question,
                        "answer": answer,
                        "sources": search_results,
                    })
            except (APIError, ValueError) as error:
                st.error(f"The question could not be answered: {error}")

        # keep earlier questions visible during the current session
        for history_item in reversed(st.session_state["question_history"]):
            with st.chat_message("user"):
                st.write(history_item["question"])

            with st.chat_message("assistant", avatar="◈"):
                st.write(history_item["answer"])

                with st.expander("Inspect supporting passages"):
                    for source_number, source in enumerate(
                        history_item["sources"],
                        start=1,
                    ):
                        st.markdown(
                            f"**Source {source_number} · PDF page "
                            f"{source['page_number']} · relevance "
                            f"{source['score']:.3f}**"
                        )
                        st.write(source["text"])

    with analysis_tab:
        st.markdown("### Evidence-checked paper analysis")
        st.write(
            "Generate the paper type, hypothesis or objective, methodology, "
            "major findings, and limitations with page-level evidence."
        )

        if st.button("Generate analysis", type="primary"):
            try:
                with st.spinner("Analyzing and checking the evidence..."):
                    model = get_embedding_model()
                    st.session_state["analysis"] = analyze_document(
                        chunks=document["chunks"],
                        embeddings=document["embeddings"],
                        embedding_model=model,
                    )
            except (APIError, ValueError) as error:
                st.error(f"The analysis could not be completed: {error}")

        analysis = st.session_state.get("analysis")

        if analysis is not None:
            show_structured_analysis(analysis)
