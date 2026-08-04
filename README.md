# CiteBack

> **Understand the paper. Check the evidence.**

CiteBack is an evidence-first academic paper assistant for students. It turns a text-based research PDF into searchable, page-aware passages, answers questions using retrieved evidence, and extracts structured research details without hiding where its claims came from.

Unlike a generic PDF chatbot, CiteBack treats trust as a product requirement. Generated claims include supporting passages and PDF page numbers, missing information is reported honestly, and a second verification pass narrows or removes claims that exceed their evidence.

## Why I Built It

Academic papers are often long, dense, and sometimes hard to undertand. Students may need to locate a hypothesis, understand a methodology, compare findings, or identify limitations under tight coursework deadlines.

I built CiteBack to explore how retrieval-augmented generation can make that process more approachable while still directing students back to the original source.

## Core Features

- Upload and process one text-based PDF at a time.
- Extract text while retaining user-facing PDF page numbers.
- Remove repeated headers, footers, and bibliography content from search.
- Split pages into overlapping, context-aware passages.
- Retrieve passages using local semantic embeddings.
- Ask grounded questions about the uploaded paper.
- Extract the document type, hypothesis or objective, methodology, findings, and limitations.
- Separate author-stated limitations from AI-suggested limitations.
- Display supporting passages and PDF page citations.
- Report when requested information cannot be found.
- Delete the temporary uploaded PDF immediately after extraction.
- Limit hosted AI usage per browser session in the public demo.

## Trust and Validation

CiteBack uses several checks before displaying a structured analysis:

1. **Grounded generation** — the model receives only retrieved passages from the uploaded paper.
2. **Citation validation** — every cited page must have been retrieved.
3. **Passage validation** — every quoted passage must exist on its claimed page.
4. **Semantic verification** — a second model pass narrows or removes claims that are broader than their evidence.
5. **Final validation** — citations are checked again after verification.

The application assists reading; it does not replace the original paper or guarantee academic correctness. Important claims should always be checked against the source.

## How It Works

```mermaid
flowchart LR
    A[Upload PDF] --> B[Extract page-aware text]
    B --> C[Clean repeated and reference content]
    C --> D[Create overlapping passages]
    D --> E[Generate local embeddings]
    E --> F[Retrieve relevant evidence]
    F --> G[Generate grounded response]
    G --> H[Validate and display citations]
```

## Technology

- **Python** for the application and processing pipeline
- **Streamlit** for the interactive interface
- **pypdf** for page-aware PDF extraction
- **LangChain text splitters** for context-aware chunking
- **Sentence Transformers** for local embeddings
- **NumPy** for similarity search
- **Groq and Llama 3.3 70B** for grounded generation and verification
- **pytest** for automated tests

## Run Locally

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install the requirements

```bash
.venv/bin/python -m pip install -r requirements.txt
```

### 3. Add the Groq API key

Create a local `.env` file in the project folder:

```text
GROQ_API_KEY=your_key_here
```

The `.env` file is ignored by Git and should never be committed.

### 4. Start CiteBack

```bash
.venv/bin/python -m streamlit run app.py
```

## Run the Tests

```bash
.venv/bin/python -m pytest
```

The test suite covers PDF extraction, cleaning, chunking, prompt construction, API request behavior, structured analysis, evidence validation, semantic verification, and interface startup.

## Current Scope

- English-language PDFs with selectable text
- One paper per session
- Page-level citations based on PDF page order
- Local in-memory document processing
- No OCR for scanned documents yet
- Demo limit of 10 questions and 2 structured analyses per browser session

## Deploy to Streamlit Community Cloud

1. Commit and push the project to GitHub.
2. Choose the repository, the `main` branch, and `app.py` as the entrypoint.
3. Open **Advanced settings** and select Python 3.11 to match local development.
4. Add the Groq key through Streamlit Secrets rather than GitHub:

```toml
GROQ_API_KEY = "your_key_here"
```

5. Choose an available app URL and deploy.

The public demo uses per-session limits to reduce accidental API usage. These limits can be reset by starting a new browser session, so a production release would also require accounts, server-side rate limiting, and provider spending controls.

## Planned Improvements

- Add OCR support for scanned papers.
- Compare findings across multiple papers.
- Export structured notes with citations.
- Add evaluation datasets for retrieval and answer quality.
- Add user accounts and server-wide usage limits.

## Privacy

Uploaded files are written to a temporary location only because the PDF extractor requires a file path. The temporary file is deleted immediately after extraction. Processed text remains in memory for the current Streamlit session and is not intentionally stored permanently by CiteBack.

The uploaded PDF file itself is not sent to Groq. When a student asks a question or generates a structured analysis, CiteBack sends relevant retrieved text passages to the Groq API for response generation. Users should only upload documents they have permission to process and should not upload confidential, sensitive, or personally identifying material.
