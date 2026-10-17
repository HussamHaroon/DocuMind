# DocuMind — RAG Q&A over Course PDFs, with citations

DocuMind answers questions about a folder of course PDFs using retrieval-augmented
generation. Every answer cites the exact file and page it came from, and questions the
documents don't cover get refused instead of guessed.

**Stack:** pdfplumber → LangChain → all-MiniLM-L6-v2 embeddings → Chroma → gpt-oss-20b (Groq) → Streamlit

```
data/pdfs/*.pdf ──ingest.py──> chunks + MiniLM embeddings ──> chroma_db/
                                     │
question ──> rag.py: top-4 retrieve ──> similarity guard ──> gpt-oss-20b (Groq)
                                                        └──> app.py (Streamlit)
```

The corpus in `data/pdfs/` is 22 of my own 5th-semester course PDFs: Information
Security lectures, HCI chapters, Software Construction lectures, and AI notes
(ANN / NLP / knowledge representation), converted to PDF where needed.

## Quick start

```
pip install -r requirements.txt
copy .env.example .env      # put your GROQ_API_KEY in it (console.groq.com, free)
python ingest.py            # chunks + embeds data/pdfs/ into chroma_db/
streamlit run app.py        # ask questions in the browser
```

Terminal alternative: `python rag.py "what is phishing"`

## How it works

- **Ingest** parses every PDF page by page with pdfplumber, so each chunk knows which
  file and page it came from. That metadata is what becomes the citation.
- **Retrieve** embeds the question with the same MiniLM model and pulls the top-4
  nearest chunks from Chroma. The corpus was calibrated so on-topic questions land at
  L2 distance <= ~1.25 and off-corpus questions >= ~1.40; a guard at 1.32 refuses the
  latter before any LLM call.
- **Generate** asks gpt-oss-20b (temperature 0, Groq API) to answer strictly from the
  retrieved passages. Citations are appended by the app from chunk metadata, so the
  model cannot invent a file or page number.
- **Self-check loop** (LangGraph): after retrieval, a grader step checks whether the
  passages actually contain the answer. If not, the question is rephrased with
  different keywords and retrieval runs once more; only if that also fails does
  DocuMind refuse. So an awkwardly worded question still gets answered, while truly
  off-corpus questions never reach the answer step.

## Evaluation

`eval/results.md` holds the question-set results (context recall, faithfulness,
answer relevance) comparing the base pipeline with the self-checking variant.

## Limitations

- Scanned / image-only PDFs are skipped at ingest (they need OCR); one such file is
  in the corpus and named in the ingest output.
- Two full textbooks sit outside the index in `data/books/` on purpose: a whole book
  floods the index and dilutes retrieval for course-specific questions.
- English documents only, single-turn questions, no chat memory.
