"""Ingest data/pdfs/*.pdf into a Chroma vector store (one chunked, embedded collection).

Every chunk keeps [source, page] metadata, which is where the citations later come from.
Run:  python ingest.py
"""
from pathlib import Path

import pdfplumber
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_DIR = Path('data/pdfs')
DB_DIR = 'chroma_db'
CHUNK_SIZE = 900       # ~ a solid paragraph or two, small enough to stay on one topic
CHUNK_OVERLAP = 150    # so a sentence cut in half still appears whole in a neighbour chunk


def extract_pages(pdf_path: Path):
    """Yield (page_number, text) for every page that actually has text."""
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or '').strip()
            if text:
                yield i, text


def main():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    documents = []
    kept, skipped = [], []
    for pdf_path in sorted(PDF_DIR.glob('*.pdf')):
        pages = list(extract_pages(pdf_path))
        if not pages:
            skipped.append(pdf_path.name)   # scanned PDF with no selectable text
            continue
        kept.append(pdf_path.name)
        for page_no, text in pages:
            for chunk in splitter.split_text(text):
                documents.append(Document(page_content=chunk,
                                          metadata={'source': pdf_path.name, 'page': page_no}))

    if not documents:
        raise SystemExit('No text found in any PDF. Scanned PDFs need OCR first.')

    # same model must embed docs here and questions in rag.py
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=DB_DIR,
        collection_name='documind',
    )

    print(f'kept {len(kept)} PDFs, skipped {len(skipped)} (no text): {skipped}')
    print(f'{len(documents)} chunks -> {DB_DIR}')
    n = documents[0].metadata
    print(f"sample chunk metadata: source={n['source']} page={n['page']}")


if __name__ == '__main__':
    main()
