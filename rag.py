"""DocuMind retrieval + generation. Terminal test: python rag.py "your question"."""
import os
import sys

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate

DB_DIR = 'chroma_db'
TOP_K = 4
# langchain-chroma returns L2 distance: smaller = more similar. Calibrated on the
# real corpus: on-topic questions land <= ~1.25, off-corpus >= ~1.40, so 1.32
# sits in the empty gap between them.
MAX_DISTANCE = 1.32
NOT_FOUND = "I couldn't find this in your documents."

PROMPT = ChatPromptTemplate.from_messages([
    ('system',
     'You are DocuMind, a QA assistant that answers ONLY from the provided context '
     'passages taken from a student\'s course PDFs.\n'
     'Rules:\n'
     '1. Answer using only the context passages. Do not use outside knowledge.\n'
     '2. Do NOT mention files, pages or citations in your answer, plain prose only.\n'
     '   Sources are attached automatically after you answer.\n'
     '3. If the context does not contain the answer, reply exactly:\n'
     '   "I couldn\'t find this in your documents." and nothing else.\n'
     'Keep answers under 120 words unless asked for more.'),
    ('human', '{context}\n\nQuestion: {question}'),
])


def load_store():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return Chroma(persist_directory=DB_DIR, collection_name='documind',
                  embedding_function=embeddings)


def retrieve(store, question, k=TOP_K):
    hits = store.similarity_search_with_score(question, k=k)
    docs, scores, metas = [], [], []
    for doc, distance in hits:
        docs.append(doc.page_content)
        scores.append(distance)
        metas.append(doc.metadata)
    return docs, scores, metas


def build_context(docs, metas):
    parts = []
    for text, meta in zip(docs, metas):
        parts.append(f"[SOURCE: {meta['source']}, p.{meta['page']}]\n{text}")
    return '\n\n'.join(parts)


def answer(question, store=None, llm=None):
    """Full pipeline: retrieve, guard, generate. Returns (answer_text, sources, scores)."""
    store = store or load_store()
    load_dotenv()
    llm = llm or ChatGroq(model='openai/gpt-oss-20b', temperature=0)

    docs, scores, metas = retrieve(store, question)
    if min(scores) > MAX_DISTANCE:
        # nothing even close in the corpus: refuse without burning an LLM call
        return NOT_FOUND, [], scores

    chain = PROMPT | llm
    response = chain.invoke({'context': build_context(docs, metas), 'question': question})

    # citations appended from retrieval metadata: the LLM never writes them,
    # so it cannot invent a file or page number
    seen, cited = set(), []
    for meta in metas:
        key = (meta['source'], meta['page'])
        if key not in seen:
            seen.add(key)
            cited.append(f"{meta['source']}, p.{meta['page']}")
    text = response.content + '\n\nSources: ' + '; '.join(cited)
    return text, metas, scores


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise SystemExit('usage: python rag.py "your question"')
    q = ' '.join(sys.argv[1:])
    text, sources, scores = answer(q)
    print('ANSWER:\n' + text)
    print('\nSOURCES:')
    for meta, score in zip(sources, scores):
        print(f"  {meta['source']} p.{meta['page']}  (distance {score:.3f})")
