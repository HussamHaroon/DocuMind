"""DocuMind retrieval + generation with a LangGraph self-check loop.

retrieve -> grade -> (relevant) generate
                |-> (not relevant, first try) rephrase -> retrieve
                |-> (not relevant, second try) refuse
Terminal test: python rag.py "your question"."""
import sys
from typing import TypedDict

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

DB_DIR = 'chroma_db'
TOP_K = 4
MAX_RETRIES = 1
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

GRADE_PROMPT = ChatPromptTemplate.from_messages([
    ('system', 'Judge whether the context passages contain information relevant to answering '
               'the question. Be lenient: partial coverage still counts as relevant. '
               'Reply with exactly one word: yes or no.'),
    ('human', 'Question: {question}\n\nContext:\n{context}'),
])

REPHRASE_PROMPT = ChatPromptTemplate.from_messages([
    ('system', 'Rewrite the question with different wording but the same meaning, using '
               'more search-friendly keywords. Reply with only the rewritten question.'),
    ('human', '{question}'),
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


# ---------------- LangGraph self-check loop ----------------

class SelfCheckState(TypedDict):
    question: str
    docs: list
    scores: list
    metas: list
    relevant: bool
    retries: int
    answer_text: str


def _citations(metas):
    seen, cited = set(), []
    for meta in metas:
        key = (meta['source'], meta['page'])
        if key not in seen:
            seen.add(key)
            cited.append(f"{meta['source']}, p.{meta['page']}")
    return '; '.join(cited)


def build_selfcheck_graph(store, llm):
    def retrieve_node(state):
        docs, scores, metas = retrieve(store, state['question'])
        return {'docs': docs, 'scores': scores, 'metas': metas}

    def grade_node(state):
        # distance guard first: if the best hit is still far, this is off-corpus,
        # no point asking the LLM about it
        if not state['scores'] or min(state['scores']) > MAX_DISTANCE:
            return {'relevant': False}
        verdict = (GRADE_PROMPT | llm).invoke(
            {'question': state['question'],
             'context': build_context(state['docs'], state['metas'])}).content.strip().lower()
        return {'relevant': verdict.startswith('yes')}

    def rephrase_node(state):
        new_q = (REPHRASE_PROMPT | llm).invoke({'question': state['question']}).content.strip()
        print(f'[self-check] retrying with rephrased question: {new_q}')
        return {'question': new_q, 'retries': state['retries'] + 1}

    def generate_node(state):
        response = (PROMPT | llm).invoke(
            {'context': build_context(state['docs'], state['metas']),
             'question': state['question']})
        text = response.content + '\n\nSources: ' + _citations(state['metas'])
        return {'answer_text': text}

    def refuse_node(state):
        return {'answer_text': NOT_FOUND}

    def route_after_grade(state):
        if state['relevant']:
            return 'generate'
        if state['retries'] < MAX_RETRIES:
            return 'rephrase'
        return 'refuse'

    builder = StateGraph(SelfCheckState)
    builder.add_node('retrieve', retrieve_node)
    builder.add_node('grade', grade_node)
    builder.add_node('rephrase', rephrase_node)
    builder.add_node('generate', generate_node)
    builder.add_node('refuse', refuse_node)
    builder.set_entry_point('retrieve')
    builder.add_edge('retrieve', 'grade')
    builder.add_conditional_edges('grade', route_after_grade,
                                  {'generate': 'generate', 'rephrase': 'rephrase',
                                   'refuse': 'refuse'})
    builder.add_edge('rephrase', 'retrieve')
    builder.add_edge('generate', END)
    builder.add_edge('refuse', END)
    return builder.compile()


def answer(question, store=None, llm=None):
    """Full pipeline: retrieve, grade, rephrase-retry once, generate. Returns (answer_text, sources, scores)."""
    store = store or load_store()
    load_dotenv()
    llm = llm or ChatGroq(model='openai/gpt-oss-20b', temperature=0)

    graph = build_selfcheck_graph(store, llm)
    state = graph.invoke({'question': question, 'docs': [], 'scores': [], 'metas': [],
                          'relevant': False, 'retries': 0, 'answer_text': ''})
    return state['answer_text'], state['metas'], state['scores']


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise SystemExit('usage: python rag.py "your question"')
    q = ' '.join(sys.argv[1:])
    text, sources, scores = answer(q)
    print('ANSWER:\n' + text)
    if text.strip() != NOT_FOUND:
        print('\nSOURCES:')
        for meta, score in zip(sources, scores):
            print(f"  {meta['source']} p.{meta['page']}  (distance {score:.3f})")
