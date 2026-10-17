"""DocuMind Streamlit app: ask questions about your course PDFs, get cited answers."""
import os

import streamlit as st
from dotenv import load_dotenv

from rag import MAX_DISTANCE, NOT_FOUND, answer, load_store

load_dotenv()

st.set_page_config(page_title='DocuMind', page_icon='📚', layout='wide')
st.title('📚 DocuMind')
st.caption('RAG question answering over your course PDFs, with citations.')

if not os.path.exists('chroma_db'):
    st.warning('No index found. Put PDFs in data/pdfs/ and run `python ingest.py` first.')
    st.stop()


@st.cache_resource
def get_store():
    return load_store()


@st.cache_resource
def get_llm():
    from langchain_groq import ChatGroq
    return ChatGroq(model='openai/gpt-oss-20b', temperature=0)


store = get_store()
llm = get_llm()

with st.sidebar:
    st.header('Index')
    try:
        count = store._collection.count()
    except Exception:
        count = '?'
    st.metric('Chunks indexed', count)
    st.caption('python ingest.py rebuilds it')
    st.divider()
    st.caption(f'similarity guard: distance > {MAX_DISTANCE} -> refusal')

if 'history' not in st.session_state:
    st.session_state.history = []

question = st.chat_input('Ask something from your course documents...')

if question:
    st.session_state.history.append(('user', question, [], []))
    with st.spinner('retrieving...'):
        text, sources, scores = answer(question, store=store, llm=llm)
    st.session_state.history.append(('bot', text, sources, scores))

for role, text, sources, scores in st.session_state.history:
    with st.chat_message(role):
        st.markdown(text)
        for meta, score in zip(sources, scores):
            with st.expander(f"📄 {meta['source']} — p.{meta['page']}  (distance {score:.3f})"):
                where = {'$and': [{'source': {'$eq': meta['source']}},
                                  {'page': {'$eq': meta['page']}}]}
                hit = store.get(where=where)
                for doc in hit['documents'][:1]:
                    st.write(doc[:600] + ' ...')
