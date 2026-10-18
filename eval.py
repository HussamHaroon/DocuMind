"""Evaluation: ~30 QA pairs scored by an LLM judge, baseline vs self-check pipeline.

Metrics per in-corpus question: context recall (did retrieval fetch the right info),
faithfulness (is the answer supported by the cited context), relevancy (does it answer
the question). Out-of-scope questions are scored on refusal correctness.
Output: eval/results.md
"""
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import RateLimitError
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from rag import MAX_DISTANCE, NOT_FOUND, build_context, build_selfcheck_graph, load_store, retrieve

QA = [
    # Information Security lectures
    ('What is a firewall and what does it do?', 'network barrier, allow, block, traffic'),
    ('Explain the CIA triad.', 'confidentiality, integrity, availability'),
    ('What is phishing?', 'email, fake, attacker, credentials'),
    ('What is the difference between symmetric and asymmetric encryption?', 'same key, key pair, public, private'),
    ('What is a denial of service attack?', 'overwhelm, resource, service, unavailable'),
    ('What is malware and what types exist?', 'malicious, virus, worm, trojan'),
    ('What is a buffer overflow?', 'memory, bounds, overwrite, input'),
    ('What is access control?', 'permissions, who, resource, authorized'),
    ('What is social engineering?', 'human, manipulation, trick, information'),
    ('What are the goals of cryptography?', 'confidentiality, authentication, integrity, non-repudiation'),
    # HCI chapters
    ('What is requirement analysis in HCI?', 'user needs, requirements, gather, system'),
    ('What is usability?', 'easy, learn, use, effectiveness'),
    ('What data gathering techniques are used in HCI?', 'interview, questionnaire, observation'),
    ('What is iterative design?', 'cycle, prototype, evaluate, refine'),
    ('Why is understanding users important in design?', 'needs, expectations, errors, satisfaction'),
    # Software Construction lectures
    ('What is cohesion in software design?', 'module, single, responsibility, related'),
    ('What is coupling?', 'modules, depend, connection, change'),
    ('What is refactoring?', 'structure, behaviour, improve, code'),
    ('What are code smells?', 'symptom, design, problem, maintainability'),
    ('What is unit testing?', 'single, component, test, isolated'),
    # ABIS notes (ANN / NLP / KR)
    ('How does an artificial neuron work?', 'inputs, weights, sum, activation'),
    ('What is backpropagation?', 'error, gradient, backwards, weights'),
    ('What is an activation function?', 'nonlinear, output, decides, fires'),
    ('What is tokenization in NLP?', 'text, split, words, tokens'),
    ('What is the difference between stemming and lemmatization?', 'suffix, root, dictionary, base form'),
    ('What is an ontology in knowledge representation?', 'concepts, relationships, domain, formal'),
    # atomcamp briefs
    ('What is transfer learning?', 'pretrained, model, reuse, task'),
    ('How do you containerize an app with Docker?', 'Dockerfile, build, image, run'),
    ('What does the Monitor Agent do in Sentinel?', 'logs, telemetry, anomaly, detect'),
    ('What are the deliverables of the final project?', 'proposal, video, repository, form'),
]

OOS = [
    'What is the capital of France?',
    'What is the best pasta recipe?',
    'Who won the last world cup?',
    'What is the weather in Lahore today?',
]

JUDGE_YESNO = ChatPromptTemplate.from_messages([
    ('system', 'You are a strict evaluator. Reply with exactly one word: yes or no.'),
    ('human', '{question}'),
])

RECALL_Q = ('Question the student asked: {q}\n'
            'Reference information the correct answer needs: {ref}\n\n'
            'Retrieved context passages:\n{context}\n\n'
            'Does the context contain the information the reference describes?')
FAITH_Q = ('Context passages the answer was generated from:\n{context}\n\n'
           'Generated answer:\n{answer}\n\n'
           'Is every claim in the answer supported by the context passages, with no added outside facts?')
RELEV_Q = ('Question: {q}\n\nAnswer:\n{answer}\n\nDoes the answer actually address the question?')


def safe_invoke(chain, kw):
    """Invoke a chain, backing off when the Groq free-tier TPM limit kicks in."""
    waits = [15, 30, 45, 60, 90, 120]
    for wait in waits:
        try:
            return chain.invoke(kw)
        except RateLimitError as e:
            m = re.search(r'try again in ([0-9.]+)s', str(e))
            w = float(m.group(1)) + 3 if m else wait
            w = max(w, wait)
            print(f'    rate limited, waiting {w:.0f}s', flush=True)
            time.sleep(w)
    raise RuntimeError('rate limited too many times')


def judge(llm, template, **kw):
    text = safe_invoke(JUDGE_YESNO | llm, {'question': template.format(**kw)}).content.strip().lower()
    return text.startswith('yes')


def get_page_context(store, metas):
    """Pull the chunk text for the cited (source, page) pairs, like the app expander does."""
    docs = []
    seen = set()
    for m in metas:
        key = (m['source'], m['page'])
        if key in seen:
            continue
        seen.add(key)
        where = {'$and': [{'source': {'$eq': m['source']}}, {'page': {'$eq': m['page']}}]}
        hit = store.get(where=where)
        if hit['documents']:
            docs.append(hit['documents'][0])
    return docs


def baseline(store, llm, question):
    """Single retrieve + guard + generate: the no-self-check pipeline."""
    from rag import PROMPT
    docs, scores, metas = retrieve(store, question)
    if min(scores) > MAX_DISTANCE:
        return NOT_FOUND, [], scores, docs
    text = safe_invoke(PROMPT | llm, {'context': build_context(docs, metas),
                                      'question': question}).content
    return text, metas, scores, docs


def selfcheck(store, llm, question):
    graph = build_selfcheck_graph(store, llm)
    init = {'question': question, 'docs': [], 'scores': [], 'metas': [],
            'relevant': False, 'retries': 0, 'answer_text': ''}
    # the graph's internal LLM calls (grade/rephrase) can hit the TPM limit;
    # retry the whole question after a pause instead of crashing the run
    for attempt in range(6):
        try:
            state = graph.invoke(init)
            return state['answer_text'], state['metas'], state['scores'], state['docs']
        except RateLimitError as e:
            m = re.search(r'try again in ([0-9.]+)s', str(e))
            w = float(m.group(1)) + 5 if m else 30 * (attempt + 1)
            print(f'    graph rate limited, waiting {w:.0f}s', flush=True)
            time.sleep(w)
    raise RuntimeError('graph rate limited too many times')


CKPT = Path('eval/progress.json')


def load_ckpt():
    if CKPT.exists():
        import json
        return json.loads(CKPT.read_text(encoding='utf-8'))
    return {'rows': [], 'oos': []}


def save_ckpt(data):
    import json
    CKPT.parent.mkdir(exist_ok=True)
    CKPT.write_text(json.dumps(data), encoding='utf-8')


def main():
    load_dotenv()
    llm = ChatGroq(model='openai/gpt-oss-20b', temperature=0)
    store = load_store()

    ckpt = load_ckpt()
    done = [r['q'] for r in ckpt['rows']]
    rows = ckpt['rows']
    for i, (q, ref) in enumerate(QA, 1):
        if q in done:
            continue
        t0 = time.time()
        b_text, b_metas, b_scores, b_docs = baseline(store, llm, q)
        s_text, s_metas, s_scores, s_docs = selfcheck(store, llm, q)

        b_ctx = build_context(b_docs, b_metas)[:2500] if b_docs else '(nothing retrieved)'
        s_ctx = build_context(s_docs, s_metas)[:2500] if s_docs else '(nothing retrieved)'

        b_refused = b_text.startswith("I couldn't find")
        s_refused = s_text.startswith("I couldn't find")

        b_recall = 0 if b_refused else judge(llm, RECALL_Q, q=q, ref=ref, context=b_ctx)
        b_faith = 0 if b_refused else judge(llm, FAITH_Q, context=b_ctx, answer=b_text)
        b_relev = 0 if b_refused else judge(llm, RELEV_Q, q=q, answer=b_text)

        s_recall = 0 if s_refused else judge(llm, RECALL_Q, q=q, ref=ref, context=s_ctx)
        s_faith = 0 if s_refused else judge(llm, FAITH_Q, context=s_ctx, answer=s_text)
        s_relev = 0 if s_refused else judge(llm, RELEV_Q, q=q, answer=s_text)

        rows.append(dict(q=q, ref=ref, b=(b_recall, b_faith, b_relev, b_refused),
                         s=(s_recall, s_faith, s_relev, s_refused),
                         b_dist=min(b_scores) if b_scores else None,
                         s_dist=min(s_scores) if s_scores else None))
        save_ckpt({'rows': rows, 'oos': oos_rows if 'oos_rows' in dir() else []})
        print(f'[{i:2d}/{len(QA)}] {q[:50]:52s} base={sum(rows[-1]["b"][:3])}/3 self={sum(rows[-1]["s"][:3])}/3  ({time.time()-t0:.0f}s)', flush=True)
        time.sleep(20)

    oos_rows = ckpt['oos']
    oos_done = [r[0] for r in oos_rows]
    for q in OOS:
        if q in oos_done:
            continue
        b_text, _, b_scores, _ = baseline(store, llm, q)
        s_text, _, s_scores, _ = selfcheck(store, llm, q)
        b_ok = b_text.startswith("I couldn't find")
        s_ok = s_text.startswith("I couldn't find")
        oos_rows.append((q, b_ok, s_ok, min(b_scores) if b_scores else None))
        save_ckpt({'rows': rows, 'oos': oos_rows})
        print(f'[oos] {q[:50]:52s} base_refused={b_ok} self_refused={s_ok}', flush=True)

    write_report(rows, oos_rows)


def pct(vals):
    return f'{100 * sum(vals) / max(len(vals), 1):.0f}%' if vals else 'n/a'


def write_report(rows, oos_rows):
    out = ['# DocuMind evaluation results\n']
    out.append(f'{len(rows)} in-corpus questions + {len(oos_rows)} out-of-scope questions. '
               'Both pipelines use the same retriever (MiniLM + Chroma, top-4) and the same '
               'generator (gpt-oss-20b, temperature 0). The baseline does a single retrieve; '
               'the self-check pipeline adds the LangGraph grade/rephrase/retry loop. '
               'All three metrics are scored yes/no by the same LLM acting as judge.\n')

    out.append('| Metric | Baseline | Self-check |')
    out.append('|---|---|---|')
    out.append('| Context recall | ' + pct([r['b'][0] for r in rows]) + ' | ' + pct([r['s'][0] for r in rows]) + ' |')
    out.append('| Faithfulness | ' + pct([r['b'][1] for r in rows]) + ' | ' + pct([r['s'][1] for r in rows]) + ' |')
    out.append('| Relevancy | ' + pct([r['b'][2] for r in rows]) + ' | ' + pct([r['s'][2] for r in rows]) + ' |')
    out.append('| Refusal on out-of-scope | ' + pct([r[1] for r in oos_rows]) + ' | ' + pct([r[2] for r in oos_rows]) + ' |')
    out.append('')

    out.append('## Per-question detail\n')
    out.append('| # | Question | Base (R/F/Rel) | Self (R/F/Rel) | Best dist B/S |')
    out.append('|---|---|---|---|---|')
    for i, r in enumerate(rows, 1):
        b = f"{r['b'][0]}{r['b'][1]}{r['b'][2]}" + (' REFUSED' if r['b'][3] else '')
        s = f"{r['s'][0]}{r['s'][1]}{r['s'][2]}" + (' REFUSED' if r['s'][3] else '')
        bd = f"{r['b_dist']:.2f}" if r['b_dist'] else '-'
        sd = f"{r['s_dist']:.2f}" if r['s_dist'] else '-'
        out.append(f"| {i} | {r['q']} | {b} | {s} | {bd} / {sd} |")
    out.append('')

    out.append('## Out-of-scope questions\n')
    for q, b_ok, s_ok, dist in oos_rows:
        out.append(f'- {q}  (best distance {dist:.2f})  baseline refused: {b_ok}, self-check refused: {s_ok}')
    out.append('')

    out.append('## Failure analysis\n')
    fails = [r for r in rows if sum(r['s'][:3]) < 3]
    if not fails:
        out.append('No in-corpus question failed all criteria on the self-check pipeline.')
    for r in fails:
        bd = f"{r['b_dist']:.2f}" if r['b_dist'] else '-'
        sd = f"{r['s_dist']:.2f}" if r['s_dist'] else '-'
        out.append(f"- **{r['q']}** (base {r['b'][:3]}, self {r['s'][:3]}, distances {bd} / {sd})")
    Path('eval').mkdir(exist_ok=True)
    Path('eval/results.md').write_text('\n'.join(out), encoding='utf-8')
    print('\nwrote eval/results.md')


if __name__ == '__main__':
    main()
