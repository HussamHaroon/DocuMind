# DocuMind evaluation results

30 in-corpus questions + 4 out-of-scope questions. Both pipelines use the same retriever (MiniLM + Chroma, top-4) and the same generator (gpt-oss-20b, temperature 0). The baseline does a single retrieve; the self-check pipeline adds the LangGraph grade/rephrase/retry loop. All three metrics are scored yes/no by the same LLM acting as judge.

| Metric | Baseline | Self-check |
|---|---|---|
| Context recall | 60% | 60% |
| Faithfulness | 47% | 50% |
| Relevancy | 70% | 60% |
| Refusal on out-of-scope | 100% | 100% |

## Per-question detail

| # | Question | Base (R/F/Rel) | Self (R/F/Rel) | Best dist B/S |
|---|---|---|---|---|
| 1 | What is a firewall and what does it do? | TrueTrueTrue | TrueTrueFalse | 1.03 / 1.03 |
| 2 | Explain the CIA triad. | TrueFalseTrue | TrueFalseTrue | 1.15 / 1.15 |
| 3 | What is phishing? | TrueTrueTrue | TrueTrueTrue | 0.69 / 0.69 |
| 4 | What is the difference between symmetric and asymmetric encryption? | TrueTrueTrue | TrueTrueTrue | 0.46 / 0.46 |
| 5 | What is a denial of service attack? | TrueTrueTrue | TrueTrueTrue | 0.64 / 0.64 |
| 6 | What is malware and what types exist? | TrueTrueTrue | TrueTrueTrue | 0.55 / 0.55 |
| 7 | What is a buffer overflow? | 000 REFUSED | 000 REFUSED | 1.39 / 1.46 |
| 8 | What is access control? | TrueFalseTrue | TrueTrueTrue | 0.85 / 0.85 |
| 9 | What is social engineering? | TrueTrueTrue | TrueTrueTrue | 0.83 / 0.83 |
| 10 | What are the goals of cryptography? | FalseFalseFalse | FalseFalseFalse | 0.78 / 0.78 |
| 11 | What is requirement analysis in HCI? | TrueTrueTrue | TrueTrueTrue | 0.25 / 0.25 |
| 12 | What is usability? | TrueTrueTrue | TrueTrueTrue | 0.86 / 0.86 |
| 13 | What data gathering techniques are used in HCI? | 000 REFUSED | 000 REFUSED | 0.78 / 0.74 |
| 14 | What is iterative design? | FalseTrueTrue | 000 REFUSED | 0.92 / 0.92 |
| 15 | Why is understanding users important in design? | FalseTrueTrue | FalseTrueTrue | 0.94 / 0.94 |
| 16 | What is cohesion in software design? | 000 REFUSED | 000 REFUSED | 0.95 / 0.93 |
| 17 | What is coupling? | 000 REFUSED | 000 REFUSED | 1.25 / 0.96 |
| 18 | What is refactoring? | TrueTrueTrue | TrueTrueTrue | 1.03 / 1.03 |
| 19 | What are code smells? | 000 REFUSED | 000 REFUSED | 1.18 / 1.16 |
| 20 | What is unit testing? | TrueTrueTrue | TrueFalseTrue | 1.08 / 1.08 |
| 21 | How does an artificial neuron work? | TrueFalseTrue | TrueFalseTrue | 0.73 / 0.73 |
| 22 | What is backpropagation? | 000 REFUSED | 000 REFUSED | 1.15 / 0.91 |
| 23 | What is an activation function? | TrueTrueTrue | TrueTrueTrue | 0.90 / 0.90 |
| 24 | What is tokenization in NLP? | TrueFalseTrue | TrueTrueFalse | 0.88 / 0.88 |
| 25 | What is the difference between stemming and lemmatization? | 000 REFUSED | 000 REFUSED | 1.27 / 1.09 |
| 26 | What is an ontology in knowledge representation? | FalseFalseTrue | FalseTrueTrue | 0.54 / 0.54 |
| 27 | What is transfer learning? | TrueFalseTrue | TrueFalseTrue | 1.18 / 1.18 |
| 28 | How do you containerize an app with Docker? | TrueFalseTrue | TrueFalseTrue | 0.84 / 0.84 |
| 29 | What does the Monitor Agent do in Sentinel? | TrueTrueTrue | TrueTrueTrue | 0.86 / 0.86 |
| 30 | What are the deliverables of the final project? | 000 REFUSED | 000 REFUSED | 0.97 / 1.04 |

## Out-of-scope questions

- What is the capital of France?  (best distance 1.66)  baseline refused: True, self-check refused: True
- What is the best pasta recipe?  (best distance 1.56)  baseline refused: True, self-check refused: True
- Who won the last world cup?  (best distance 1.74)  baseline refused: True, self-check refused: True
- What is the weather in Lahore today?  (best distance 1.60)  baseline refused: True, self-check refused: True

## Failure analysis

- **What is a firewall and what does it do?** (base [True, True, True], self [True, True, False], distances 1.03 / 1.03)
- **Explain the CIA triad.** (base [True, False, True], self [True, False, True], distances 1.15 / 1.15)
- **What is a buffer overflow?** (base [0, 0, 0], self [0, 0, 0], distances 1.39 / 1.46)
- **What are the goals of cryptography?** (base (False, False, False), self (False, False, False), distances 0.78 / 0.78)
- **What data gathering techniques are used in HCI?** (base (0, 0, 0), self (0, 0, 0), distances 0.78 / 0.74)
- **What is iterative design?** (base (False, True, True), self (0, 0, 0), distances 0.92 / 0.92)
- **Why is understanding users important in design?** (base (False, True, True), self (False, True, True), distances 0.94 / 0.94)
- **What is cohesion in software design?** (base (0, 0, 0), self (0, 0, 0), distances 0.95 / 0.93)
- **What is coupling?** (base (0, 0, 0), self (0, 0, 0), distances 1.25 / 0.96)
- **What are code smells?** (base (0, 0, 0), self (0, 0, 0), distances 1.18 / 1.16)
- **What is unit testing?** (base (True, True, True), self (True, False, True), distances 1.08 / 1.08)
- **How does an artificial neuron work?** (base (True, False, True), self (True, False, True), distances 0.73 / 0.73)
- **What is backpropagation?** (base (0, 0, 0), self (0, 0, 0), distances 1.15 / 0.91)
- **What is tokenization in NLP?** (base (True, False, True), self (True, True, False), distances 0.88 / 0.88)
- **What is the difference between stemming and lemmatization?** (base (0, 0, 0), self (0, 0, 0), distances 1.27 / 1.09)
- **What is an ontology in knowledge representation?** (base (False, False, True), self (False, True, True), distances 0.54 / 0.54)
- **What is transfer learning?** (base (True, False, True), self (True, False, True), distances 1.18 / 1.18)
- **How do you containerize an app with Docker?** (base (True, False, True), self (True, False, True), distances 0.84 / 0.84)
- **What are the deliverables of the final project?** (base (0, 0, 0), self (0, 0, 0), distances 0.97 / 1.04)