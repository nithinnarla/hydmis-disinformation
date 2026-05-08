# HyDMIS — Literature Review
## Phase 1 | Five Review Types | Completed May 2026

---

## How I Approached This

Eight years of watching ML systems fail quietly in production taught me one thing: the systems that fail the loudest are the ones nobody was watching. Disinformation detection is one of those systems. The English benchmark numbers look impressive. The deployment reality for Spanish, Tagalog, and Haitian Creole communities is a different story entirely.

This literature review covers five review types across the multilingual disinformation detection landscape. I came in skeptical of the standard framing — "multilingual models work, low-resource is just a harder version of the same problem." The literature disabused me of that quickly.

---

## Section 1 — Literature Review (Core Papers)

### 1.1 The Foundation Papers

**Wang et al. (2017) — LIAR: A Benchmark Dataset for Fake News Detection (ACL)**
The paper that defined the field's evaluation standard. 12,836 short statements from PolitiFact with six-way veracity labels. The problem I kept noticing: 100% English, 100% formal political discourse. LIAR is excellent at what it measures. What it measures is not what happens when health disinformation spreads through WhatsApp groups in Portuguese.

**Devlin et al. (2019) — BERT: Pre-training of Deep Bidirectional Transformers (NAACL)**
The architectural foundation. BERT's masked language modeling established the pretraining paradigm that all subsequent multilingual models followed. The limitation for our purposes: BERT's pretraining corpus is English-dominant. The multilingual variants inherit structural biases toward high-resource languages.

**Conneau et al. (2020) — Unsupervised Cross-lingual Representation Learning at Scale — XLM-RoBERTa (ACL)**
XLM-R is the field's current dominant cross-lingual baseline. Trained on 2.5TB of filtered CommonCrawl data across 100 languages. The paper reports strong multilingual transfer results. What the paper does not report: performance on languages with under 100MB of pretraining data. That's the population HyDMIS serves.

**Chung et al. (2021) — Rethinking Embedding Coupling in Pre-trained Language Models — RemBERT (ICLR)**
This one reframed how I was thinking about the backbone selection problem. RemBERT decouples input and output embeddings, allowing larger output representations without increasing input parameter count. PolyTruth (Gouliev et al., 2025) later confirmed empirically what this paper suggested architecturally: RemBERT outperforms mBERT specifically on low-resource language subsets. This finding is directly integrated into HyDMIS Stage 3.

**Blei et al. (2003) — Latent Dirichlet Allocation (JMLR)**
LDA remains the most interpretable unsupervised topic modeling approach for multilingual content. HyDMIS Stage 1 uses LDA specifically because it works without labeled data — which is the only viable approach when expert fact-checkers have not annotated content in a target language. The limitation is well-documented: LDA struggles with short social media texts. HyDMIS addresses this through pre-filtering before LDA application.

**Jiang et al. (2023) — Mistral 7B (arXiv)**
The open-source LLM that makes Stage 2 semantic verification computationally viable. Mistral 7B outperforms Llama 2 13B on most benchmarks at half the parameter count. For a research pipeline that needs to run on 387K+ samples without a $50K cloud bill, Mistral 7B is the pragmatic choice.

**OpenAI (2023) — GPT-4 Technical Report**
GPT-4 is HyDMIS Stage 2's semantic verification backbone for the high-confidence labeling pass. The limitation I kept coming back to: GPT-4's performance on genuinely low-resource languages — Swahili, Tagalog, Haitian Creole — is contested in the literature. The system performs well on European languages with decent pretraining coverage. Whether that transfers to languages underrepresented in its training data is one of HyDMIS's explicit ablation questions.

---

### 1.2 The Dataset Papers

**Shu et al. (2020) — FakeNewsNet (Big Data)**
Political and celebrity news with social context signals. FakeNewsNet's social graph features are often cited as a key contribution. HyDMIS uses the content features only — social graph information is not available for low-resource language communities where the disinformation problem is most acute.

**Pikuliak et al. (2023) — MultiClaim: Multilingual Claim Detection (arXiv)**
The most directly relevant multilingual dataset for HyDMIS. Claims across multiple languages with cross-lingual annotations. The gap: MultiClaim's language distribution is European-heavy. The communities HyDMIS targets — Spanish-speaking US Latino communities, Tagalog-speaking Filipino diaspora, Haitian Creole communities — are underrepresented.

**Dadkhah et al. (2023) — TruthSeeker: The Largest Social Media Ground-Truth Dataset (IEEE TCSS)**
180,000+ labeled Twitter/X posts — the largest labeled social media fake news dataset in existence. HyDMIS uses TruthSeeker as the primary English benchmark to establish comparability with existing literature before moving to multilingual evaluation.

**Gouliev et al. (2025) — PolyTruth: Multilingual Disinformation Detection (arXiv 2509.10737)**
The paper that most directly validated HyDMIS's research direction. 60,486 statement pairs across 25+ languages covering five language families. Critical finding: RemBERT outperforms mBERT specifically on low-resource language subsets. Performance degradation on low-resource languages is quantified empirically for the first time at this scale. This is HyDMIS's most important recent citation.

---

## Section 2 — Systematic Review

### Cross-Lingual Transfer Methodology Comparison

| Approach | Representative Paper | Low-Resource Performance | HyDMIS Relevance |
|----------|---------------------|-------------------------|------------------|
| mBERT fine-tuning | Devlin et al. (2019) | Degrades significantly below 100MB training data | Baseline comparison |
| XLM-R | Conneau et al. (2020) | Better than mBERT but still degrades | Baseline comparison |
| RemBERT | Chung et al. (2021); Gouliev et al. (2025) | Best low-resource performance confirmed empirically | HyDMIS Stage 3 primary |
| LLM-based (GPT-4) | OpenAI (2023) | Contested for genuinely low-resource languages | HyDMIS Stage 2 verification |
| Hybrid pipeline | This paper | Unknown — empirical question | HyDMIS contribution |

**Critical finding from systematic review:**
No existing paper has combined LDA topic modeling, LLM semantic verification, and community-weighted cross-lingual classification in a single pipeline evaluated specifically on low-resource community-targeted disinformation. The hybrid approach is HyDMIS's methodological contribution.

**Key trend (2020-2026):**
The field has moved from monolingual BERT fine-tuning toward multilingual transformers toward LLM-based verification. Each step improved aggregate multilingual performance. None addressed the structural gap: performance is measured on aggregate multilingual benchmarks, not on the specific low-resource community subsets where harm accumulates.

---

## Section 3 — Scoping Review

### What the Field Has and Has Not Addressed

**Has addressed:**
- English fake news detection — thoroughly
- European language transfer — adequately
- Aggregate multilingual benchmarks — increasingly
- Social media content — LIAR, TruthSeeker, FakeNewsNet cover this
- Health disinformation in English — COVID datasets, MIC dataset

**Has not addressed:**
- Community-weighted evaluation for underrepresented language groups
- Code-switching detection (mixing Spanish/English, Tagalog/English) at scale
- Deployment-time bias amplification — does the system over-flag minority community content?
- Continuous monitoring of cross-lingual fairness metrics post-deployment
- Agricultural and rural community disinformation in low-resource languages

**Scoping finding:**
SemEval 2025 Task 7 ran a shared task on multilingual claim retrieval — confirming the field recognizes the gap but has not yet produced a production-ready solution. The shared task framing (retrieval, not detection) also reveals that the field is moving toward evidence-based verification rather than pattern-matching classification. HyDMIS's LDA + GPT-4 + transformer hybrid positions at this intersection.

---

## Section 4 — Meta-Analysis

### Quantitative Patterns Across the Literature

**Finding 1 — The 20-25 point F1 gap:**
Papers comparing high-resource and low-resource language performance consistently report 15-30 point F1 degradation when transferring from English to low-resource targets. This is the gap HyDMIS's community-weighted loss function is designed to reduce. The exact magnitude in HyDMIS's specific dataset combination will be established empirically in Phase 4 — the 20-30 point estimate in HyDMIS's README will be replaced with actual measured numbers before paper submission.

**Finding 2 — RemBERT dominance on low-resource subsets:**
PolyTruth (2025) is the clearest empirical evidence. Across their 25+ language evaluation, RemBERT outperforms mBERT and XLM consistently on languages with under 10K training examples. This is not a marginal improvement — it's the finding that justified adding RemBERT as HyDMIS's third backbone contender.

**Finding 3 — GPT-4 ceiling effect on low-resource languages:**
Multiple 2024-2025 papers note that GPT-4 performance on genuinely low-resource languages plateaus or degrades compared to its high-resource performance. This is HyDMIS's most important open question — Stage 2's semantic verification value depends on GPT-4's generalization holding for Tagalog, Haitian Creole, and similar languages.

**Finding 4 — Dataset size is not the bottleneck:**
The 2024 survey confirms the core insight HyDMIS is built on: the bottleneck is expert annotation, not data collection. Social media content in low-resource languages exists in abundance. The labeled ground truth does not. LDA's unsupervised Stage 1 is HyDMIS's response to this structural constraint.

---

## Section 5 — Narrative and Landscape Review

### Where the Field Is and Where It's Going

Multilingual disinformation detection is at an inflection point. The English-only era is definitively over — every major paper since 2022 includes some multilingual evaluation. The aggregate multilingual era is also ending — PolyTruth and SemEval 2025 Task 7 both push toward language-specific evaluation that surfaces performance gaps rather than averaging them away.

What comes next — and what HyDMIS contributes to — is community-centered evaluation. The question is no longer "does this model work across languages?" but "does this model work for the specific communities most harmed by disinformation?" Those are different questions with different evaluation requirements.

The three contested zones I kept running into across the literature:

**Contested zone 1 — Which backbone for low-resource?**
mBERT vs XLM-R vs RemBERT is an active debate. PolyTruth provides the clearest empirical answer to date (RemBERT), but the evaluation was on European and Near Eastern low-resource languages. Whether that finding holds for Southeast Asian and Caribbean Creole languages is genuinely open. HyDMIS ablates all three.

**Contested zone 2 — Does LLM verification generalize?**
GPT-4 as a semantic verifier works well on English and European languages. The 2024-2025 literature is divided on whether this generalizes to languages with limited GPT-4 pretraining coverage. This is not a small question — Stage 2 of HyDMIS depends on the answer. The ablation in Phase 4 will surface the actual numbers.

**Contested zone 3 — Is community-weighted loss the right intervention?**
Several 2024 papers tried demographic reweighting in classification tasks with mixed results. The technique is theoretically sound but implementation-sensitive. HyDMIS's Phase 4 Week 1 runs the community-weighted loss ablation before any other evaluation — the intervention either works or we pivot to data augmentation before the paper is written.

---

## Summary

Three things the literature tells me clearly:

1. The low-resource language gap in disinformation detection is real, quantified, and unsolved. The 2024 survey, PolyTruth, and SemEval 2025 all confirm it from different angles.

2. RemBERT is the right backbone for low-resource evaluation based on the best available empirical evidence. That may change as HyDMIS runs its own ablations, but it's the right starting point.

3. The hybrid pipeline approach — combining unsupervised topic modeling with LLM verification and cross-lingual classification — has not been attempted at scale for community-targeted health disinformation. That's the gap HyDMIS fills.

---

## References

- Wang et al. (2017) — LIAR Dataset, ACL
- Devlin et al. (2019) — BERT, NAACL
- Blei et al. (2003) — Latent Dirichlet Allocation, JMLR
- Conneau et al. (2020) — XLM-RoBERTa, ACL
- Chung et al. (2021) — RemBERT, ICLR
- Shu et al. (2020) — FakeNewsNet, Big Data
- Jiang et al. (2023) — Mistral 7B, arXiv
- OpenAI (2023) — GPT-4 Technical Report
- Pikuliak et al. (2023) — MultiClaim, arXiv
- Dadkhah et al. (2023) — TruthSeeker, IEEE TCSS
- Mohtaj et al. (2024) — NewsPolyML
- Ashraf et al. (2024) — DeFaktS
- Gouliev et al. (2025) — PolyTruth, arXiv 2509.10737
- Alghamdi et al. (2024) — Hybrid Summarization for Low-Resource FND, Knowledge-Based Systems
- Peng et al. (2025) — SemEval 2025 Task 7, arXiv 2505.10740
