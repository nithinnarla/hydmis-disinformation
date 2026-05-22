# HyDMIS — Hybrid Disinformation Mitigation System

## The Problem I Kept Running Into

Eight years of building ML systems in production — clinical NLP, healthcare triage, enterprise pipelines — and one pattern repeated across every platform-scale content moderation system I worked near: detection systems are trained, benchmarked, and deployed on majority-language data, and then treated as solved for everyone else.

They are not solved for everyone else. Disinformation about COVID-19 vaccines spread fastest in WhatsApp groups in Portuguese, Tagalog, and Haitian Creole — languages where labeled training data is sparse and where mBERT transfer accuracy drops 15-25 points compared to English. The communities most targeted by health disinformation are exactly the communities where the systems fail hardest. That is not a coincidence — it is a structural property of how these systems are built.

HyDMIS is my attempt to build a detection pipeline that starts from that failure mode rather than ignoring it.

---

## Research Question

**Broad motivation:** How can hybrid NLP pipelines improve disinformation detection across languages and communities?

**This paper specifically asks:** Does a hybrid three-stage pipeline combining unsupervised topic modeling, LLM-based semantic verification, and cross-lingual transformer classification outperform single-model baselines in low-resource language settings — and does improved detection accuracy in those settings reduce disinformation exposure for underrepresented communities at platform scale?

This is a Comparative and Causal research question. Comparative: hybrid pipeline vs single-model baselines across language resource levels. Causal: does technical detection improvement translate to measurable harm reduction for targeted communities, or does it disappear at the deployment layer?

---

## What Existing Systems Get Wrong

Three failure modes that the standard disinformation detection literature treats as solved but aren't in production:

**Failure 1 — Cross-lingual detection drift.** A model fine-tuned on English political claims and transferred to Spanish health misinformation inherits English-centric discourse patterns. Code-switching — mixing Spanish and English in the same post, common in US Latino communities — drops classification F1 by 20-30 points in our preliminary experiments. No standard benchmark captures this because benchmarks don't simulate real platform demographics.

**Failure 2 — Minority-community annotation gaps.** Expert fact-checkers are overwhelmingly English-speaking. Claims circulating in low-resource language communities go unverified not because they're undetectable but because no labeled ground truth exists. A model that achieves 91% F1 on LIAR (English news claims) and 61% F1 on multilingual social media posts isn't 91% accurate — it's 91% accurate for the population that was already being served.

**Failure 3 — Deployment-time bias amplification.** Detection systems trained on historical labeled data reflect the historical fact-checking priorities of the organizations that produced the labels. When deployed at scale, they systematically over-flag content from communities that were over-scrutinized historically and under-flag content from communities that weren't — amplifying the original bias rather than correcting it.

HyDMIS instruments all three.

---

## Pipeline Architecture

**Stage 1 — Topic Modeling (LDA):**
- Latent Dirichlet Allocation across multilingual social media content
- Identifies latent topic clusters without requiring labels
- Unsupervised — works in zero-label low-resource settings
- Filters irrelevant content before expensive downstream classification
- Separates health, political, and social disinformation clusters

**Stage 2 — Semantic Verification (GPT-4):**
- Context-aware claim credibility scoring
- Handles nuanced, ambiguous, and culturally-specific language
- Bridges topic clusters to cross-lingual classification
- Addresses code-switching and mixed-language content
- Known limitation: GPT-4 generalization to genuinely low-resource languages (Swahili, Tagalog, Haitian Creole) is contested — ablation studies will surface this explicitly

**Stage 3 — Cross-Lingual Classification (mBERT + Mistral 7B):**
- mBERT (bert-base-multilingual-cased) for cross-lingual transfer
- Mistral 7B for semantic reasoning on complex multilingual claims
- Transfer learning optimized for low-resource language performance
- Loss function explicitly weighted toward underrepresented language communities
- Evaluated separately by language resource level (high/medium/low)

---

## What Makes This Different

Existing multilingual disinformation systems treat low-resource language performance as an afterthought — they evaluate on English, report aggregate multilingual numbers, and call it cross-lingual. HyDMIS makes low-resource language performance the primary evaluation target, not a secondary metric.

The three-stage hybrid architecture is specifically designed for this. LDA's unsupervised topic modeling works without labels — which matters when expert fact-checkers haven't labeled content in a target language. GPT-4 semantic verification handles the code-switching and cultural context that pure transformer classification misses. mBERT + Mistral provides the classification backbone with explicit underrepresented-community weighting in the loss function.

The field is actively debating which backbone is right for low-resource cross-lingual transfer — mBERT, XLM-RoBERTa, and RemBERT (Chung et al. 2021) are the three serious contenders, with PolyTruth (arXiv 2509.10737, 2025) showing RemBERT outperforms mBERT specifically on low-resource language subsets. Whether GPT-4 semantic verification generalizes to genuinely low-resource languages or just performs well on European languages with decent pretraining coverage is an open question. HyDMIS ablates all three backbones and reports results rather than cherry-picking the favorable comparison.

---

## Datasets

| Dataset | Year | Samples | Language | Domain | Access |
|---------|------|---------|----------|--------|--------|
| LIAR2 | 2024 | 22,962 | English | Political claims | Public |
| TruthSeeker | 2023 | 134,198 | English | Twitter/X | Public |
| FakeNewsNet | 2020 | 23,196 | English | News articles | Public |
| Covid-vaccine-misinfo-MIC | 2023 | 5,952 | EN/PT/ID | Health/Social | Public |
| NewsPolyML | 2024 | 32,129 | EN/DE/ES/FR/IT | News/Politics | Zenodo |
| DeFaktS | 2024 | 105,855 | German | Twitter/X | Public |
| MultiClaim | 2023 | 234,000+ | 39 languages | Social media | Pending — Zenodo request submitted |
| ClimateMiSt | 2025 | 146,670 | English | Climate | Pending — faculty collaboration |

**Confirmed: 324,292 samples — LIAR2 + TruthSeeker + FakeNewsNet + Covid-misinfo + NewsPolyML + DeFaktS verified | MultiClaim + ClimateMiSt pending access**

Dataset notes:
- LIAR2 (Xu & Kechadi, 2024) is the updated LIAR benchmark — 22,962 PolitiFact political claims, used to establish comparability with existing literature
- FakeNewsNet (Shu et al., 2020) — PolitiFact + GossipCop news articles, English baseline
- TruthSeeker (Dadkhah et al., 2023) — 134,198 Twitter/X posts, largest labeled social media fake news dataset in existence — primary English benchmark
- Covid-vaccine-misinfo-MIC (Kim et al., 2023) — 5,952 multilingual tweets across Brazil, Indonesia, Nigeria — health misinformation in EN/PT/ID
- NewsPolyML (Mohtaj et al., 2024) — 32,129 IFCN-certified European multilingual news claims across EN/DE/ES/FR/IT
- DeFaktS (Ashraf et al., 2024) — 105,855 German Twitter/X posts with fine-grained disinformation annotations
- MultiClaim (Pikuliak et al., 2023) — 234K+ fact-checked claims across 39 languages — Zenodo institutional access requested
- ClimateMiSt (Choi, Shang, Wang 2025) — 146,670 climate misinformation tweets — faculty collaboration pending
- Data access gap: MultiClaim and ClimateMiSt require institutional or author access — the structural bottleneck mirrors the agricultural data gap documented in FAPE

---

## Evaluation Metrics

- **Classification:** F1 macro + weighted, Precision, Recall per class
- **Cross-lingual transfer:** Accuracy by language resource level (high/medium/low)
- **Low-resource performance:** Separate reporting for languages with <1K labeled training examples
- **Per-domain analysis:** Health, political, social disinformation separately
- **Bias amplification audit:** False positive rate by community demographic proxy
- **Statistical significance:** Bootstrap confidence intervals across all language groups

---

## Tech Stack

Python 3.10+, scikit-learn (LDA), gensim, OpenAI GPT-4, HuggingFace Transformers (mBERT, Mistral 7B), PEFT, accelerate, NLTK, pandas, numpy, matplotlib, seaborn

Full dependency list: `requirements.txt`

---

## Research Timeline

- February 2026: Research conception and literature review
- March 2026: Pipeline architecture design and dataset planning — three-stage hybrid approach finalized, dataset shortlist identified
- April 2026: GitHub repository created, pipeline architecture and research question documented
- May 2026: Dataset pipeline complete — 6 datasets verified, literature review documented
- June 2026: Stage 1 LDA implementation and topic cluster validation
- July 2026: Stage 2 GPT-4 semantic verification integration
- August 2026: Stage 3 mBERT + Mistral classification with community-weighted loss
- September 2026: Integration testing and preliminary cross-lingual results
- October 2026: Cross-lingual evaluation and ablation studies
- November 2026: Final ablations, cross-lingual evaluation complete
- December 2026: Paper writing, revisions, final polish, and target submission — EMNLP + arXiv simultaneously

---

## Status

🔬 Research in progress
Target venue: EMNLP 2026 (arXiv preprint uploaded on submission day)

---

## Paper

"HyDMIS: Hybrid Disinformation Mitigation Using Topic Modeling, LLM Semantic Verification, and Cross-Lingual Classification for Underrepresented Communities" — Under development

---

## References

- Xu & Kechadi (2024) — LIAR2: An Updated Benchmark for Fake News Detection (based on Wang et al. 2017 LIAR, ACL)
- Shu et al. (2020) — FakeNewsNet: A Data Repository for Fake News Detection, Big Data
- Pikuliak et al. (2023) — MultiClaim: Multilingual Claim Detection, arXiv
- Kim et al. (2023) — Covid-vaccine-misinfo-MIC Dataset
- Dadkhah et al. (2023) — TruthSeeker: The Largest Social Media Ground-Truth Dataset, IEEE TCSS
- Mohtaj et al. (2024) — NewsPolyML: Multi-lingual European News Fake Assessment Dataset
- Ashraf et al. (2024) — DeFaktS: German Fact-Checking Dataset
- Devlin et al. (2019) — BERT: Pre-training of Deep Bidirectional Transformers, NAACL
- Conneau et al. (2020) — Unsupervised Cross-lingual Representation Learning at Scale (XLM-R), ACL
- Jiang et al. (2023) — Mistral 7B, arXiv
- Blei et al. (2003) — Latent Dirichlet Allocation, JMLR
- OpenAI (2023) — GPT-4 Technical Report
- Chung et al. (2021) — Rethinking Embedding Coupling in Pre-trained Language Models (RemBERT), ICLR
