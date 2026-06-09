# HyDMIS — Research Design Rationale
## Multilingual Disinformation Detection — Design Decisions and Framework

**Period:** March 2026 — May 2026
**Researcher:** Nithin Narla
**Status:** Complete — design rationale documented before Phase 4 implementation

**Verified dataset pipeline at time of writing:**
- LIAR2: 22,962 records (Wang et al. 2017, extended version)
- TruthSeeker: 134,198 records (Dadkhah et al. 2023)
- FakeNewsNet: 23,196 records (Shu et al. 2020)
- Covid-vaccine-misinfo-MIC: 5,952 records (Kim et al. 2023 — venue under verification before Phase 5)
- NewsPolyML: 32,129 records (Mohtaj et al. 2024)
- DeFaktS: 105,855 records (Ashraf et al. 2024)
- MultiClaim: 234,000+ pending Zenodo access (Pikuliak et al. 2023)
- ClimateMiSt: 146,670 pending faculty collaboration (Choi, Shang, Wang 2025)

**Confirmed verified total: 324,292 records**
(22,962 + 134,198 + 23,196 + 5,952 + 32,129 + 105,855 = 324,292)

---

## Research Question

Does a hybrid three-stage pipeline combining LDA unsupervised topic clustering, LLM semantic verification, and community-weighted cross-lingual classification outperform single-model baselines on low-resource language subsets representing targeted minority communities — and does the community-weighted loss function specifically reduce the performance gap on low-resource community languages compared to standard cross-entropy?

This question drives every design decision documented below.

---

## 1. The Core Design Problem

The literature analysis surfaced one finding that drove every design decision in HyDMIS: the performance gap in multilingual disinformation detection is not evenly distributed. PolyTruth (Gouliev et al., 2025, arXiv 2509.10737) quantified it — 15-30 point F1 degradation when transferring from high-resource to low-resource languages. Aggregate multilingual benchmarks hide this gap by averaging across languages. The communities most targeted by health and agricultural disinformation — Spanish-speaking, Tagalog-speaking, Haitian Creole-speaking — are exactly the communities where performance degrades most.

This framing changed what the research question needed to be. Not "can a multilingual model detect disinformation across languages?" but "does the system work for the specific communities being failed?" Those are different questions. The second one requires different datasets, a different evaluation framework, and a different loss function.

The agricultural connection reinforces this framing. The same communities underserved by agricultural AI systems — small farmers, farm laborers, rural Spanish-speaking, Tagalog-speaking, and Haitian Creole-speaking communities — are targeted by agricultural disinformation about GMOs, pesticides, and farm subsidies. This overlap connects HyDMIS directly to FAPE's finding that small farmers and farm laborers are systematically underserved by production ML fairness systems. No labeled agricultural disinformation dataset exists in these languages. HyDMIS addresses the broader community-targeted detection problem and documents agricultural disinformation as the critical unaddressed gap.

Three design constraints followed directly from this framing:

**Constraint 1 — Must work without labeled data in target community languages.**
Annotated disinformation datasets in Tagalog and Haitian Creole do not exist at scale. Any pipeline that requires labeled data in the target language for Stage 1 cannot be deployed for these communities. This constraint ruled out supervised topic classification for Stage 1 and pointed directly to LDA.

**Constraint 2 — Must evaluate by community, not aggregate.**
A system that achieves 85% aggregate F1 while failing at 60% on Haitian Creole content is not an improvement for the communities being served. The evaluation framework must stratify by language and community subset — not report average performance across all languages.

**Constraint 3 — Must quantify and correct for community-level detection bias.**
If the system over-flags content from historically scrutinized minority communities — higher false positive rate for Spanish-language content than English-language content on equivalent claims — that is a harm, not just a performance gap. The loss function must account for this explicitly.

---

## 2. Three-Stage Pipeline — Design Rationale

### Stage 1 — LDA Topic Clustering

**Decision: Latent Dirichlet Allocation for unsupervised topic identification**

I considered three approaches for Stage 1: LDA, neural topic models, and LLM-based topic extraction. Neural topic models require training data that does not exist in target community languages — ruling them out under Constraint 1. LLM-based topic extraction via GPT-4 API calls across 324,292+ records is computationally prohibitive for a filtering stage — the cost is unacceptable for what is essentially a preprocessing step. LDA (Blei et al., 2003) works without labeled data, is language-agnostic at the token level with appropriate preprocessing, produces interpretable clusters, and has 20+ years of deployment evidence. LDA is the only viable choice.

**Known limitation:** LDA struggles with short social media texts — this is documented throughout the topic modeling literature. HyDMIS addresses this through minimum token threshold pre-filtering before LDA application. The exact threshold value will be determined in Phase 4 preprocessing experiments. Content below the threshold bypasses Stage 1 entirely and enters Stage 2 directly with an unsorted flag — meaning Stage 2 processes it without a topic cluster assignment. Stage 2 handles unsorted content through domain-agnostic semantic verification rather than topic-conditioned verification. The unsorted flag is carried through to Stage 3 as a feature.

**What Stage 1 delivers:** Rough topical clusters — health, climate, agricultural, political — that allow Stage 2 semantic verification to operate on thematically coherent batches rather than the full undifferentiated corpus. Stage 1 is a filter, not a classifier. Noisy Stage 1 clusters propagate errors into Stage 2 and Stage 3 — cluster coherence scores are monitored before any content passes downstream.

---

### Stage 2 — LLM Semantic Verification

**Decision: GPT-4 for high-confidence labeling pass, Mistral 7B for computational scale**

The nearest prior hybrid pipeline attempt is Alghamdi et al. (2024) — hybrid summarization for low-resource fake news detection. Their approach combined extractive summarization with classification but did not include LLM semantic verification as a distinct pipeline stage and was scoped to summarization quality rather than community-stratified detection. HyDMIS extends beyond this by treating semantic verification as a core pipeline stage with its own ablation.

ClimateMiSt (Choi, Shang, Wang 2025) is the most direct evidence for the LLM verification design choice. Their finding that GPT-4 outperforms baseline models on both misinformation detection and stance classification for climate content validated using LLMs for semantic verification rather than keyword matching or surface-level classification.

GPT-4 handles the high-confidence labeling pass — claims where semantic complexity requires genuine natural language understanding. Mistral 7B (Jiang et al., 2023) handles full-corpus verification at scale. Mistral 7B outperforms Llama 2 13B on most benchmarks at half the parameter count (Jiang et al., 2023 — specific benchmark table to be cited before Phase 5) — this matters for a pipeline that needs to run on 324,292+ verified samples without prohibitive infrastructure cost.

These two models do not run in parallel. The pipeline runs GPT-4 first on a stratified sample to establish high-confidence labels, then uses those labels as fine-tuning signal for Mistral 7B before running Mistral on the full corpus. Disagreements between GPT-4 labels and Mistral predictions on the stratified sample are flagged for manual review before full-corpus Mistral verification begins. This GPT-4 to Mistral distillation mechanism is a proposed design — implementation details to be finalized in Phase 4.

**The open design question:** Does GPT-4 semantic verification generalize to Tagalog and Haitian Creole? ClimateMiSt validates for English climate content only. Multiple 2024-2025 papers document performance degradation on genuinely low-resource languages. This is the most fragile assumption in HyDMIS Stage 2. Claims in languages where GPT-4 verification confidence falls below a threshold — to be calibrated in Phase 4 ablation — are passed to Stage 3 without a Stage 2 label rather than with a potentially unreliable one. This is a proposed design decision, not prior art — Phase 4 ablation validates whether confidence-based routing improves community-level accuracy.

---

### Stage 3 — Cross-Lingual Classification with Community-Weighted Loss

**Decision: mBERT + XLM-R + RemBERT + Mistral backbone comparison with community-weighted cross-entropy**

**Backbone selection rationale:**

The backbone decision was resolved by PolyTruth (Gouliev et al., 2025, arXiv 2509.10737). Their five-model comparison — including mBERT, XLM-R, RemBERT, and two additional baselines — across 25+ languages with per-language resource level stratification is the most rigorous empirical evidence available. Finding: RemBERT outperforms mBERT and XLM-R specifically on low-resource language subsets — not marginally but consistently across the evaluation.

The architectural reason is in Chung et al. (2021). RemBERT's decoupled input and output embeddings allow larger output representations without increasing input parameter count — this gives the model more capacity to represent low-resource language semantics without the parameter explosion that would make deployment impractical.

PolyTruth's evaluation focused on European and Near Eastern low-resource languages. Whether RemBERT's advantage holds for Southeast Asian (Tagalog) and Caribbean Creole (Haitian Creole) languages is genuinely untested. The ablation across all four backbones — mBERT (Devlin et al., 2019) as baseline, XLM-R (Conneau et al., 2020) as current standard, RemBERT as best-supported challenger, Mistral as LLM classification baseline — produces the evidence rather than assuming PolyTruth's finding transfers directly. Note: Mistral appears in both Stage 2 (as verifier) and Stage 3 (as classification baseline). These are distinct uses — Stage 2 uses Mistral for generative semantic verification, Stage 3 uses Mistral as a sequence classification baseline to measure LLM classification performance against encoder-based models.

**Community-weighted loss function rationale:**

Standard cross-entropy loss treats all misclassifications equally. In a community-stratified training distribution, equal weighting implicitly prioritizes majority-language performance because majority-language examples dominate the gradient signal. The community-weighted loss assigns higher weight to misclassifications on low-resource community language examples during training — formally, each example is weighted by the inverse frequency of its community language group in the training distribution, so rare community languages receive proportionally higher gradient signal.

The empirical evidence for community-weighted loss in adjacent tasks is mixed in the 2024 fairness-aware classification literature. Theoretically sound. Not universally effective. Phase 4 begins with this ablation — community-weighted loss vs standard cross-entropy on low-resource subset F1 — before any other evaluation. If community-weighted loss does not improve low-resource subset performance, the paper reports this honestly and the loss function design is revised before paper writing begins.

Phase 4 begins June 2026 per project timeline. The community-weighted loss ablation runs in the first experimental block before any cross-backbone or cross-domain evaluation.

---

## 3. Dataset Selection Rationale

Six datasets verified across three selection criteria. 15+ languages represented across health, political, climate, and multilingual news domains. Coverage spans 6 distinct disinformation domains.

**Criterion 1 — Coverage of target community languages and domains:**
The pipeline needs multilingual content across health, climate, agricultural, and political disinformation domains. No single dataset covers all four. The six verified datasets collectively cover health, political, and multilingual news. Climate domain added when ClimateMiSt access is granted.

**Criterion 2 — Freely downloadable without data access agreements:**
MultiClaim and ClimateMiSt are pending access. The six verified datasets were selected specifically because they are immediately downloadable — pipeline development cannot be gated on pending approvals.

**Criterion 3 — Size sufficient for cross-lingual evaluation:**
Fairness metrics become unreliable below roughly 500 examples per subgroup. Datasets with insufficient community-level coverage for stratified evaluation were deprioritized.

**Dataset-by-dataset rationale:**

**LIAR2 (22,962):** English political baseline. LIAR2 is the extended version of the original LIAR benchmark (Wang et al., 2017) — original paper cited for benchmark lineage, LIAR2 extension citation to be confirmed before Phase 5. Required for comparability with existing literature. Limitation: English-only, formal political discourse. Used for baseline benchmarking only.

**TruthSeeker (134,198):** Largest verified social media fake news dataset. Multiple fact-checking organization ground truth reduces label bias. English-only but at scale sufficient for robust English baseline. Primary English benchmark dataset.

**FakeNewsNet (23,196):** Social context features available but HyDMIS uses content features only — social graph information is unavailable for low-resource community content and any evaluation that depends on it cannot generalize to target communities.

**Covid-vaccine-misinfo-MIC (5,952):** The only health domain multilingual dataset in the verified pipeline. Small but essential — covers the specific health misinformation domain most relevant to target communities. Kim et al. (2023) venue under verification before Phase 5 paper writing.

**NewsPolyML (32,129):** European multilingual news claims. Adds non-English European language coverage that strengthens cross-lingual transfer evaluation even though European languages are not the primary target communities.

**DeFaktS (105,855):** German-heavy multilingual Twitter dataset. Largest verified dataset with social media content format — closest to real-world deployment context. German dominance is a limitation but the size provides substantial cross-lingual training signal.

**MMCFND (Bansal et al., 2024) — considered, not included:** Seven Indic language multimodal fake news dataset. Relevant for low-resource evaluation but multimodal features (image + text) are outside HyDMIS's text-only pipeline scope. Text-only content features could be extracted in future work.

**MultiClaim (234,000+ pending):** Most directly relevant multilingual dataset for HyDMIS. Will substantially expand language coverage when Zenodo access is granted. Language distribution is European-heavy but the scale is valuable.

**ClimateMiSt (146,670 pending):** First dataset with both veracity and stance annotations for climate change content. Validates Stage 2 design with English climate content. Adds climate domain coverage to a pipeline currently weighted toward health and political disinformation.

When MultiClaim and ClimateMiSt access is granted, Section 3 dataset rationale will be updated with verified counts and this document will be revised to reflect any design changes driven by expanded language coverage.

---

## 4. Evaluation Framework Design

**Decision: Community-stratified evaluation as primary metric, aggregate multilingual F1 as secondary**

Every paper in the literature reports aggregate multilingual F1. PolyTruth (2025) showed why this is insufficient — aggregate numbers hide community-level failures. HyDMIS inverts this: community-stratified performance by language and resource level is the primary evaluation metric.

Resource level classification uses the following thresholds, consistent with PolyTruth's general stratification approach — exact alignment to be verified before Phase 5. High resource: languages with 100MB+ pretraining data and 10K+ labeled examples; medium resource: 10-100MB pretraining data or 1K-10K labeled examples; low resource: under 10MB pretraining data or under 1K labeled examples. Tagalog, Haitian Creole, and agricultural community languages fall in the low-resource category.

SemEval 2025 Task 7 (Peng et al., 2025) ran a shared task on multilingual claim retrieval — confirming the field recognizes the evaluation gap but has not produced community-stratified evaluation standards. HyDMIS goes beyond the shared task framing by focusing on detection rather than retrieval and on community-specific metrics rather than aggregate performance.

**Evaluation dimensions:**
- Per-language F1 stratified by resource level (high/medium/low per definitions above)
- Community-specific false positive rate — over-flagging bias for minority community content
- Cross-domain generalization — health vs climate vs political (agricultural documented as future work — no labeled data exists in community languages)
- Stage-level ablation — Stage 1 only, Stage 1+2, full pipeline
- Backbone comparison — mBERT vs XLM-R vs RemBERT vs Mistral

**Why false positive rate matters as much as F1:**
A system that correctly identifies 80% of Spanish-language health disinformation while also flagging 30% of legitimate Spanish-language health content is not an improvement for Spanish-speaking communities — it is a different kind of failure. F1 captures one dimension. Community-stratified false positive rate captures the dimension that matters most for deployment safety.

**Subgroup sample size audit:**
Fairness metrics become unreliable below roughly 500 examples per subgroup. Phase 4 begins with a subgroup size audit across all community language subsets before any community-stratified metrics are reported. If subgroup sample sizes for Tagalog, Haitian Creole, or other low-resource community languages fall below threshold, community-stratified evaluation for those groups is flagged as statistically limited rather than reported as reliable.

---

## 5. Scope Limitations Documented Before Phase 4

**Limitation 1 — Agricultural disinformation in community languages is not covered:**
No labeled agricultural disinformation dataset exists in Spanish, Tagalog, or Haitian Creole. HyDMIS documents this gap and positions it as future work. The pipeline design accommodates agricultural content through LDA topic clustering but cannot be evaluated on it without labeled ground truth.

**Limitation 2 — Code-switching is not addressed:**
Tagalog-English and Spanish-English code-switching in social media content is common in target communities. None of the six verified datasets contain significant code-switched content. HyDMIS acknowledges this as a deployment gap.

**Limitation 3 — Harm reduction is approximated, not measured:**
Better F1 on a benchmark does not automatically mean less disinformation exposure for real communities. HyDMIS approximates harm reduction through platform-scale simulation — modeling what system-level intervention would look like if deployed on a platform serving target communities, using verified dataset content as a proxy for production traffic. This is the honest limitation of academic research without live deployment API access.

**Limitation 4 — Subgroup sample sizes for low-resource community languages may be insufficient:**
Fairness metrics become unreliable below roughly 500 examples per subgroup. When 324,292 verified records are broken down by language and community subset, Tagalog, Haitian Creole, and agricultural community language examples may fall below this threshold. Phase 4 subgroup audit determines which community-stratified metrics can be reported reliably.

**Limitation 5 — Cross-domain transfer across health, climate, and political disinformation is unverified:**
LIAR2 is political. ClimateMiSt is climate. Covid-vaccine-misinfo-MIC is health. These domains use different vocabulary, different claim structures, and different disinformation tactics. Cross-domain generalization is tested in Phase 4. If transfer does not hold, paper scope narrows to domains where it does.

**Limitation 6 — MultiClaim and ClimateMiSt access is pending:**
Phase 4 experiments begin on 324,292 verified records. When pending datasets arrive, experiments will be extended and results updated. If access does not come through before paper submission, the paper explicitly scopes to the six verified datasets with MultiClaim and ClimateMiSt noted as future extensions.

---

## Summary

Three design choices drive HyDMIS. First, LDA for Stage 1 — the only viable unsupervised approach for communities where labeled data does not exist. Second, community-weighted cross-entropy loss — directly addressing the performance gap during training rather than post-hoc. Third, community-stratified evaluation as the primary metric — measuring whether the system works for the specific communities being failed, not just whether it works on aggregate benchmarks. Every dataset selection, backbone choice, and evaluation dimension follows from these three choices and from the core finding that motivated the research: the performance gap in multilingual disinformation detection accumulates exactly where the harm accumulates.

---

## References

- Wang et al. (2017) — LIAR Dataset, ACL
- Blei et al. (2003) — Latent Dirichlet Allocation, JMLR
- Devlin et al. (2019) — BERT, NAACL
- Conneau et al. (2020) — XLM-RoBERTa, ACL
- Chung et al. (2021) — RemBERT, ICLR
- Shu et al. (2020) — FakeNewsNet, Big Data
- Jiang et al. (2023) — Mistral 7B, arXiv
- OpenAI (2023) — GPT-4 Technical Report
- Pikuliak et al. (2023) — MultiClaim, arXiv
- Dadkhah et al. (2023) — TruthSeeker, IEEE TCSS
- Kim et al. (2023) — Covid-vaccine-misinfo-MIC (venue under verification)
- Mohtaj et al. (2024) — NewsPolyML (venue under verification before Phase 5)
- Ashraf et al. (2024) — DeFaktS (venue under verification before Phase 5)
- Bansal et al. (2024) — MMCFND, 7 Indic languages (venue under verification before Phase 5)
- Alghamdi et al. (2024) — Hybrid Summarization for Low-Resource FND, Knowledge-Based Systems
- Choi, Shang, Wang (2025) — ClimateMiSt, ASONAM 2024
- Gouliev et al. (2025) — PolyTruth, arXiv 2509.10737
- Peng et al. (2025) — SemEval 2025 Task 7, arXiv 2505.10740
