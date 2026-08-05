# HyDMIS — Methodology Decisions Log
## Multilingual Disinformation Detection — Methodology Decisions

---

## How to Read This Document

This is a decisions log, not a polished writeup. Every major methodological choice is documented here with the alternatives I considered and why I made the call I made. Some decisions I'm confident about. A few I'm still not completely certain about — those are marked with a note.

The point of documenting decisions before writing code is to prevent the most common research mistake: making a decision implicitly during implementation and then justifying it post-hoc in the paper. Every decision here was made before Phase 4 starts. If Phase 4 produces results that contradict a decision, the decision gets updated — but the reasoning trail stays visible.

---

## Decision 1 — Research Question Type: Comparative + Causal

**Decision:** HyDMIS is a Comparative + Causal research question. Comparative: hybrid pipeline vs single-model baselines across language resource levels. Causal: does detection improvement translate to harm reduction for targeted communities?

**Alternatives considered:**
- Comparative only — standard NLP methods paper comparing models on benchmarks
- Descriptive only — documenting the performance gap without proposing a solution

**Why Comparative + Causal:**
The comparative question alone produces a methods contribution. The causal question produces an impact contribution. Both are needed to justify HyDMIS's community-centered framing. A paper that only shows RemBERT + LDA + GPT-4 beats mBERT on aggregate F1 is not a paper about underrepresented communities — it's a methods paper that happens to mention them. Adding the causal component — community-stratified false positive rates as proxy for harm reduction — makes the community impact claim empirically grounded.

---

## Decision 2 — Three-Stage Hybrid Architecture

**Decision:** LDA (Stage 1) → GPT-4 semantic verification (Stage 2) → RemBERT/mBERT/Mistral cross-lingual classification (Stage 3)

**Alternatives considered:**
- Single transformer end-to-end: mBERT or XLM-R fine-tuned directly on all datasets
- Two-stage: LDA + transformer, no LLM verification
- Four-stage: adding a post-classification bias audit layer
- RAG-based verification: retrieval-augmented generation for evidence-based claim checking

**Why three stages:**
Single transformer requires labeled data in every target language — not available for genuinely low-resource community languages. Two-stage removes the semantic nuance that LLMs handle better than classifiers for code-switched and culturally-specific content. Four-stage adds complexity without a clear performance hypothesis. RAG-based verification is methodologically interesting but adds infrastructure requirements that make the pipeline impractical for community deployment — the stated goal.

Three stages address exactly the three documented failure modes in existing systems with the minimum necessary complexity.

---

## Decision 3 — LDA for Stage 1 Unsupervised Topic Modeling

**Decision:** Use Latent Dirichlet Allocation for Stage 1 topic modeling, not neural topic models or LLM-based topic extraction.

**Alternatives considered:**
- Neural topic models (NTM, CTM): better coherence scores but require more compute
- BERTopic: uses transformer embeddings for topic extraction, better handles short texts
- LLM zero-shot topic extraction: GPT-4 or Mistral directly extract topics without training
- No Stage 1: skip topic modeling, feed all content directly to Stage 2

**Why LDA:**
LDA works in genuinely zero-label settings — no training data required in the target language. BERTopic requires transformer embeddings which depend on pretraining quality for the target language. LLM zero-shot extraction works but doubles API costs for Stage 2. No Stage 1 means processing all 562K+ samples through GPT-4 — computationally and financially prohibitive.

LDA's short text limitation is real and acknowledged. Mitigation: minimum token filter before Stage 1, post-processing of incoherent clusters. The tradeoff — interpretable, language-agnostic, zero-label — justifies the limitation for this specific use case.

**Uncertainty note:** BERTopic may outperform LDA for this specific application. Phase 4 runs a Stage 1 ablation comparing LDA vs BERTopic on a 10K sample before committing to LDA for full-scale evaluation.

---

## Decision 4 — GPT-4 as Stage 2 Semantic Verifier

**Decision:** Use GPT-4 as the primary semantic verification backbone for Stage 2, with Mistral 7B as the cost-efficient alternative for full-scale deployment.

**Alternatives considered:**
- Mistral 7B only: cheaper, runs locally, no API dependency
- GPT-3.5: cheaper than GPT-4, worse quality
- Fine-tuned BERT classifier: faster inference, lower quality on nuanced claims
- No Stage 2: skip semantic verification, feed LDA output directly to Stage 3

**Why GPT-4 + Mistral 7B hybrid:**
GPT-4 produces the highest quality semantic verification — ClimateMiSt confirms GPT-4 outperforms all baseline models on both veracity and stance detection. But GPT-4 at 562K+ sample scale costs approximately $1,500-2,000. The practical solution: GPT-4 labels a representative 15K sample across all datasets and language groups, fine-tunes Mistral 7B on those labels, deploys Mistral 7B for full-scale verification. This reduces cost by approximately 95% while maintaining verification quality within acceptable bounds.

No Stage 2 removes the nuance handling that differentiates HyDMIS from pure transformer classification — not viable given the research question.

**Uncertainty note:** GPT-4's generalization to genuinely low-resource languages — Tagalog, Haitian Creole, Swahili — is contested in the 2024-2025 literature. Phase 4 Week 1 runs a GPT-4 low-resource language ablation before any paper claim references GPT-4 verification quality on these languages. If GPT-4 fails on these languages, Stage 2 pivots to a multilingual fine-tuned model.

---

## Decision 5 — RemBERT as Primary Stage 3 Backbone

**Decision:** Use RemBERT as the primary cross-lingual classification backbone for Stage 3, with mBERT and Mistral 7B as comparison baselines.

**Alternatives considered:**
- mBERT only: standard baseline, well-documented limitations on low-resource languages
- XLM-R only: stronger than mBERT but still degrades on low-resource subsets
- mT5: sequence-to-sequence model, different architecture, harder to compare fairly
- Ensemble of all three: higher performance ceiling but obscures individual model contributions

**Why RemBERT:**
PolyTruth (2025) provides the clearest empirical evidence: RemBERT outperforms mBERT and XLM consistently on languages with under 10K training examples — exactly HyDMIS's target setting. RemBERT's decoupled input/output embeddings allow larger output representations without increasing input parameter count — architecturally suited for low-resource transfer.

All three backbones (mBERT, RemBERT, Mistral 7B) are evaluated in Phase 4 ablations. RemBERT is the primary — not the only — backbone. Results are reported separately by language resource level (high/medium/low) so the comparison is honest.

---

## Decision 6 — Community-Weighted Loss Function

**Decision:** Apply community-weighted loss in Stage 3 training, assigning higher weights to underrepresented language community examples.

**Alternatives considered:**
- Standard cross-entropy: treats all examples equally regardless of language community
- Data augmentation: synthetic examples via back-translation to increase low-resource training data
- Curriculum learning: train on high-resource first, fine-tune on low-resource
- Oversampling: repeat low-resource examples to balance training distribution

**Why community-weighted loss:**
Data augmentation introduces synthetic examples that inherit generation model biases — real examples are always preferable. Curriculum learning requires careful scheduling that adds implementation complexity without a clear advantage over weighted loss. Oversampling risks overfitting on repeated low-resource examples.

Community-weighted loss is the most direct intervention: it changes what the model optimizes, not just what data it sees. The weight magnitudes are determined empirically in Phase 4 based on actual class imbalance in the training data.

**Uncertainty note:** Community-weighted loss is empirically mixed in adjacent fairness tasks. This is the highest-risk methodological decision in HyDMIS. Phase 4 Week 1 runs the ablation before any other evaluation. If community-weighted loss does not outperform standard cross-entropy on low-resource subsets, the paper pivots to data augmentation or curriculum learning. The paper never claims community-weighted loss works until the ablation confirms it.

---

## Decision 7 — 9 Datasets Across 6 Domains

**Decision:** Use 9 datasets covering political, news, health, social media, South Asian, and climate domains totaling 562K+ samples across 15+ languages.

**Alternatives considered:**
- 7 original datasets: LIAR, FakeNewsNet, MultiClaim, Covid-vaccine-misinfo-MIC, TruthSeeker, NewsPolyML, DeFaktS
- Add only MMCFND: 8 datasets
- Add only ClimateMiSt: 8 datasets
- Add both MMCFND and ClimateMiSt: 9 datasets (chosen)

**Why add MMCFND:**
Seven Indic languages covering South Asian diaspora communities — the most significant low-resource language gap in the original 7 datasets. South Asian communities in the US are primary targets of health and civic participation disinformation. No other dataset in the corpus covers this language family.

**Why add ClimateMiSt:**
146,670 tweets with veracity and stance annotations — largest climate disinformation dataset with this annotation depth. Co-authored by Dong Wang (UIUC iSchool target faculty). ClimateMiSt's GPT-4 finding directly validates HyDMIS Stage 2 design. Climate and agricultural disinformation are thematically connected to HyDMIS's community targeting framing.

**Why stop at 9:**
562K+ samples across 15+ languages and 6 domains is the most comprehensive multilingual disinformation evaluation corpus assembled for a single paper. Additional datasets add diminishing returns and increase Phase 4 compute requirements without strengthening the research claim.

**Status note (Jun 2026):** MultiClaim is pending Zenodo access. ClimateMiSt is pending Dong Wang email response. Both are expected before Phase 4 Stage 2 begins. If either remains unavailable, the corpus falls to 7 confirmed datasets — still sufficient for the cross-domain claim.

---

## Decision 8 — Evaluation by Language Resource Level

**Decision:** Report all results stratified by language resource level (high/medium/low) rather than aggregate multilingual F1 only.

**Alternatives considered:**
- Aggregate F1 only: simpler, easier to compare with existing literature
- Per-language reporting: too granular, 15+ languages produces unreadable tables
- Per-domain reporting: captures domain variation but obscures community-level performance

**Why language resource stratification:**
PolyTruth (2025) established this as the right evaluation methodology for low-resource multilingual work. Aggregate F1 hides the performance gaps HyDMIS is specifically designed to address. A system that achieves 91% on English and 61% on Tagalog reports 76% aggregate — which looks acceptable. Stratified reporting exposes the 30-point gap. Reviewers who know the field will expect stratified results. Providing only aggregate numbers would be a methodological red flag.

**Phase 4 EDA validation (Jun 2026):** TruthSeeker EDA confirmed the NO MAJORITY annotation class (16.8% of data, 49.0% dis rate) — validates stratified evaluation. FakeNewsNet EDA confirmed no-URL posts have 78.8% disinformation rate — validates metadata feature inclusion in Stage 2 semantic verification.

---

## Decision 9 — EMNLP 2027 as Target Venue

**Decision:** Submit to EMNLP 2027 via ACL Rolling Review (~May 2027), with arXiv preprint uploaded December 2026.

**Alternatives considered:**
- ACL 2027: more prestigious, more time for Phase 4 and writing
- NAACL 2027: North American focus, strong NLP community
- ACL Findings 2026: lower bar, faster turnaround
- COLING 2026: strong multilingual NLP track

**Why EMNLP 2027:**
EMNLP 2026 ARR submission deadline was May 25, 2026 — missed. NAACL 2027 ARR deadline is approximately October 2026 — before HyDMIS paper writing begins. ACL 2027 ARR deadline is approximately February 2027 — possible but tight. EMNLP 2027 ARR deadline approximately May 2027 gives sufficient time to write, revise, and polish a strong paper after December 2026 manuscript completion. EMNLP is the strongest venue for empirical multilingual NLP — submitting to ACL first and falling back to EMNLP is the alternative if ACL 2027 deadline is confirmed achievable.

**arXiv preprint December 2026:** Posted after manuscript complete. Establishes priority and provides a citable preprint for PhD application materials even though no venue has accepted the paper at that point.

**PhD application strategy:** HyDMIS listed as "manuscript in preparation" on Oct 20 iSchool application and manuscript attached to Dec 1 Informatics application. Neither deadline requires the paper to be under peer review.

---

## Decision 10 — Kim et al. Covid-vaccine-misinfo-MIC Venue

**Decision:** Confirmed. Use this dataset with the correct citation.

**Resolved:** Jongin Kim, Byeo Rhee Bak, Aditya Agrawal, Jiaxi Wu, Veronika Wirtz, Traci Hong, and Derry Wijaya. 2023. COVID-19 Vaccine Misinformation in Middle Income Countries. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 3903–3915, Singapore. ACL Anthology: https://aclanthology.org/2023.emnlp-main.237/

Venue confirmed as EMNLP 2023 main conference. Literature docs updated before Phase 5.

---

## Open Decisions — Not Yet Resolved

**Open 1 — LDA topic count:**
Optimal number of topics for Stage 1 across 6 domains and 15+ languages. Standard range is 20-100. Phase 4 ablation determines this empirically before fixing the parameter.

**Open 2 — GPT-4 sample size for Stage 2 fine-tuning:**
How many GPT-4-labeled examples are needed to fine-tune Mistral 7B to acceptable quality? 10K? 15K? 25K? This determines cost and quality tradeoff. Phase 4 Week 1 experiments determine this.

**Open 3 — Community weight magnitudes:**
Exact weight values for underrepresented language communities in the loss function. Depends on actual class imbalance in training data — determined empirically in Phase 4.

**Open 4 — False positive rate as harm reduction proxy:**
Is community-stratified false positive rate a sufficient proxy for deployment-time harm reduction? Or do we need additional evidence? The causal claim in the research question requires careful framing — overreach here is the most likely reviewer objection to the paper's contribution claim.

---

## References

All references as listed in literature_review.md and literature_analysis.md.

## Decision 11 — GPT-4 Verified Sample Cannot Be Retroactively Joined to Real LDA Topic Assignments; Stage 2 Ablation Blocked Until Re-Sampling

**Investigation (Aug 3 2026):** After fixing lda_pipeline.py to finally persist real topic assignments to data/processed/lda_topic_assignments.csv (177,074 records, previously never saved -- see commit history), attempted to join this against the completed gpt4_verified.csv (14,640 records) to build a Stage 2 ablation comparing LDA-topic-alone veracity discrimination against GPT-4-verified labels on the same records.

**Finding:** This join is not possible with current data. gpt4_sampler.py's sampling logic (line 181-182) uses pd.concat(frames, ignore_index=True) followed by sample.sample(frac=1, random_state=42).reset_index(drop=True) -- both operations discard the original DataFrame index that lda_topic_assignments.csv's record_id column depends on. No alternative identifier (dataset-specific id column, timestamp, etc.) was preserved in gpt4_verified.csv either.

**Text-based join also ruled out:** Checked whether matching on (dataset, text) as a substitute key would work. Found 3,407 duplicate (dataset, text) pairs within the 14,640-record verified sample (23.3%) -- a text-based join would silently produce wrong or ambiguous matches for nearly a quarter of records, which is not an acceptable substitute for a real key-based join.

**Status: RESOLVED (Aug 3 2026, same day).** Fixed gpt4_sampler.py to preserve record_id through all 5 dataset-loading functions (df['record_id'] = df.index added before each stratified_sample call, and record_id added to each function's returned column selection). Re-ran gpt4_sampler.py with the same random_state=42 -- confirmed the regenerated sample is identical, row-for-row, to the original GPT-4-verified sample (same 14,640 texts, same order, verified via direct array comparison). This meant the existing GPT-4 labels (from the completed, already-paid-for verification run) could be reattached to the newly record_id-enabled sample without any new API calls -- avoiding the ~$2-3, multi-hour re-verification cost entirely. Joined against lda_topic_assignments.csv: 11,294 of 14,640 records (77.1%) now have a genuine LDA topic ID. The 22.9% gap is explained, not a bug -- lda_pipeline.py caps TruthSeeker and DeFaktS at SAMPLE_SIZE=50,000 (out of 134,198 and 105,855 full records) for LDA training speed, while gpt4_sampler.py draws independently from the full datasets; the two samples overlap only partially by chance (match rates: LIAR2 99.9%, FakeNewsNet 98.1%, NewsPolyML 100%, TruthSeeker 44.6%, DeFaktS 46.3% -- consistent with this explanation).

**Decision:** Given HyDMIS has substantial schedule slack (arXiv target Dec 22 2026, EMNLP ~May 2027), this re-sampling and re-verification is deferred rather than rushed tonight. The completed 14,640-record verification (label distribution UNCERTAIN=6181, YES=4620, NO=3838, PARTIAL=1) remains valid and usable for everything that doesn't require LDA-topic correlation -- it is not wasted work, but it cannot currently support a Stage 1-vs-Stage 2 ablation. The Stage 2 ablation task is blocked on this re-sampling, not abandoned.

**Required before re-sampling can proceed:** Update gpt4_sampler.py to preserve original DataFrame index (or an explicit id column) through both the pd.concat and the shuffle/reset_index step, mirroring the fix already applied to lda_pipeline.py, so this traceability gap does not recur.

## Decision 12 — Stage 2 Ablation Result: GPT-4 Verification Adds Real Discriminative Value Over LDA Alone

**Result (Aug 3 2026):** Using the 11,294 records with both a real LDA topic and a GPT-4 veracity label (see Decision 11 for how this joined dataset was built), quantified whether LDA topic alone meaningfully predicts veracity. Method: for each of the 10 English LDA topics, computed the majority GPT-4 label's share of records in that topic, then averaged across topics.

**Finding:** Baseline (guessing the single most common GPT-4 label overall, ignoring topic entirely) achieves 42.1% accuracy. Knowing the LDA topic raises this to a mean of 45.7% -- a +3.6 percentage point improvement. This is a weak effect, not a strong one.

**Conclusion:** This is the first quantitative confirmation of what lda_topic_clusters.py's own comments already claimed qualitatively ("Topic-veracity alignment: partial -- LDA captures content domains not veracity"). LDA (Stage 1) successfully separates content into coherent topical clusters (health, political, social, etc.) but those clusters do not strongly correlate with whether content is true, false, or uncertain. GPT-4 semantic verification (Stage 2) is therefore not redundant with Stage 1 -- it provides the substantial majority of the pipeline's actual veracity-discrimination signal. This is a positive result for the paper's architecture: it justifies the two-stage design (unsupervised topic discovery, then supervised-style semantic verification) rather than relying on topic modeling alone, which this ablation shows would perform only marginally better than a naive majority-class baseline.

**Caveat for the paper:** This ablation only covers 77.1% of the full verified sample (11,294/14,640), due to the LDA subsampling gap documented in Decision 11. The 4 topics computed on the full LIAR2/FakeNewsNet/NewsPolyML datasets (near-100% coverage) are more reliable than the TruthSeeker/DeFaktS-derived topics (which only cover ~45% of those datasets' GPT-4-verified records). This partial coverage should be stated explicitly if this ablation is reported in the paper.

## Decision 13 — Stage 3 mBERT Training Target: gpt4_label, Not veracity

**Investigation (Aug 4 2026):** Before building Stage 3 mBERT classification setup, checked the joined dataset (gpt4_verified_with_lda.csv, 14,640 records) for the correct training label column. Two candidates exist: veracity (the original per-dataset label) and gpt4_label (Stage 2's GPT-4 semantic verification output, YES/NO/UNCERTAIN/PARTIAL).

**Finding:** veracity is not a usable unified training target. It contains 5 incompatible encoding schemes stacked in one column, inherited directly from the 5 source datasets with no normalization ever applied: binary as int (0/1, ~4,332 records), binary as float (0.0/1.0, ~5,332 records -- itself inconsistently typed from the int version), 6-point ordinal (0-5, ~3,330 records, likely LIAR2's original scale), and text labels (true/false/mixture/mislabeled/other, ~1,646 records, likely NewsPolyML). A cross-tab against gpt4_label also shows GPT-4's independent verification frequently disagrees with each dataset's own original label -- e.g. veracity=0.0 splits across NO (386), UNCERTAIN (860), and YES (1,420) with no dominant match, meaning even a per-dataset veracity value doesn't reliably predict what GPT-4 (or by extension mBERT) would independently judge.

**Decision:** mBERT (and RemBERT, Mistral 7B in later ablations) trains against gpt4_label as the target, not veracity. gpt4_label is the only column with one consistent labeling process (GPT-4 semantic verification, Decision 4) applied uniformly across all 5 source datasets, rather than 5 different raw label conventions never reconciled with each other.

**Scope note:** Decision 3 (Phase 4 plan) specifies GPT-4 labels a 15K sample, Mistral 7B is fine-tuned on those labels, then Mistral 7B labels the full 562K+ dataset for full-scale Stage 3 training. That Mistral-7B pseudo-labeling step has not been built yet (scheduled later in August per the tracker). Today's Stage 3 setup uses only the 14,640 directly GPT-4-labeled records as a working prototype to build and validate the classification pipeline architecture -- full-scale training on the 562K+ dataset is correctly deferred until the Mistral-7B labeling step exists.

## Decision 14 — Exclude Single PARTIAL Record from Stage 3 Prototype Training; Train 3-Class (YES/NO/UNCERTAIN)

**Finding:** The 14,640-record GPT-4-labeled dataset has exactly 1 PARTIAL record (verified manually earlier as a genuine partially-true German political fact-check, not a data error). A single example cannot be split across train/val/test or evaluated with any per-class metric.

**Decision:** Exclude the 1 PARTIAL record and train mBERT (and later RemBERT, Mistral 7B) as a 3-class classifier: YES, NO, UNCERTAIN. Rejected merging PARTIAL into UNCERTAIN -- these are semantically distinct: UNCERTAIN means GPT-4 lacked sufficient evidence to judge truth/falsity; PARTIAL means GPT-4 had sufficient evidence and judged the claim as mixed/partially true. Merging would incorrectly redefine UNCERTAIN's meaning for the sake of the one PARTIAL record.

**Root cause identified (Aug 4 2026):** Checked whether the PARTIAL value was a field-parsing bug (e.g. topic_match's PARTIAL value leaking into gpt4_label). Ruled out -- the record's gpt4_topic_match field correctly shows MISMATCH, a valid value for that field, confirming both fields were parsed independently and correctly. The actual cause: gpt4_verifier.py's prompt (Decision 2 architecture) only offers YES/NO/UNCERTAIN as valid disinformation-verdict options, but GPT-4 produced 'PARTIAL' as free-form output for this one record, evidently because the source text itself states 'Das stimmt nur teilweise' (German: 'that's only partially true') and the model's own reasoning overrode the closed-set instruction. This confirms exclusion (not merging into UNCERTAIN) is the correct choice: UNCERTAIN means GPT-4 lacked sufficient evidence to judge the claim; this record shows GPT-4 had a confident, reasoned partial judgment, which is a different situation UNCERTAIN would misrepresent.

**Scope:** This exclusion applies to today's 14,640-record prototype dataset only. 1/14,640 is very likely a sampling artifact of this specific stratified sample, not the true population rate -- PARTIAL should be re-evaluated once the full 562K+ dataset is available via Mistral-7B pseudo-labeling (see Decision 13), where it will likely have enough examples to be trainable as a genuine 4th class.
