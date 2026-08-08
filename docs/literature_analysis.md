# HyDMIS, Literature Analysis
## Multilingual Disinformation Detection, All 9 Research Protocols

---

## Protocol 1, Intake: Paper Table + Clusters + Conflicts

### Paper Table

| Paper | Year | Venue | Core Claim | Cluster | Conflict |
|-------|------|-------|------------|---------|----------|
| Wang et al., LIAR | 2017 | ACL | English political claim benchmark | Monolingual baselines | English-only scope |
| Blei et al., LDA | 2003 | JMLR | Unsupervised topic modeling | Unsupervised methods | Short text limitation |
| Devlin et al., BERT | 2019 | NAACL | Bidirectional pretraining | Transformer backbone | English-dominant pretraining |
| Conneau et al., XLM-R | 2020 | ACL | Cross-lingual representation at scale | Cross-lingual transfer | Low-resource degradation |
| Shu et al., FakeNewsNet | 2020 | Big Data | Social context fake news dataset | Dataset | English-only |
| Chung et al., RemBERT | 2021 | ICLR | Decoupled embedding coupling | Cross-lingual transfer | Computational cost |
| Jiang et al., Mistral 7B | 2023 | arXiv | Efficient open-source LLM | LLM verification | Low-resource generalization |
| OpenAI, GPT-4 | 2023 | Tech Report | SOTA LLM semantic reasoning | LLM verification | Low-resource generalization contested |
| Pikuliak et al., MultiClaim | 2023 | arXiv | Multilingual claim dataset | Dataset | European-heavy distribution |
| Dadkhah et al., TruthSeeker | 2023 | IEEE TCSS | Largest social media fake news dataset | Dataset | English-only |
| Kim et al., Covid-vaccine-misinfo-MIC | 2023 |, | Health misinformation multilingual | Dataset | EMNLP 2023 |
| Mohtaj et al., NewsPolyML | 2024 |, | European multilingual news claims | Dataset | European-only |
| Ashraf et al., DeFaktS | 2024 |, | German/multilingual Twitter dataset | Dataset | German-heavy |
| Alghamdi et al. | 2024 | KBS | Hybrid summarization low-resource FND | Hybrid methods | Summarization scope only |
| Bansal et al., MMCFND | 2024 |, | 7 Indic language news misinformation | Dataset | Indic languages only |
| Choi, Shang, Wang, ClimateMiSt | 2025 | ASONAM | Climate misinformation + stance detection | Dataset | English-only, climate domain |
| Gouliev et al., PolyTruth | 2025 | arXiv | RemBERT best for low-resource | Cross-lingual backbone | European/Near Eastern focus |
| Peng et al., SemEval 2025 Task 7 | 2025 | arXiv | Multilingual claim retrieval shared task | Evaluation | Retrieval not detection |

### Clusters

Six conversations in this literature. They don't talk to each other nearly enough.

**The monolingual baseline conversation:** LIAR, FakeNewsNet, TruthSeeker. These papers built the field's benchmarks in English and stopped there. LIAR is 2017. TruthSeeker is 2023. Six years later and the dominant evaluation datasets are still English-only political and social media content. That's the structural problem HyDMIS is built around.

**The cross-lingual transfer conversation:** BERT, XLM-R, RemBERT, PolyTruth. Each paper pushed multilingual representation further. PolyTruth (2025) is the one that actually asked the right question, not just "does this work multilingually" but "where specifically does it fail." The answer is low-resource languages, which is exactly where HyDMIS operates.

**The LLM verification conversation:** GPT-4, Mistral 7B, Alghamdi et al. The open question is whether LLM semantic verification generalizes to genuinely low-resource languages. ClimateMiSt validates GPT-4 for English climate content. Nobody has tested Tagalog or Haitian Creole. That's HyDMIS Stage 2.

**The multilingual dataset conversation:** MultiClaim, Covid-vaccine-misinfo-MIC, NewsPolyML, DeFaktS, MMCFND, ClimateMiSt. The datasets are getting better, more languages, more domains, more annotation quality. But they're still European-heavy or domain-specific. The communities most targeted by health and agricultural disinformation are underrepresented in every one of these.

**The unsupervised methods conversation:** LDA and its descendants. Blei et al. (2003) is still the most practical choice when you have no labeled data, which is the situation HyDMIS faces in Stage 1 for languages where annotated disinformation datasets don't exist.

**The evaluation frameworks conversation:** SemEval 2025 Task 7. Retrieval task, not detection. Useful for methodology but doesn't answer HyDMIS's research question directly.

### Direct Clashes

**Clash 1:** XLM-R (Conneau 2020) vs RemBERT (Chung 2021; Gouliev 2025)
XLM-R claims strong multilingual performance. PolyTruth shows RemBERT consistently outperforms on low-resource subsets. Both are right, aggregate vs low-resource evaluation surfaces different winners.

**Clash 2:** GPT-4 generalization (OpenAI 2023) vs low-resource ceiling (2024-2025 literature)
OpenAI claims strong multilingual performance. Multiple 2024-2025 papers document performance degradation on genuinely low-resource languages. The conflict is unresolved, HyDMIS ablates this directly.

**Clash 3:** Aggregate benchmark evaluation vs community-specific evaluation
Every paper reports aggregate multilingual F1. No paper reports performance specifically for the communities most harmed. These produce different conclusions about system quality.

**Clash 4:** Dataset size as quality proxy vs annotation bottleneck reality
Scaling literature assumes more data = better performance. The 2024 survey confirms the real bottleneck is expert annotation not data volume, directly relevant to HyDMIS's LDA unsupervised Stage 1 design.

---

## Protocol 2, Contradiction Finder

| Contradiction | Paper A | Paper B | Resolution for HyDMIS |
|--------------|---------|---------|----------------------|
| Best backbone for low-resource | XLM-R dominant (Conneau 2020) | RemBERT better on low-resource (Gouliev 2025) | Ablate all three, report by language resource level |
| GPT-4 low-resource generalization | GPT-4 strong multilingual (OpenAI 2023) | Performance degrades on low-resource (2024-2025) | Run ablation Phase 4 Week 1, actual numbers replace estimates |
| Community-weighted loss effectiveness | Theoretically sound (fairness literature) | Mixed empirical results (2024 papers) | Run ablation before writing any claim about it |
| LDA for short social media text | LDA gold standard topic modeling (Blei 2003) | LDA struggles with short texts (NLP practice) | Pre-filter content length before LDA |
| Dataset size vs annotation quality | More data = better (scaling literature) | Bottleneck is annotation not data volume (2024 survey) | HyDMIS design validates annotation bottleneck claim |

---

## Protocol 3, Citation Chain (3 Concepts Tracked)

### Concept 1: Cross-lingual transfer for low-resource disinformation

BERT (2019) → XLM-R (2020) → RemBERT (2021) → PolyTruth (2025)

Each step established pretraining, scaled to 100 languages, redesigned embedding coupling, and empirically confirmed RemBERT's low-resource advantage. Where it stands now: RemBERT is the best-supported backbone for low-resource evaluation, but not yet tested on Southeast Asian or Caribbean Creole languages. That's HyDMIS's frontier.

### Concept 2: Unsupervised topic modeling for unlabeled multilingual content

Blei et al. LDA (2003) → Neural topic models (2017-2020) → LLM-based topic extraction (2023-2025)

LDA remains the most practical choice for zero-label settings, no API calls, interpretable clusters, works in any language. HyDMIS uses LDA in Stage 1 precisely because the pipeline must work where labeled data doesn't exist.

### Concept 3: Community-centered evaluation for disinformation systems

English-only FND (2017-2020) → Aggregate multilingual FND (2020-2023) → Low-resource focus (2023-2025) → Community-centered evaluation (HyDMIS)

An operational standard for community-centered disinformation evaluation remains an open problem in the field.

---

## Protocol 4, Gap Scanner (5 Gaps Ranked)

**Gap 1, Community-targeted low-resource evaluation [CRITICAL]**
No paper evaluates disinformation detection specifically for US Latino Spanish speakers, Filipino Tagalog speakers, and Haitian Creole communities. Aggregate multilingual benchmarks hide community-level failures.
Nearest attempt: PolyTruth (2025), European/Near Eastern focus only.
HyDMIS fills this gap directly.

**Gap 2, Hybrid pipeline at scale [HIGH]**
No paper combines LDA + LLM semantic verification + community-weighted cross-lingual classification in a single evaluated pipeline.
Nearest attempt: Alghamdi et al. (2024), hybrid summarization but not this combination.
HyDMIS's three-stage architecture fills this gap.

**Gap 3, Deployment-time bias amplification [HIGH]**
Does disinformation detection deployed at scale systematically over-flag content from historically scrutinized minority communities?
Nearest attempt: Fairness literature, not applied to disinformation systems specifically.
HyDMIS Stage 3 community-stratified false positive rate analysis fills this gap.

**Gap 4, GPT-4 verification on genuinely low-resource languages [MEDIUM]**
Does GPT-4 semantic verification generalize to Tagalog, Haitian Creole, Swahili?
Multiple 2024-2025 papers document the question but don't answer it.
HyDMIS Phase 4 ablation fills this gap.

**Gap 5, Agricultural and climate disinformation in low-resource languages [MEDIUM]**
No labeled agricultural disinformation dataset exists in Spanish, Tagalog, or Haitian Creole, the languages where it spreads fastest in farming communities.
ClimateMiSt covers English climate disinformation. MMCFND covers Indic languages.
HyDMIS documents this gap and positions it as future work.

---

## Protocol 5, Methodology Audit

### Strongest methodological papers:

**PolyTruth (Gouliev et al., 2025):** Systematic five-model comparison across 25+ languages with per-language resource level stratification. Right methodology, stratified evaluation surfaces what aggregate numbers hide.

**TruthSeeker (Dadkhah et al., 2023):** Ground truth from multiple fact-checking organizations. Reduces label bias. Limitation: English-only, Twitter/X only.

**ClimateMiSt (Choi, Shang, Wang 2025):** First dataset with both veracity and stance annotations for climate change. GPT-4 outperforms baselines on both tasks, directly validates HyDMIS Stage 2 GPT-4 verification approach.

**SemEval 2025 Task 7:** Shared task methodology, gold standard for fair comparison. Limitation: retrieval task, not detection.

### Weakest paper under methodological scrutiny:

**LIAR (Wang et al., 2017):** Six-way veracity labels from single annotator organization. 100% English, 100% formal political discourse. Holds up worst on: generalizability to social media content, non-English claims, health disinformation. HyDMIS uses LIAR2 (the extended version) only as baseline comparability benchmark.

---

## Protocol 6, Master Synthesis (400 words)

### What the Evidence Actually Supports

The multilingual disinformation detection field has a structural problem it has not solved: performance is measured where labels exist, not where harm accumulates.

LIAR established English political claim classification as the benchmark task. FakeNewsNet added social context. TruthSeeker verified at 134,198 labeled posts. ClimateMiSt added climate veracity and stance. MMCFND added seven Indic languages. These are well-executed papers addressing measurable problems. The communities where disinformation about COVID vaccines, GMOs, pesticide regulations, and farm subsidies spreads fastest, Spanish-speaking, Tagalog-speaking, Haitian Creole-speaking communities, produce abundant content and almost no labeled ground truth.

PolyTruth (2025) is the field's most honest recent paper. It quantifies the performance gap across 25 languages and produces actionable findings: RemBERT outperforms mBERT on low-resource subsets, aggregate accuracy numbers are misleading, and evaluation methodology itself needs to change. This is the paper HyDMIS builds directly on.

The transformer backbone debate is effectively resolved at the aggregate level but open at the community level. PolyTruth's RemBERT finding is persuasive for European and Near Eastern low-resource languages. Whether it holds for Southeast Asian and Caribbean Creole languages is the empirical question HyDMIS answers.

ClimateMiSt's finding that GPT-4 outperforms baseline models on both misinformation and stance detection validates HyDMIS's Stage 2 design, but only for English climate content. Whether that generalization holds for low-resource community languages remains untested.

### The Central Empirical Question

Does a hybrid three-stage pipeline combining LDA topic modeling, GPT-4 semantic verification, and RemBERT cross-lingual classification with community-weighted loss outperform single-model baselines on low-resource language subsets representing targeted minority communities, and does improved detection accuracy translate to measurable harm reduction for those communities?

Nobody has answered this. The literature has the components. Nobody has assembled and evaluated the combination at scale on community-targeted datasets across health, climate, and political disinformation domains simultaneously. That is HyDMIS's contribution.

---

## Protocol 7, Assumption Killer (6 Assumptions)

**Assumption 1, Community-weighted loss will improve low-resource performance:**
Theoretically sound. Empirically mixed in adjacent tasks. I'm not going to claim this works until Phase 4 Week 1 ablation confirms it. If the community-weighted loss doesn't outperform standard cross-entropy on low-resource subsets, the paper reports that honestly rather than burying it.

**Assumption 2, GPT-4 semantic verification generalizes to low-resource community languages:**
This is the most fragile assumption in HyDMIS. ClimateMiSt validates GPT-4 for English climate content, that's one domain, one language. Tagalog and Haitian Creole generalization is completely untested. I cannot write a paper claiming GPT-4 verification works on these languages until the Phase 4 ablation runs. If it doesn't generalize, Stage 2 gets replaced or the scope gets narrowed.

**Assumption 3, LDA topic clusters are meaningful for short social media text:**
LDA's short text limitation is documented going back to the topic modeling literature. Pre-filtering content below a minimum token threshold mitigates this but doesn't eliminate it. The real risk is noisy Stage 1 clusters propagating errors into Stage 2 and Stage 3, garbage in, garbage out across the whole pipeline. Monitoring cluster coherence scores before passing to Stage 2.

**Assumption 4, 324,292 verified samples provides sufficient low-resource community language coverage:**
Aggregate count looks strong. But when I break it down by language and community subset, Tagalog, Haitian Creole, and agricultural community language examples may be under 1K each. Fairness metrics become unreliable below roughly 500 examples per group. If subgroup sample sizes are too small, the community-stratified evaluation I'm planning becomes statistically indefensible.

**Assumption 5, Cross-domain transfer holds across health, climate, and political disinformation:**
LIAR2 is political. ClimateMiSt is climate. Covid-vaccine-misinfo-MIC is health. These domains use different vocabulary, different claim structures, different disinformation tactics. The pipeline must generalize across all three, tested in Phase 4 cross-dataset evaluation. If it doesn't generalize, the paper scope narrows to the domains where it does.

**Assumption 6, Detection improvement translates to harm reduction at deployment:**
This is the hardest assumption to defend and the most important one. Better F1 on a benchmark doesn't automatically mean less disinformation exposure for real communities. HyDMIS approximates harm reduction through platform-scale simulation rather than live deployment, which is the honest limitation of academic research that doesn't have API access to production systems. The paper will say this explicitly.

---

## Protocol 8, Knowledge Map

```
CORE PROBLEM
└── Disinformation harms underrepresented communities most
    ├── Detection systems trained on majority-language data
    ├── Performance gap quantified: 15-30 point F1 drop (PolyTruth 2025)
    └── Communities most targeted have least labeled data

METHODOLOGICAL LANDSCAPE
├── Unsupervised: LDA (Stage 1), works without labels
├── LLM verification: GPT-4, Mistral 7B (Stage 2)
│   └── ClimateMiSt confirms GPT-4 outperforms baselines on English
└── Cross-lingual classification (Stage 3)
    ├── mBERT, baseline, known low-resource degradation
    ├── XLM-R, stronger but still degrades
    └── RemBERT, best empirical evidence (PolyTruth 2025)

DATASET LANDSCAPE
├── English baselines: LIAR2, FakeNewsNet, TruthSeeker
├── Multilingual general: MultiClaim, NewsPolyML, DeFaktS
├── Health-specific: Covid-vaccine-misinfo-MIC
├── Climate/environment: ClimateMiSt (Dong Wang, UIUC)
└── South Asian low-resource: MMCFND (7 Indic languages)

AGRICULTURAL CONNECTION
├── Disinformation about GMOs, pesticides spreads in Spanish, Tagalog, Haitian Creole
├── Same communities underserved by agricultural AI systems (FAPE finding)
└── No labeled agricultural disinformation dataset exists, documented gap

OPEN QUESTIONS
├── Does RemBERT finding hold for Southeast Asian/Caribbean languages?
├── Does GPT-4 verification generalize beyond English/European?
├── Does community-weighted loss work empirically?
└── Does detection improvement translate to harm reduction?

HYDMIS CONTRIBUTION
└── First hybrid pipeline evaluated on community-targeted low-resource disinformation
    ├── 6 verified datasets, 324,292 samples, 15+ languages, 6 domains
    ├── LDA → GPT-4 → RemBERT/mBERT/Mistral combination
    ├── Community-weighted loss function
    └── Community-stratified evaluation not aggregate multilingual
```

**What I'm confident about going into Phase 4:**
- The performance gap is real and quantified. PolyTruth (2025) documents 15-30 point F1 drop on low-resource languages. This is the problem HyDMIS is solving.
- RemBERT is the best-supported backbone for low-resource evaluation, PolyTruth established this across 25 languages.
- LDA works without labeled data, the only practical choice for Stage 1 across languages where annotated disinformation datasets don't exist.
- GPT-4 outperforms baselines on English climate misinformation, ClimateMiSt confirmed this. Whether it generalizes is the open question.
- The six datasets I have verified (LIAR2, TruthSeeker, FakeNewsNet, Covid-vaccine-misinfo-MIC, NewsPolyML, DeFaktS) cover English, German, multilingual European, and health domains. MultiClaim and ClimateMiSt are pending access.

**What I think is true but haven't confirmed yet:**
- Community-weighted loss will outperform standard cross-entropy on low-resource subsets, theoretically sound, empirically untested in this configuration.
- The three-stage hybrid pipeline will outperform single-model baselines, the combination hasn't been evaluated together before.
- Agricultural disinformation in Spanish, Tagalog, and Haitian Creole follows similar linguistic patterns to health disinformation, same communities, same information ecosystems.

**What I'm genuinely uncertain about:**
- Whether GPT-4 verification degrades on Tagalog and Haitian Creole specifically, this is the most consequential uncertainty in the whole design.
- Whether subgroup sample sizes for low-resource community languages will be large enough for statistically reliable fairness metrics.
- Whether MultiClaim and ClimateMiSt access comes through before Phase 4 starts, the pipeline design accounts for them but the experiments can't wait indefinitely.

**The agricultural connection:**
The same communities underserved by agricultural AI systems in FAPE, small farmers, farm laborers, rural communities, are the communities most targeted by agricultural disinformation in Spanish, Tagalog, and Haitian Creole. No labeled agricultural disinformation dataset exists in these languages. HyDMIS documents this gap and positions it as future work while addressing the broader community-targeted detection problem.

---

## Protocol 9, So What Test (3 Points in Plain English)

**Point 1:**
Fake news detection works well in English. It fails badly in Spanish, Tagalog, and Haitian Creole, the languages where health and agricultural disinformation spreads fastest in minority communities. The performance gap is real, quantified, and unsolved.

**Point 2:**
HyDMIS builds the first pipeline specifically designed for this failure mode. Three stages: unsupervised topic clustering that works without labeled data, LLM semantic verification that handles nuanced claims across domains including health and climate, and cross-lingual classification with explicit weighting toward the communities being failed.

**Point 3:**
If it works, HyDMIS changes what multilingual disinformation detection means, from "works on aggregate benchmarks" to "works for the specific communities most targeted." That's the difference between a paper that reports good numbers and a system that actually reduces harm for minority farming, health, and civic communities.

---

## References

All references as listed in literature_review.md plus:
- Bansal et al. (2024), MMCFND: Multimodal Multilingual Fake News Detection, 7 Indic languages
- Choi, Shang, Wang (2025), ClimateMiSt: Climate Change Misinformation and Stance Detection Dataset, ASONAM 2024
- Kim et al. (2023), Covid-vaccine-misinfo-MIC, EMNLP 2023 (confirmed)
