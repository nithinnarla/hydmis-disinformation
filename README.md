# HyDMIS — Hybrid Disinformation Mitigation System

## Overview
HyDMIS is a three-stage hybrid NLP pipeline for multilingual disinformation detection on social media. It combines unsupervised topic modeling, LLM-based semantic verification, and cross-lingual transformer classification to detect disinformation across languages, platforms, and communities.

Evaluated across 7 real-world datasets totaling 387,000+ samples spanning health, political, and social domains across 10+ languages.

## Research Question
How can hybrid NLP pipelines combining unsupervised topic modeling with cross-lingual transformer classification improve disinformation detection accuracy across low-resource languages and underrepresented communities?

## Pipeline Architecture

**Stage 1 — Topic Modeling:**
- LDA (Latent Dirichlet Allocation)
- Identifies latent topic clusters in social media content
- Unsupervised — no labels required at this stage
- Filters irrelevant content before classification

**Stage 2 — Semantic Verification:**
- GPT-4 semantic analysis
- Context-aware claim credibility scoring
- Handles nuanced and ambiguous language
- Bridges topic clusters to classification

**Stage 3 — Cross-Lingual Classification:**
- mBERT (Multilingual BERT)
- Mistral 7B
- Cross-lingual transfer learning
- Optimized for low-resource language support

## Datasets

| Dataset | Year | Samples | Language | Domain |
|---------|------|---------|----------|--------|
| LIAR | 2017 | 12,800+ | English | News claims |
| FakeNewsNet | 2020 | 23,000+ | English | News articles |
| MultiClaim | 2023 | 28,000+ | Multilingual | Social media |
| Covid-vaccine-misinfo-MIC | 2023 | 5,952 | Multilingual | Health/Social |
| TruthSeeker | 2023 | 180,000+ | English | Twitter/X |
| NewsPolyML | 2024 | 32,000+ | European | News/Politics |
| DeFaktS | 2024 | 105,855 | German/Multi | Twitter/X |

**Total: 387,000+ samples across 7 datasets, 
10+ languages, 4 domains**

Dataset notes:
- LIAR and FakeNewsNet used as established English baselines
- TruthSeeker (2023) is the largest labeled social media fake news dataset in existence (Dadkhah et al., IEEE 2023)
- MultiClaim and Covid-vaccine-misinfo-MIC address multilingual and underrepresented community coverage
- NewsPolyML provides IFCN-certified European multilingual fact-checked claims
- DeFaktS (2024) provides fine-grained Twitter/X labels across elections, climate, and health topics
- Note: Disinformation datasets are inherently smaller than general tabular ML datasets due to expert fact-checking requirements. 387K+       represents one of the largest multi-dataset collections in this field.

## Evaluation Metrics
- F1 score (macro + weighted)
- Precision and Recall per class
- Cross-lingual transfer accuracy
- Low-resource language performance
- Per-domain classification analysis
- Statistical significance testing across all language groups

## Tech Stack
Python, LDA (scikit-learn), GPT-4, mBERT, Mistral 7B, HuggingFace Transformers, NLTK, pandas, numpY, matplotlib, seaborn

## Research Timeline
- February 2026: Research conception and literature review
- March 2026: Pipeline architecture design
- April 2026: Stage 1 LDA implementation
- May 2026: Stage 2 GPT-4 semantic verification
- June 2026: Stage 3 mBERT + Mistral classification
- August 2026: Cross-lingual evaluation and ablation studies
- November 2026: Target submission to EMNLP

## Status
🔬 Research in progress
Target venue: EMNLP 2026

## Paper
"HyDMIS: Hybrid Disinformation Mitigation Using Topic Modeling and Cross-Lingual Classification" — Under development

## References
- Wang et al. (2017) — LIAR: A Benchmark Dataset for 
  Fake News Detection
- Shu et al. (2020) — FakeNewsNet: A Data Repository 
  for Fake News Detection
- Pikuliak et al. (2023) — MultiClaim: Multilingual 
  Claim Detection
- Kim et al. (2023) — Covid-vaccine-misinfo-MIC Dataset
- Dadkhah et al. (2023) — TruthSeeker: The Largest 
  Social Media Ground-Truth Dataset, IEEE TCSS
- Mohtaj et al. (2024) — NewsPolyML: Multi-lingual 
  European News Fake Assessment Dataset
- Ashraf et al. (2024) — DeFaktS: German Fact-Checking 
  Dataset
- Devlin et al. (2019) — BERT: Pre-training of Deep 
  Bidirectional Transformers
- Jiang et al. (2023) — Mistral 7B
- Blei et al. (2003) — Latent Dirichlet Allocation
- OpenAI (2023) — GPT-4 Technical Report
