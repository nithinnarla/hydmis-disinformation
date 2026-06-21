"""
HyDMIS — NewsPolyML EDA
Phase 4 — Exploratory Data Analysis
European Multilingual News Domain

EDA on NewsPolyML v2 — 32,129 European multilingual news claims (Mohtaj et al., 2024).
IFCN-certified fact-checked claims across EN/DE/ES/FR/IT.
Used as HyDMIS European multilingual evaluation benchmark.

Label: normalized_label (false/mixture/true/mislabeled/other)
Key finding: 71.4% false — severe class imbalance; LLaMA agreement 65.4%
"""

import pandas as pd
import numpy as np
import sys
import os
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from newspolyml_loader import load_newspolyml


def run_eda():
    print("HyDMIS Phase 4 — NewsPolyML EDA")
    print("=" * 50)

    result = load_newspolyml()
    df = result['data'].copy()

    df['body_len'] = df['article_body'].astype(str).str.len()
    df['body_words'] = df['article_body'].astype(str).str.split().str.len()
    df['claim_date'] = pd.to_datetime(df['claim_publication_date'], errors='coerce')
    df['year'] = df['claim_date'].dt.year

    print(f"\nDataset shape: {df.shape}")

    print(f"\n--- Dataset Overview ---")
    print(f"  Total records:     {len(df):,}")
    print(f"  Languages:         {df['article_language'].nunique()} (EN/DE/ES/FR/IT)")
    print(f"  Publishers:        {df['article_publisher_name'].nunique()}")
    print(f"  Label classes:     {df['normalized_label'].nunique()}")
    print(f"  Date range:        1995 to 2024 (30 years)")

    print(f"\n--- Normalized Label Distribution ---")
    label_counts = df['normalized_label'].value_counts()
    for label, count in label_counts.items():
        print(f"  {label:<15} {count:,} ({count/len(df):.1%})")
    print(f"  Note: false dominates at {label_counts.get('false',0)/len(df):.1%} — severe class imbalance")
    print(f"  HyDMIS binary mapping: false+mislabeled=disinformation, true=credible, mixture=ambiguous")

    print(f"\n--- Language Distribution ---")
    lang_counts = df['article_language'].value_counts()
    lang_names = {'en': 'English', 'es': 'Spanish', 'it': 'Italian', 'de': 'German', 'fr': 'French'}
    for lang, count in lang_counts.items():
        print(f"  {lang} ({lang_names.get(lang, lang):<10}) {count:,} ({count/len(df):.1%})")
    print(f"  Note: English + Spanish = {(lang_counts.get('en',0)+lang_counts.get('es',0))/len(df):.1%} — most coverage")

    print(f"\n--- Publisher Distribution ---")
    pub_counts = df['article_publisher_name'].value_counts()
    for pub, count in pub_counts.items():
        print(f"  {pub:<25} {count:,} ({count/len(df):.1%})")

    print(f"\n--- Label by Language ---")
    for lang in ['en', 'es', 'it', 'de', 'fr']:
        subset = df[df['article_language'] == lang]
        false_rate = (subset['normalized_label'] == 'false').mean()
        true_rate = (subset['normalized_label'] == 'true').mean()
        mix_rate = (subset['normalized_label'] == 'mixture').mean()
        print(f"  {lang} ({lang_names.get(lang):<10}) false={false_rate:.1%} true={true_rate:.1%} mixture={mix_rate:.1%} n={len(subset):,}")

    print(f"\n--- LLaMA Label Distribution ---")
    llama_counts = df['llama_label'].value_counts()
    for label, count in llama_counts.items():
        print(f"  {label:<15} {count:,} ({count/len(df):.1%})")

    print(f"\n--- LLaMA vs Normalized Label Agreement ---")
    df_both = df[df['normalized_label'].notna() & df['llama_label'].notna()]
    agree = (df_both['normalized_label'] == df_both['llama_label']).sum()
    print(f"  Agreement: {agree:,}/{len(df_both):,} ({agree/len(df_both):.1%})")
    print(f"  Disagreement: {len(df_both)-agree:,} ({(len(df_both)-agree)/len(df_both):.1%})")
    print(f"  Note: 34.6% disagreement — LLaMA overestimates false, underestimates mixture")

    print(f"\n--- LLaMA Confidence Distribution ---")
    conf_counts = df['llama_confidence'].value_counts().sort_index()
    for conf, count in conf_counts.items():
        print(f"  Confidence {conf:.0f}: {count:,} ({count/len(df):.1%})")
    print(f"  mean={df['llama_confidence'].mean():.2f} | median={df['llama_confidence'].median():.2f}")
    print(f"  Note: 83.6% confidence=5 — LLaMA highly confident but 34.6% wrong")

    print(f"\n--- Article Sentiment Distribution ---")
    sent_counts = df['article_sentiment'].value_counts()
    for sent, count in sent_counts.items():
        print(f"  {sent:<12} {count:,} ({count/len(df):.1%})")
    print(f"  Note: 93.9% negative — fact-check articles overwhelmingly negative framing")

    print(f"\n--- Article Body Length ---")
    print(f"  mean={df['body_len'].mean():.0f} chars | median={df['body_len'].median():.0f}")
    print(f"  mean={df['body_words'].mean():.0f} words | median={df['body_words'].median():.0f}")
    for lang in ['en', 'es', 'it', 'de', 'fr']:
        subset = df[df['article_language'] == lang]
        print(f"  {lang} ({lang_names.get(lang):<10}) mean={subset['body_len'].mean():.0f} chars | mean={subset['body_words'].mean():.0f} words")

    print(f"\n--- Temporal Distribution ---")
    year_counts = df['year'].value_counts().sort_index()
    recent = df[df['year'] >= 2019]['year'].notna().sum()
    print(f"  Date range: 1995-2024 (30 years)")
    print(f"  Records with dates: {df['claim_date'].notna().sum():,} ({df['claim_date'].notna().mean():.1%})")
    print(f"  Records 2019+: {recent:,} ({recent/len(df):.1%}) — post-COVID disinformation era")
    for year in [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]:
        count = year_counts.get(year, 0)
        print(f"  {year}: {count:,}")

    print(f"\n--- Article Sentiment by Language ---")
    for lang in ['en', 'es', 'it', 'de', 'fr']:
        subset = df[df['article_language'] == lang]
        neg = subset['article_sentiment'].eq('negative').mean()
        pos = subset['article_sentiment'].eq('positive').mean()
        neu = subset['article_sentiment'].eq('neutral').mean()
        print(f"  {lang} ({lang_names.get(lang):<10}) neg={neg:.1%} pos={pos:.1%} neu={neu:.1%} n={len(subset):,}")
    it_neg = df[df['article_language']=='it']['article_sentiment'].eq('negative').mean()
    print(f"  Note: Italian neg={it_neg:.1%} — lower than EN/ES due to pagellapolitica rating style")

    print(f"\n--- Claim Review Sentiment ---")
    review_sent = df['claim_review_sentiment'].value_counts()
    for sent, count in review_sent.items():
        print(f"  {sent:<12} {count:,} ({count/len(df):.1%})")
    print(f"  Note: claim_review_sentiment vs article_sentiment — reviewer framing vs article framing")
    for lang in ['en', 'es', 'it', 'de', 'fr']:
        subset = df[df['article_language'] == lang]
        neg = subset['claim_review_sentiment'].eq('negative').mean()
        print(f"  {lang} ({lang_names.get(lang):<10}) review_neg={neg:.1%}")

    print(f"\n--- LLaMA Confidence by Language ---")
    for lang in ['en', 'es', 'it', 'de', 'fr']:
        subset = df[df['article_language'] == lang]
        mean_conf = subset['llama_confidence'].mean()
        high_conf = subset['llama_confidence'].eq(5).mean()
        print(f"  {lang} ({lang_names.get(lang):<10}) mean={mean_conf:.2f} | conf=5: {high_conf:.1%} n={len(subset):,}")
    print(f"  Note: Italian lower confidence reflects label ambiguity in mixture-heavy dataset")

    print(f"\n--- Italian Label Anomaly ---")
    it_subset = df[df['article_language'] == 'it']
    afp_subset = df[df['article_publisher_name'] == 'afp']
    pag_subset = df[df['article_publisher_name'] == 'pagellapolitica']
    print(f"  Italian (pagellapolitica): false={it_subset['normalized_label'].eq('false').mean():.1%} true={it_subset['normalized_label'].eq('true').mean():.1%} mixture={it_subset['normalized_label'].eq('mixture').mean():.1%}")
    print(f"  AFP overall:               false={afp_subset['normalized_label'].eq('false').mean():.1%} true={afp_subset['normalized_label'].eq('true').mean():.1%} mixture={afp_subset['normalized_label'].eq('mixture').mean():.1%}")
    print(f"  Note: Italian anomaly is publisher-driven — pagellapolitica uses different rating scale")
    print(f"  HyDMIS must account for publisher-level label bias in cross-lingual evaluation")

    print(f"\n--- Missing Values ---")
    key_cols = ['normalized_label', 'article_language', 'article_body',
                'article_sentiment', 'llama_label', 'llama_confidence',
                'claim_publication_date', 'article_description']
    for col in key_cols:
        nulls = df[col].isnull().sum()
        print(f"  {col:<35} {nulls:,} ({nulls/len(df):.1%})")

    print(f"\n--- Key Observations ---")
    print(f"  Total records: {len(df):,} across 5 European languages")
    print(f"  false dominates: {label_counts.get('false',0):,} ({label_counts.get('false',0)/len(df):.1%}) — severe class imbalance")
    print(f"  AFP covers {pub_counts.get('afp',0)/len(df):.1%} — single publisher dominance may bias cross-publisher evaluation")
    print(f"  LLaMA agreement: 65.4% — 34.6% disagreement signals label noise in mixture class")
    print(f"  93.9% negative sentiment — consistent with fact-check article framing")
    print(f"  Italian anomaly: pagellapolitica false=16.7% vs AFP false=78.3% — publisher-driven label bias")
    print(f"  HyDMIS uses NewsPolyML for European multilingual disinformation evaluation")

    print(f"\n--- NewsPolyML EDA complete ---")
    print(f"  Ready for HyDMIS Stage 1 LDA topic modeling")

    return df


if __name__ == "__main__":
    df = run_eda()
