"""
HyDMIS, GPT-4 Semantic Verification Sampler
Phase 4, Stage 2: GPT-4 Semantic Verification Setup

Builds stratified 15K sample across all 5 HyDMIS datasets and language groups
for GPT-4 semantic verification of LDA topic assignments.

Architecture (Decision 4):
- GPT-4 labels 15K representative samples → fine-tunes Mistral 7B
- Mistral 7B deployed for full-scale verification (95% cost reduction)
- Stratification by dataset + language group + veracity label

Column mapping (verified against loaders):
- LIAR2:       text=statement,      label=label,            language=en
- TruthSeeker: text=statement,      label=BinaryNumTarget,  language=en
- FakeNewsNet: text=title,          label=label,            language=en
- DeFaktS:     text=text,           label=binary_label,     language=Language
- NewsPolyML:  text=article_description, label=normalized_label, language=article_language

Output:
- data/processed/gpt4_sample.csv, 15K stratified sample with prompts
- data/processed/gpt4_sample_stats.csv, sample statistics

Script type: pipeline/infrastructure, no notebook, no figures
"""

import os
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)

sys.path.insert(0, os.path.dirname(__file__))

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(REPO_ROOT, 'data', 'processed')
os.makedirs(PROCESSED_DIR, exist_ok=True)

TOTAL_SAMPLE = 15000

SAMPLE_ALLOCATION = {
    'liar2':       4000,
    'truthseeker': 4000,
    'fakenewsnet': 3000,
    'defakts':     2000,
    'newspolyml':  2000,
}

np.random.seed(42)


def build_verification_prompt(text, dataset, language, lda_topic_id, lda_top_words):
    """Build GPT-4 prompt for semantic verification of LDA topic assignment."""
    return (
        f"You are a fact-checking assistant evaluating multilingual disinformation.\n\n"
        f"Dataset: {dataset}\n"
        f"Language: {language}\n"
        f"Text: {str(text)[:500]}\n\n"
        f"LDA Topic Assignment: Topic {lda_topic_id}\n"
        f"Topic Keywords: {lda_top_words}\n\n"
        f"Tasks:\n"
        f"1. Is this text disinformation? Answer: YES / NO / UNCERTAIN\n"
        f"2. Does the LDA topic assignment match the text content? Answer: MATCH / MISMATCH / PARTIAL\n"
        f"3. What is the primary disinformation category? "
        f"Answer: HEALTH / POLITICAL / ECONOMIC / SOCIAL / OTHER\n\n"
        f"Respond in JSON format only:\n"
        f'{{"disinformation": "YES/NO/UNCERTAIN", '
        f'"topic_match": "MATCH/MISMATCH/PARTIAL", '
        f'"category": "HEALTH/POLITICAL/ECONOMIC/SOCIAL/OTHER"}}'
    )


def stratified_sample(df, label_col, n, random_state=42):
    """Stratified sample by label column."""
    n_labels = df[label_col].nunique()
    if n_labels == 0:
        return df.sample(min(n, len(df)), random_state=random_state)
    sampled = df.groupby(label_col, group_keys=False).apply(
        lambda x: x.sample(min(len(x), n // n_labels), random_state=random_state)
    )
    return sampled.sample(min(n, len(sampled)), random_state=random_state)


def load_and_sample_liar2(n):
    from liar2_loader import load_liar2
    d = load_liar2()
    df = d['data'].copy()
    df['text'] = df['statement'].astype(str)
    df['veracity'] = df['label'].astype(str)
    df['language'] = 'en'
    df['dataset'] = 'liar2'
    df['record_id'] = df.index
    sampled = stratified_sample(df, 'veracity', n)
    print(f"  LIAR2:       {len(sampled):,} records sampled")
    return sampled[['text', 'dataset', 'language', 'veracity', 'record_id']]


def load_and_sample_truthseeker(n):
    from truthseeker_loader import load_truthseeker
    d = load_truthseeker()
    df = d['data'].copy()
    df['text'] = df['statement'].astype(str)
    df['veracity'] = df['BinaryNumTarget'].astype(str)
    df['language'] = 'en'
    df['dataset'] = 'truthseeker'
    df['record_id'] = df.index
    sampled = stratified_sample(df, 'veracity', n)
    print(f"  TruthSeeker: {len(sampled):,} records sampled")
    return sampled[['text', 'dataset', 'language', 'veracity', 'record_id']]


def load_and_sample_fakenewsnet(n):
    from fakenewsnet_loader import load_fakenewsnet
    d = load_fakenewsnet()
    df = d['data'].copy()
    df['text'] = df['title'].astype(str)
    df['veracity'] = df['label'].astype(str)
    df['language'] = 'en'
    df['dataset'] = 'fakenewsnet'
    df['record_id'] = df.index
    sampled = stratified_sample(df, 'veracity', n)
    print(f"  FakeNewsNet: {len(sampled):,} records sampled")
    return sampled[['text', 'dataset', 'language', 'veracity', 'record_id']]


def load_and_sample_defakts(n):
    from defakts_loader import load_defakts
    d = load_defakts()
    df = d['data'].copy()
    df['text'] = df['text'].astype(str)
    df['veracity'] = df['binary_label'].astype(str)
    df['language'] = df['Language'].astype(str)
    df['dataset'] = 'defakts'
    df['record_id'] = df.index
    sampled = stratified_sample(df, 'veracity', n)
    print(f"  DeFaktS:     {len(sampled):,} records sampled")
    return sampled[['text', 'dataset', 'language', 'veracity', 'record_id']]


def load_and_sample_newspolyml(n):
    from newspolyml_loader import load_newspolyml
    d = load_newspolyml()
    df = d['data'].copy()
    df['text'] = df['article_description'].astype(str)
    df['veracity'] = df['normalized_label'].astype(str)
    df['language'] = df['article_language'].astype(str)
    df['dataset'] = 'newspolyml'
    df['record_id'] = df.index
    # Drop rows with empty text
    df = df[df['text'].str.len() > 10]
    sampled = stratified_sample(df, 'veracity', n)
    print(f"  NewsPolyML:  {len(sampled):,} records sampled")
    return sampled[['text', 'dataset', 'language', 'veracity', 'record_id']]


def run_gpt4_sampler():
    print("HyDMIS Phase 4, Stage 2: GPT-4 Semantic Verification Sampler")
    print("=" * 65)
    print(f"  Target sample: {TOTAL_SAMPLE:,} records across 5 datasets")

    print("\n--- Loading and Sampling Datasets ---")

    frames = []
    loaders = [
        ('liar2',       load_and_sample_liar2,       SAMPLE_ALLOCATION['liar2']),
        ('truthseeker', load_and_sample_truthseeker,  SAMPLE_ALLOCATION['truthseeker']),
        ('fakenewsnet', load_and_sample_fakenewsnet,  SAMPLE_ALLOCATION['fakenewsnet']),
        ('defakts',     load_and_sample_defakts,      SAMPLE_ALLOCATION['defakts']),
        ('newspolyml',  load_and_sample_newspolyml,   SAMPLE_ALLOCATION['newspolyml']),
    ]

    for name, loader_fn, n in loaders:
        try:
            df = loader_fn(n)
            frames.append(df)
        except Exception as e:
            print(f"  WARNING: {name} failed, {e}")

    if not frames:
        print("ERROR: No datasets loaded successfully")
        return None

    sample = pd.concat(frames, ignore_index=True)
    sample = sample.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\n--- Sample Statistics ---")
    print(f"  Total records: {len(sample):,}")
    print(f"  Dataset distribution:")
    for ds, count in sample['dataset'].value_counts().items():
        print(f"    {ds:<15} {count:,}")
    print(f"  Language distribution:")
    for lang, count in sample['language'].value_counts().items():
        print(f"    {lang:<15} {count:,}")

    print("\n--- Building GPT-4 Prompts ---")
    sample['lda_topic_id'] = -1
    sample['lda_top_words'] = ''
    sample['gpt4_prompt'] = sample.apply(
        lambda row: build_verification_prompt(
            row['text'], row['dataset'], row['language'],
            row['lda_topic_id'], row['lda_top_words']
        ), axis=1
    )
    sample['gpt4_label'] = ''
    sample['gpt4_topic_match'] = ''
    sample['gpt4_category'] = ''

    out_path = os.path.join(PROCESSED_DIR, 'gpt4_sample.csv')
    sample.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")

    stats = sample.groupby(['dataset', 'language', 'veracity']).size().reset_index(name='count')
    stats_path = os.path.join(PROCESSED_DIR, 'gpt4_sample_stats.csv')
    stats.to_csv(stats_path, index=False)
    print(f"  Saved: {stats_path}")

    print(f"\n--- GPT-4 Sampler complete ---")
    print(f"  {len(sample):,} records ready for GPT-4 verification")
    print(f"  Prompts built, awaiting GPT-4 API integration (Jul 28)")
    print(f"  LDA topic assignments pending, run lda_pipeline first")
    cost_low = len(sample) * 0.0002
    cost_high = len(sample) * 0.0003
    print(f"  Cost estimate: ~${cost_low:.0f}-${cost_high:.0f} at GPT-4 rates")

    return sample


if __name__ == "__main__":
    run_gpt4_sampler()
