"""
HyDMIS, FakeNewsNet EDA
Phase 4, Exploratory Data Analysis
News Disinformation Domain

EDA on FakeNewsNet dataset, 23,196 news articles (Shu et al., 2020).
PolitiFact (political) and GossipCop (celebrity) news sources.
Content features only, title text available, no article body.

Label mapping: 1=Disinformation, 0=Credible
Disinformation rate: 24.8%, imbalanced dataset.
GossipCop dominates: 95.4% of records.
"""

import pandas as pd
import numpy as np
import sys
import os
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from fakenewsnet_loader import load_fakenewsnet

SOURCES = ['politifact_fake','politifact_real','gossipcop_fake','gossipcop_real']
SOURCE_COLORS = {'politifact_fake':'#d9534f','politifact_real':'#5cb85c',
                 'gossipcop_fake':'#f0ad4e','gossipcop_real':'steelblue'}


def run_eda():
    print("HyDMIS Phase 4, FakeNewsNet EDA")
    print("=" * 50)

    result = load_fakenewsnet(data_dir='data/raw')
    df = result['data']
    df['title_len'] = df['title'].astype(str).str.len()
    df['word_count'] = df['title'].astype(str).str.split().str.len()
    df['platform'] = df['source'].str.split('_').str[0]

    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    print(f"\n--- Label Distribution ---")
    for label, name in [(1,'Disinformation'),(0,'Credible')]:
        count = (df['label']==label).sum()
        print(f"  {name:<15} {count:,} ({count/len(df):.1%})")
    print(f"  Note: Imbalanced, 24.8% disinformation")

    print(f"\n--- Source Distribution ---")
    for src, count in df['source'].value_counts().items():
        dis_rate = df[df['source']==src]['label'].mean()
        print(f"  {src:<25} n={count:,} ({count/len(df):.1%}) | dis_rate={dis_rate:.1%}")

    print(f"\n--- Platform Distribution ---")
    for platform in ['gossipcop','politifact']:
        subset = df[df['platform']==platform]
        print(f"  {platform:<15} n={len(subset):,} ({len(subset)/len(df):.1%}) | dis_rate={subset['label'].mean():.1%}")

    print(f"\n--- Title Length Distribution ---")
    print(f"  Mean: {df['title_len'].mean():.0f} chars | Median: {df['title_len'].median():.0f}")
    print(f"  Min: {df['title_len'].min()} | Max: {df['title_len'].max()}")
    corr = df['title_len'].corr(df['label'])
    print(f"  Title length-disinformation correlation: {corr:.3f}")
    for label, name in [(1,'Disinformation'),(0,'Credible')]:
        mean_len = df[df['label']==label]['title_len'].mean()
        print(f"  {name:<15}: mean {mean_len:.0f} chars")

    print(f"\n--- Title Length by Source ---")
    for src in SOURCES:
        subset = df[df['source']==src]
        print(f"  {src:<25} mean={subset['title_len'].mean():.0f} | median={subset['title_len'].median():.0f}")

    print(f"\n--- Keyword Analysis by Platform ---")
    from collections import Counter
    import re
    stopwords = {'the','a','an','in','of','to','and','is','for','on','at','by',
                 'with','from','this','that','was','are','be','as','it','its',
                 'not','have','has','had','he','she','they','his','her','their',
                 'will','would','could','should','but','or','about','after','before'}

    for platform in ['politifact', 'gossipcop']:
        print(f"  {platform.upper()}:")
        for label, name in [(1,'Disinformation'),(0,'Credible')]:
            subset = df[(df['platform']==platform) & (df['label']==label)]
            words = []
            for t in subset['title'].astype(str):
                words.extend([w.lower() for w in re.findall(r'\b[a-zA-Z]{3,}\b', t)
                             if w.lower() not in stopwords])
            top = Counter(words).most_common(8)
            print(f"    {name}: {[w for w,c in top]}")



    print(f"\n--- Word Count Distribution ---")
    print(f"  Mean: {df['word_count'].mean():.1f} words | Median: {df['word_count'].median():.1f}")
    for label, name in [(1,'Disinformation'),(0,'Credible')]:
        mean_wc = df[df['label']==label]['word_count'].mean()
        print(f"  {name:<15}: mean {mean_wc:.1f} words")

    print(f"\n--- Metadata Availability Signals ---")
    df['has_url'] = df['news_url'].notna()
    df['has_tweets'] = df['tweet_ids'].notna()
    for has, name in [(True,'Has URL'),(False,'No URL')]:
        subset = df[df['has_url']==has]
        print(f"  {name:<12} n={len(subset):,} | dis_rate={subset['label'].mean():.1%}")
    for has, name in [(True,'Has tweets'),(False,'No tweets')]:
        subset = df[df['has_tweets']==has]
        print(f"  {name:<15} n={len(subset):,} | dis_rate={subset['label'].mean():.1%}")
    print(f"  Note: No URL = 78.8% dis rate, missing URL is strong fake signal")

    print(f"\n--- Missing Values ---")
    nulls = df[['news_url','title','tweet_ids']].isnull().sum()
    for col, n in nulls[nulls>0].items():
        print(f"  {col:<20} {n:,} ({n/len(df):.1%})")

    print(f"\n--- Key Observations ---")
    print(f"  Total records: {len(df):,}")
    print(f"  Disinformation rate: {df['label'].mean():.1%}, imbalanced")
    print(f"  GossipCop dominates: {(df['platform']=='gossipcop').mean():.1%} of records")
    print(f"  Only title text available, no article body")
    print(f"  PolitiFact disinformation rate: {df[df['platform']=='politifact']['label'].mean():.1%}")
    print(f"  GossipCop disinformation rate: {df[df['platform']=='gossipcop']['label'].mean():.1%}")
    print(f"  tweet_ids available but social graph not used, content-only by design")

    print(f"\n--- FakeNewsNet EDA complete ---")
    print(f"  Ready for HyDMIS Stage 1 LDA topic modeling")

    return df


if __name__ == "__main__":
    df = run_eda()
