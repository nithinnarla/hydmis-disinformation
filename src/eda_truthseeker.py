"""
HyDMIS — TruthSeeker EDA
Phase 4 — Exploratory Data Analysis
Social Media Disinformation Domain

EDA on TruthSeeker dataset — 134,198 social media posts (Dadkhah et al., 2023).
Largest labeled social media fake news dataset in existence.
Primary English social media benchmark for HyDMIS.

Label mapping:
  BinaryNumTarget=1 (target=True)  = Credible
  BinaryNumTarget=0 (target=False) = Disinformation
  Binary disinformation rate: 48.6%

Unique structure: each record has both a statement AND a tweet response.
Author = PolitiFact fact-checker, not claim-maker.
"""

import pandas as pd
import numpy as np
import sys
import os
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from truthseeker_loader import load_truthseeker

# BinaryNumTarget=1=Credible, 0=Disinformation
BINARY_LABELS = {1:"Credible", 0:"Disinformation"}
FIVE_LABELS = ["Agree","Mostly Agree","NO MAJORITY","Mostly Disagree","Disagree"]
THREE_LABELS = ["Agree","Disagree"]


def run_eda():
    print("HyDMIS Phase 4 — TruthSeeker EDA")
    print("=" * 50)

    result = load_truthseeker()
    df = result["data"]
    df["binary_label"] = 1 - df["BinaryNumTarget"]  # flip: 1=disinformation, 0=credible
    df["stmt_len"] = df["statement"].astype(str).str.len()
    df["tweet_len"] = df["tweet"].astype(str).str.len()

    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    print(f"\n--- Label Distribution (Binary) ---")
    credible = (df["BinaryNumTarget"]==1).sum()
    disinfo = (df["BinaryNumTarget"]==0).sum()
    print(f"  Credible (1):        {credible:,} ({credible/len(df):.1%})")
    print(f"  Disinformation (0):  {disinfo:,} ({disinfo/len(df):.1%})")
    print(f"  Note: Nearly balanced — 51.4% credible vs 48.6% disinformation")
    print(f"  Note: binary_label flipped for HyDMIS consistency: 1=disinformation")

    print(f"\n--- Five-Way Label Distribution ---")
    five_counts = df["5_label_majority_answer"].value_counts()
    for label in FIVE_LABELS:
        count = five_counts.get(label, 0)
        print(f"  {label:<20} {count:,} ({count/len(df):.1%})")
    print(f"  Note: NO MAJORITY (16.8%) — ambiguous crowd annotation cases")

    print(f"\n--- Three-Way Label Distribution ---")
    three_counts = df["3_label_majority_answer"].value_counts()
    for label in THREE_LABELS:
        count = three_counts.get(label, 0)
        print(f"  {label:<12} {count:,} ({count/len(df):.1%})")

    print(f"\n--- Top 15 Authors ---")
    author_stats = df.groupby("author").agg(
        count=("binary_label","count"),
        dis_rate=("binary_label","mean")
    ).reset_index().sort_values("count", ascending=False).head(15)
    for _, row in author_stats.iterrows():
        print(f"  {str(row['author']):<30} n={int(row['count']):,} | dis rate: {row['dis_rate']:.1%}")

    print(f"\n--- Statement Length Distribution ---")
    print(f"  Mean: {df['stmt_len'].mean():.0f} chars")
    print(f"  Median: {df['stmt_len'].median():.0f} chars")
    print(f"  Min: {df['stmt_len'].min()} | Max: {df['stmt_len'].max()}")
    corr = df["stmt_len"].corr(df["binary_label"])
    print(f"  Statement length-disinformation correlation: {corr:.3f}")
    for label, name in [(1,"Disinformation"),(0,"Credible")]:
        mean_len = df[df["BinaryNumTarget"]==label]["stmt_len"].mean()
        print(f"  {name:<15}: mean {mean_len:.0f} chars")

    print(f"\n--- Tweet Length Distribution ---")
    print(f"  Mean: {df['tweet_len'].mean():.0f} chars")
    print(f"  Median: {df['tweet_len'].median():.0f} chars")
    print(f"  Min: {df['tweet_len'].min()} | Max: {df['tweet_len'].max()}")
    corr_tweet = df["tweet_len"].corr(df["binary_label"])
    print(f"  Tweet length-disinformation correlation: {corr_tweet:.3f}")
    for label, name in [(1,"Disinformation"),(0,"Credible")]:
        mean_len = df[df["BinaryNumTarget"]==label]["tweet_len"].mean()
        print(f"  {name:<15}: mean {mean_len:.0f} chars")

    print(f"\n--- Author Disinformation Rate Distribution ---")
    author_dis = df.groupby("author")["binary_label"].mean()
    print(f"  Authors with >80% dis rate: {(author_dis>0.8).sum()}")
    print(f"  Authors with <20% dis rate: {(author_dis<0.2).sum()}")
    print(f"  Mean author dis rate: {author_dis.mean():.1%}")
    print(f"  Std author dis rate: {author_dis.std():.1%}")

    print(f"\n--- Keyword Coverage ---")
    has_keywords = df["manual_keywords"].notna().sum()
    print(f"  Records with keywords: {has_keywords:,} ({has_keywords/len(df):.1%})")
    if has_keywords > 0:
        all_keywords = df["manual_keywords"].dropna().str.split(",").explode().str.strip()
        print(f"  Unique keywords: {all_keywords.nunique():,}")
        print(f"  Top 10 keywords:")
        for kw, cnt in all_keywords.value_counts().head(10).items():
            print(f"    {str(kw):<30} n={cnt:,}")

    print(f"\n--- Missing Values ---")
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    if len(nulls) == 0:
        print("  No missing values")
    else:
        for col, n in nulls.items():
            print(f"  {col:<25} {n:,} ({n/len(df):.1%})")


    print(f"\n--- Keywords by Label ---")
    dis_kw = df[df["binary_label"]==1]["manual_keywords"].dropna().str.split(",").explode().str.strip()
    cred_kw = df[df["binary_label"]==0]["manual_keywords"].dropna().str.split(",").explode().str.strip()
    print(f"  Top 10 Disinformation keywords:")
    for kw, cnt in dis_kw.value_counts().head(10).items():
        print(f"    {str(kw):<30} n={cnt:,}")
    print(f"  Top 10 Credible keywords:")
    for kw, cnt in cred_kw.value_counts().head(10).items():
        print(f"    {str(kw):<30} n={cnt:,}")
    print(f"  Note: Disinformation — COVID/election topics; Credible — policy debates")


    print(f"\n--- NO MAJORITY Analysis ---")
    no_maj = df[df['5_label_majority_answer']=='NO MAJORITY']
    has_maj = df[df['5_label_majority_answer']!='NO MAJORITY']
    print(f"  NO MAJORITY records: {len(no_maj):,} ({len(no_maj)/len(df):.1%})")
    print(f"  NO MAJORITY dis rate: {no_maj['binary_label'].mean():.1%}")
    print(f"  Has majority dis rate: {has_maj['binary_label'].mean():.1%}")
    print(f"  Note: NO MAJORITY cases ambiguous — special handling needed in HyDMIS Stage 1")

    print(f"\n--- Key Observations ---")
    print(f"  Total records: {len(df):,}")
    print(f"  Binary disinformation rate: {df['binary_label'].mean():.1%}")
    print(f"  Near-balanced dataset — no severe class imbalance")
    print(f"  Dual-text structure: statement + tweet response — unique to TruthSeeker")
    print(f"  NO MAJORITY cases (16.8%) require careful handling in HyDMIS Stage 1")
    print(f"  Author signal: fact-checker identity may encode credibility bias")

    print(f"\n--- TruthSeeker EDA complete ---")
    print(f"  Ready for HyDMIS Stage 1 LDA topic modeling")

    return df


if __name__ == "__main__":
    df = run_eda()
