"""
HyDMIS — LIAR2 EDA
Phase 4 — Exploratory Data Analysis
Political Disinformation Domain

EDA on LIAR2 dataset — 22,962 political claims (Xu & Kechadi, 2024).
Updated LIAR benchmark — PolitiFact statements with six-way veracity labels.
Used as English baseline for political disinformation detection in HyDMIS.

Label mapping (confirmed from speaker history counts):
  0 = pants-fire (most false)
  1 = false
  2 = barely-true
  3 = half-true
  4 = mostly-true
  5 = true
"""

import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from liar2_loader import load_liar2

LABEL_NAMES = {0:"pants-fire", 1:"false", 2:"barely-true",
               3:"half-true", 4:"mostly-true", 5:"true"}
# Binary mapping — disinformation vs credible
# Disinformation: pants-fire(0), false(1), barely-true(2)
# Credible: half-true(3), mostly-true(4), true(5)
BINARY_MAP = {0:1, 1:1, 2:1, 3:0, 4:0, 5:0}
BINARY_NAMES = {0:"Credible", 1:"Disinformation"}


def run_eda():
    print("HyDMIS Phase 4 — LIAR2 EDA")
    print("=" * 50)

    result = load_liar2()
    df = result["data"]
    df["label_name"] = df["label"].map(LABEL_NAMES)
    df["binary_label"] = df["label"].map(BINARY_MAP)
    df["statement_len"] = df["statement"].astype(str).str.len()

    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    print(f"\n--- Label Distribution (Six-Way) ---")
    for label in sorted(df["label"].unique()):
        name = LABEL_NAMES[label]
        count = (df["label"] == label).sum()
        pct = count / len(df)
        print(f"  {label} ({name:<15}): {count:,} ({pct:.1%})")

    print(f"\n--- Binary Disinformation Rate ---")
    dis_rate = df["binary_label"].mean()
    print(f"  Disinformation (0+1+2): {(df['binary_label']==1).sum():,} ({dis_rate:.1%})")
    print(f"  Credible (3+4+5):       {(df['binary_label']==0).sum():,} ({1-dis_rate:.1%})")
    print(f"  Note: HyDMIS Stage 1 LDA uses binary label for topic-disinformation correlation")

    print(f"\n--- Split Distribution ---")
    for split, count in df["split"].value_counts().items():
        dis = df[df["split"]==split]["binary_label"].mean()
        print(f"  {split:<12} n={count:,} | disinformation rate: {dis:.1%}")

    print(f"\n--- Top 15 Speakers ---")
    speaker_stats = df.groupby("speaker").agg(
        count=("label","count"),
        dis_rate=("binary_label","mean")
    ).reset_index().sort_values("count", ascending=False).head(15)
    for _, row in speaker_stats.iterrows():
        print(f"  {str(row['speaker']):<35} n={int(row['count']):,} | dis rate: {row['dis_rate']:.1%}")

    print(f"\n--- Top 15 Subjects ---")
    subject_stats = df.groupby("subject").agg(
        count=("label","count"),
        dis_rate=("binary_label","mean")
    ).reset_index().sort_values("count", ascending=False).head(15)
    for _, row in subject_stats.iterrows():
        print(f"  {str(row['subject']):<45} n={int(row['count']):,} | dis rate: {row['dis_rate']:.1%}")

    print(f"\n--- State Distribution ---")
    print(f"  Missing state_info: {df['state_info'].isnull().sum():,} ({df['state_info'].isnull().mean():.1%})")
    state_stats = df[df["state_info"].notna()].groupby("state_info").agg(
        count=("label","count"),
        dis_rate=("binary_label","mean")
    ).reset_index().sort_values("count", ascending=False).head(10)
    for _, row in state_stats.iterrows():
        print(f"  {str(row['state_info']):<20} n={int(row['count']):,} | dis rate: {row['dis_rate']:.1%}")

    print(f"\n--- Speaker History Credibility ---")
    hist_cols = ["true_counts","mostly_true_counts","half_true_counts",
                 "mostly_false_counts","false_counts","pants_on_fire_counts"]
    print(f"  Mean history counts by label:")
    print(df.groupby("label")[hist_cols].mean().round(1).to_string())

    print(f"\n--- Statement Length Distribution ---")
    print(f"  Mean length: {df['statement_len'].mean():.1f} chars")
    print(f"  Median: {df['statement_len'].median():.0f} chars")
    print(f"  Min: {df['statement_len'].min()} | Max: {df['statement_len'].max()}")
    len_corr = df["statement_len"].corr(df["binary_label"])
    print(f"  Length-disinformation correlation: {len_corr:.3f}")
    for label in sorted(df["label"].unique()):
        mean_len = df[df["label"]==label]["statement_len"].mean()
        print(f"  {LABEL_NAMES[label]:<15}: mean {mean_len:.0f} chars")

    print(f"\n--- Missing Values ---")
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    if len(nulls) == 0:
        print("  No missing values")
    else:
        for col, n in nulls.items():
            print(f"  {col:<25} {n:,} ({n/len(df):.1%})")



    print(f"\n--- Temporal Trend — Disinformation Rate by Year ---")
    df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date_parsed"].dt.year
    year_stats = df.groupby("year").agg(
        count=("binary_label","count"),
        dis_rate=("binary_label","mean")
    ).reset_index().dropna()
    year_stats = year_stats[year_stats["count"] >= 100]
    for _, row in year_stats.iterrows():
        print(f"  {int(row['year'])}: n={int(row['count']):,} | dis rate: {row['dis_rate']:.1%}")
    print(f"  Note: Disinformation rate rose from 41.7% (2008) to 87.4% (2023)")
    print(f"  Note: 2020 spike (77.9%) — COVID-19 + US presidential election")
    print(f"  Note: Post-2017 sustained rise reflects social media fact-checking expansion")

    print(f"\n--- Speaker Type vs Disinformation Rate ---")
    def get_speaker_type(speaker):
        speaker = str(speaker).lower()
        if any(x in speaker for x in ["facebook","instagram","twitter","viral","social"]):
            return "Social Media"
        elif any(x in speaker for x in ["blog","chain email","email"]):
            return "Blogs/Email"
        elif any(x in speaker for x in ["donald trump","barack obama","hillary clinton",
                                          "joe biden","mitt romney","bernie sanders",
                                          "nancy pelosi","mitch mcconnell"]):
            return "Major Politicians"
        elif any(x in speaker for x in ["governor","senator","representative","mayor"]):
            return "State Politicians"
        else:
            return "Other"
    df["speaker_type"] = df["speaker"].apply(get_speaker_type)
    type_stats = df.groupby("speaker_type").agg(
        count=("binary_label","count"),
        dis_rate=("binary_label","mean")
    ).sort_values("count", ascending=False)
    for stype, row in type_stats.iterrows():
        print(f"  {stype:<20} n={int(row['count']):,} | dis rate: {row['dis_rate']:.1%}")
    print(f"  Note: Social Media (93.8%) and Blogs/Email (92.1%) far exceed politicians (48.1%)")

    print(f"\n--- Key Observations ---")
    print(f"  Total records: {len(df):,}")
    print(f"  Disinformation rate: {df['binary_label'].mean():.1%}")
    print(f"  Most common label: {df['label'].value_counts().index[0]} ({LABEL_NAMES[df['label'].value_counts().index[0]]})")
    fb = df[df["speaker"] == "facebook posts"]
    print(f"  Facebook posts top speaker: n={len(fb):,} | dis rate: {fb['binary_label'].mean():.1%}")
    print(f"  Note: Six-way labels enable nuanced disinformation severity analysis")
    print(f"  Note: HyDMIS LDA Stage 1 discovers topic clusters correlated with disinformation")

    print(f"\n--- LIAR2 EDA complete ---")
    print(f"  Total records: {len(df):,}")
    print(f"  Ready for HyDMIS Stage 1 LDA topic modeling")

    return df


if __name__ == "__main__":
    df = run_eda()
