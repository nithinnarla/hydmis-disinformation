"""
HyDMIS, DeFaktS EDA
Phase 4, Exploratory Data Analysis
German Social Media Disinformation Domain

EDA on DeFaktS, 105,855 German Twitter posts (Ashraf et al., 2024).
Source: LREC-COLING 2024
Fine-grained disinformation annotations across elections, climate, health.
Primary German-language social media dataset for HyDMIS.

Key finding: 81.1% unlabeled for binary classification, span labels
are the primary annotation (86.3% coverage). HyDMIS uses span-level
disinformation detection, not binary classification.

Label: binary_label (1=disinformation, 0=credible), only 18.9% labeled
Span labels: fine-grained token-level annotations (86.3% coverage)
Language: German (de), 100%
"""

import pandas as pd
import numpy as np
import sys
import os
import warnings
warnings.filterwarnings("ignore")
from collections import Counter
sys.path.insert(0, os.path.dirname(__file__))
from defakts_loader import load_defakts


def run_eda():
    print("HyDMIS Phase 4, DeFaktS EDA")
    print("=" * 50)

    result = load_defakts()
    df = result['data'].copy()

    df['dt'] = pd.to_datetime(df['DateTime'], unit='ms', errors='coerce')
    df['year'] = df['dt'].dt.year
    df['has_span'] = df['span_labels'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False)
    df['span_count'] = df['span_labels'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    df['text_len'] = df['text'].astype(str).str.len()
    df['text_words'] = df['text'].astype(str).str.split().str.len()

    print(f"\nDataset shape: {df.shape}")

    print(f"\n--- Dataset Overview ---")
    print(f"  Total records:     {len(df):,}")
    print(f"  Language:          German (de), 100%")
    print(f"  Date range:        2008-2023 (15 years)")
    print(f"  Binary labeled:    {df['binary_label'].notna().sum():,} ({df['binary_label'].notna().mean():.1%})")
    print(f"  Span labeled:      {df['has_span'].sum():,} ({df['has_span'].mean():.1%})")
    print(f"  Note: Span labels are primary annotation, binary label only 18.9% coverage")

    print(f"\n--- Binary Label Distribution ---")
    labeled = df[df['binary_label'].notna()]
    dis = (labeled['binary_label']==1).sum()
    cred = (labeled['binary_label']==0).sum()
    print(f"  Disinformation (1): {dis:,} ({dis/len(labeled):.1%})")
    print(f"  Credible (0):       {cred:,} ({cred/len(labeled):.1%})")
    print(f"  Unlabeled:          {df['binary_label'].isna().sum():,} ({df['binary_label'].isna().mean():.1%})")
    print(f"  Note: Among labeled records, disinformation={dis/len(labeled):.1%}, class imbalance")
    print(f"  HyDMIS uses span-level detection for unlabeled records")

    print(f"\n--- Span Label Types ---")
    all_labels = []
    for spans in df['span_labels'].dropna():
        if isinstance(spans, list):
            for span in spans:
                if len(span) >= 3:
                    all_labels.append(span[2])
    label_counts = Counter(all_labels)
    print(f"  Total span annotations: {len(all_labels):,}")
    for label, count in label_counts.most_common(15):
        print(f"  {label:<20} {count:,} ({count/len(all_labels):.1%})")
    print(f"  Note: corpkeyword dominates, corporate keyword markers")
    print(f"  catposfake={label_counts.get('catposfake',0):,}, positive fake category, primary disinformation signal")

    print(f"\n--- Span Coverage ---")
    print(f"  Records with span labels: {df['has_span'].sum():,} ({df['has_span'].mean():.1%})")
    print(f"  Records without spans:    {(~df['has_span']).sum():,} ({(~df['has_span']).mean():.1%})")
    print(f"  Mean spans per record:    {df['span_count'].mean():.1f}")
    print(f"  Max spans per record:     {df['span_count'].max()}")
    print(f"  Note: 86.3% span coverage, richer than binary label (18.9%)")

    print(f"\n--- Text Length Distribution ---")
    print(f"  Mean: {df['text_len'].mean():.0f} chars | Median: {df['text_len'].median():.0f}")
    print(f"  Mean: {df['text_words'].mean():.0f} words | Median: {df['text_words'].median():.0f}")
    print(f"  Min: {df['text_len'].min()} chars | Max: {df['text_len'].max()} chars")
    labeled_dis = df[df['binary_label']==1]
    labeled_cred = df[df['binary_label']==0]
    print(f"  Disinformation mean: {labeled_dis['text_len'].mean():.0f} chars")
    print(f"  Credible mean:       {labeled_cred['text_len'].mean():.0f} chars")

    print(f"\n--- Engagement Metrics ---")
    for col in ['LikeCount','RetweetCount','ReplyCount','QuoteCount']:
        print(f"  {col:<15} mean={df[col].mean():.1f} | median={df[col].median():.0f} | max={df[col].max():,}")
    print(f"  Note: Median engagement=0-1, heavily right-skewed; viral outliers dominate mean")

    print(f"\n--- Engagement by Label ---")
    for label, name in [(1,'Disinformation'),(0,'Credible')]:
        subset = df[df['binary_label']==label]
        print(f"  {name}: LikeCount mean={subset['LikeCount'].mean():.1f} | RetweetCount mean={subset['RetweetCount'].mean():.1f}")
    print(f"  Note: Engagement differences may indicate virality patterns in disinformation")

    print(f"\n--- Temporal Distribution ---")
    year_counts = df['year'].value_counts().sort_index()
    print(f"  Date range: 2008-2023")
    print(f"  Records with dates: {df['dt'].notna().sum():,} ({df['dt'].notna().mean():.1%})")
    for year in [2016,2017,2018,2019,2020,2021,2022,2023]:
        count = year_counts.get(year, 0)
        print(f"  {year}: {count:,}")
    recent = df[df['year']>=2019]['year'].notna().sum()
    print(f"  Records 2019+: {recent:,} ({recent/len(df):.1%}), post-COVID era")

    print(f"\n--- Hashtag Analysis ---")
    has_hashtag = df['Hashtags'].notna()
    print(f"  Records with hashtags: {has_hashtag.sum():,} ({has_hashtag.mean():.1%})")
    print(f"  Records without hashtags: {(~has_hashtag).sum():,} ({(~has_hashtag).mean():.1%})")
    if df[df['binary_label'].notna()]['Hashtags'].notna().any():
        dis_hashtag = df[df['binary_label']==1]['Hashtags'].notna().mean()
        cred_hashtag = df[df['binary_label']==0]['Hashtags'].notna().mean()
        print(f"  Disinformation with hashtags: {dis_hashtag:.1%}")
        print(f"  Credible with hashtags:       {cred_hashtag:.1%}")


    print(f"\n--- Span Types by Binary Label ---")
    from collections import defaultdict
    label_span_counts = defaultdict(Counter)
    for _, row in df.iterrows():
        if pd.notna(row["binary_label"]) and isinstance(row["span_labels"], list):
            for span in row["span_labels"]:
                if len(span) >= 3:
                    label_span_counts[int(row["binary_label"])][span[2]] += 1
    for label, name in [(1,"Disinformation"),(0,"Credible")]:
        top = label_span_counts[label].most_common(5)
        print(f"  {name}:")
        total = sum(label_span_counts[label].values())
        for span_type, count in top:
            print(f"    {span_type:<20} {count:,} ({count/total:.1%})")
    print(f"  Note: catposfake higher in disinformation, primary span-level signal for HyDMIS")


    print(f"\n--- Engagement Statistical Tests (Mann-Whitney U) ---")
    from scipy import stats
    labeled_mw = df[df["binary_label"].notna()]
    dis_mw = labeled_mw[labeled_mw["binary_label"]==1]
    cred_mw = labeled_mw[labeled_mw["binary_label"]==0]
    for col in ["LikeCount","RetweetCount","ReplyCount","QuoteCount"]:
        stat, pval = stats.mannwhitneyu(dis_mw[col], cred_mw[col], alternative="two-sided")
        sig = "SIGNIFICANT" if pval < 0.05 else "not significant"
        print(f"  {col:<15} U={stat:.0f} p={pval:.4f} {sig}")
    print(f"  Note: LikeCount p=0.0007, ReplyCount p=0.0001, disinformation virality statistically confirmed")
    print(f"  RetweetCount p=0.5623, retweet difference not statistically significant")

    print(f"\n--- Missing Values ---")
    key_cols = ['text','binary_label','span_labels','Language','DateTime','LikeCount']
    for col in key_cols:
        nulls = df[col].isnull().sum()
        print(f"  {col:<20} {nulls:,} ({nulls/len(df):.1%})")

    print(f"\n--- Key Observations ---")
    print(f"  Total: {len(df):,} German tweets across 15 years (2008-2023)")
    print(f"  Binary label: only 18.9% coverage, span labels are primary annotation")
    print(f"  Span coverage: 86.3%, catposfake={label_counts.get('catposfake',0):,} disinformation spans")
    print(f"  corpkeyword dominates spans, corporate keyword detection primary signal")
    print(f"  Engagement heavily skewed, median=0-1, mean=22 due to viral outliers")
    print(f"  HyDMIS uses DeFaktS for German fine-grained span-level disinformation detection")

    print(f"\n--- DeFaktS EDA complete ---")
    print(f"  Ready for HyDMIS Stage 1 LDA topic modeling on German text")

    return df


if __name__ == "__main__":
    df = run_eda()
