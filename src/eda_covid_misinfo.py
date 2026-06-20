"""
HyDMIS — Covid-vaccine-misinfo-MIC EDA
Phase 4 — Exploratory Data Analysis
Health Misinformation Domain

EDA on Covid-vaccine-misinfo-MIC dataset — 5,952 annotated tweets (Kim et al., 2023 EMNLP).
Multilingual health misinformation across Brazil (PT), Indonesia (ID), Nigeria (EN).
Primary health domain multilingual dataset for HyDMIS.

Critical limitation: No tweet text available — tweets deleted from Twitter.
EDA covers annotation metadata, country/language distribution, label patterns.
Dataset used for cross-lingual label analysis only, not Stage 1 LDA text modeling.

Label mapping: Q1=1 (Misinformation), Q1=2 (Not Misinformation)
Dis rate: 61.6% — majority misinfo dataset
Missing country (n=588): 100% misinfo — strong metadata signal
"""

import pandas as pd
import numpy as np
import sys
import os
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from covid_misinfo_loader import load_covid_misinfo


def run_eda():
    print("HyDMIS Phase 4 — Covid-vaccine-misinfo-MIC EDA")
    print("=" * 55)

    result = load_covid_misinfo()
    df = result['data']
    df['is_misinfo'] = (df['Q1'] == 1).astype(int)

    print(f"\nDataset shape: {df.shape}")
    print(f"Note: No tweet text — tweets deleted from Twitter")
    print(f"EDA covers annotation metadata and distribution analysis only")

    print(f"\n--- Label Distribution ---")
    for label, name in [(1,'Misinformation'),(2,'Not Misinformation')]:
        count = (df['Q1']==label).sum()
        print(f"  {name:<20} {count:,} ({count/len(df):.1%})")
    print(f"  Dis rate: {df['is_misinfo'].mean():.1%} — majority misinfo dataset")

    print(f"\n--- Country Distribution ---")
    for country, count in df['country'].value_counts().items():
        dis = df[df['country']==country]['is_misinfo'].mean()
        print(f"  {str(country):<15} n={count:,} ({count/len(df):.1%}) | dis_rate={dis:.1%}")
    missing_n = df['country'].isna().sum()
    missing_dis = df[df['country'].isna()]['is_misinfo'].mean()
    print(f"  {'Missing':<15} n={missing_n:,} ({missing_n/len(df):.1%}) | dis_rate={missing_dis:.1%}")
    print(f"  Note: Missing country = 100% misinfo — strong metadata signal")

    print(f"\n--- Language Distribution ---")
    for lang, count in df['lang'].value_counts().head(8).items():
        dis = df[df['lang']==lang]['is_misinfo'].mean()
        print(f"  {str(lang):<8} n={count:,} ({count/len(df):.1%}) | dis_rate={dis:.1%}")


    print(f"\n--- Disinformation Rate by Language ---")
    lang_dis = df.groupby('lang')['is_misinfo'].agg(['mean','count']).reset_index()
    lang_dis = lang_dis[lang_dis['count'] >= 10].sort_values('mean', ascending=False)
    for _, row in lang_dis.iterrows():
        print(f"  {row['lang']:<8} dis_rate={row['mean']:.1%} n={int(row['count']):,}")
    print(f"  Note: ID (Indonesian) highest 81.7%, PT (Portuguese) lowest 34.5%")

    print(f"\n--- Disinformation Rate by Country ---")
    ct = pd.crosstab(df['country'], df['Q1'])
    ct.columns = ['Misinfo','Not_Misinfo']
    ct['dis_rate'] = ct['Misinfo']/(ct['Misinfo']+ct['Not_Misinfo'])
    for country, row in ct.iterrows():
        print(f"  {str(country):<15} Misinfo={row['Misinfo']:,} NotMisinfo={row['Not_Misinfo']:,} DIR={row['dis_rate']:.1%}")

    print(f"\n--- Annotation Agreement (Q2) ---")
    q2_sub = df[df['Q2'].notna()]
    print(f"  Q2 available: {len(q2_sub):,} records")
    ct2 = pd.crosstab(q2_sub['Q1'], q2_sub['Q2'])
    print(f"  Q1=1 (Misinfo): Q2=1.0={ct2.loc[1,1.0]:,} Q2=2.0={ct2.loc[1,2.0]:,}")
    print(f"  Q2=1.0 (agree misinfo): {ct2.loc[1,1.0]:,} ({ct2.loc[1,1.0]/len(q2_sub):.1%})")
    print(f"  Q2=2.0 (disagree): {ct2.loc[1,2.0]:,} ({ct2.loc[1,2.0]/len(q2_sub):.1%})")
    print(f"  Note: 82.1% annotator disagreement on misinfo records — high ambiguity")

    print(f"\n--- Missing Values ---")
    nulls = df.isnull().sum()
    for col, n in nulls[nulls>0].items():
        print(f"  {col:<12} {n:,} ({n/len(df):.1%})")

    print(f"\n--- Key Observations ---")
    print(f"  Total records: {len(df):,}")
    print(f"  Dis rate: {df['is_misinfo'].mean():.1%} — majority misinfo")
    print(f"  No tweet text — annotation metadata only")
    print(f"  Indonesia highest dis rate: {df[df['country']=='Indonesia']['is_misinfo'].mean():.1%}")
    print(f"  Nigeria lowest dis rate: {df[df['country']=='Nigeria']['is_misinfo'].mean():.1%}")
    print(f"  Missing country = 100% misinfo — metadata signal for Stage 2")
    print(f"  Indonesian lang dis rate: {df[df['lang']=='id']['is_misinfo'].mean():.1%} — highest by language")
    print(f"  HyDMIS use: Cross-lingual label analysis only — no text for Stage 1 LDA")

    print(f"\n--- Covid-misinfo-MIC EDA complete ---")
    print(f"  Ready for HyDMIS cross-lingual evaluation framework")

    return df


if __name__ == "__main__":
    df = run_eda()
