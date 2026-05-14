"""
HyDMIS — Covid-vaccine-misinfo-MIC Dataset Loader
Hybrid Disinformation Identification System

Covid-vaccine-misinfo-MIC — 5,952 multilingual tweets (Kim et al., 2023)
Health misinformation across Brazil (PT), Indonesia (ID), Nigeria (EN).
Primary health domain multilingual dataset for HyDMIS.
"""

import pandas as pd
import os

DATA_DIR = 'data/raw'


def load_covid_misinfo(data_dir: str = DATA_DIR) -> dict:
    """
    Load Covid-vaccine-misinfo-MIC — multilingual health misinformation.
    Source: Kim et al. (2023) EMNLP
    5,952 annotated tweets across Brazil (PT), Indonesia (ID), Nigeria (EN).
    Primary health domain multilingual dataset for HyDMIS.
    Covers exactly the low-resource community languages HyDMIS targets.
    """
    path = os.path.join(data_dir, 'covid_misinfo')
    print("Loading Covid-vaccine-misinfo-MIC...")

    df = pd.read_csv(
        f"{path}/annotated_data.csv",
        low_memory=False
    )
    print(f"  ✓ Covid-misinfo-MIC: {len(df):,} records | Health | EN/PT/ID")

    return {
        'name': 'covid_misinfo_mic',
        'data': df,
        'n_samples': len(df),
        'language': 'English, Portuguese, Indonesian',
        'domain': 'Health',
        'citation': 'Kim et al. (2023)'
    }


if __name__ == '__main__':
    result = load_covid_misinfo()
    print(f"\nCovid-misinfo-MIC loaded: {result['n_samples']:,} records")
