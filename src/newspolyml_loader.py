"""
HyDMIS, NewsPolyML Dataset Loader
Hybrid Disinformation Identification System

NewsPolyML v2, 32,129 European multilingual news (Mohtaj et al., 2024)
IFCN-certified fact-checked claims across EN/DE/ES/FR/IT.
Bridges English and European low-resource evaluation.
"""

import pandas as pd
import os

DATA_DIR = 'data/raw'


def load_newspolyml(data_dir: str = DATA_DIR) -> dict:
    """
    Load NewsPolyML v2, European multilingual news.
    Source: Mohtaj et al. (2024) ACM MAD Workshop
    32,129 IFCN-certified fact-checked news claims across
    English, German, Spanish, French, Italian.
    Bridges English and European low-resource evaluation.
    """
    path = os.path.join(data_dir, 'newspolyml')
    print("Loading NewsPolyML...")

    df = pd.read_csv(
        f"{path}/NewsPolyML_v2.csv",
        low_memory=False
    )
    print(f"   NewsPolyML: {len(df):,} records | News | EN/DE/ES/FR/IT")

    return {
        'name': 'newspolyml',
        'data': df,
        'n_samples': len(df),
        'language': 'English, German, Spanish, French, Italian',
        'domain': 'News/Politics',
        'citation': 'Mohtaj et al. (2024)'
    }


if __name__ == '__main__':
    result = load_newspolyml()
    print(f"\nNewsPolyML loaded: {result['n_samples']:,} records")
