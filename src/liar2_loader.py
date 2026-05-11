"""
HyDMIS — LIAR2 Dataset Loader
Hybrid Disinformation Identification System

LIAR2 — 22,962 political claims (Xu & Kechadi, 2024)
Updated LIAR benchmark — PolitiFact statements with six-way veracity labels.
Used as English baseline for political disinformation detection.
"""

import pandas as pd
import os

DATA_DIR = 'data/raw'


def load_liar2(data_dir: str = DATA_DIR) -> dict:
    """
    Load LIAR2 dataset — political claim fact-checking.
    Source: Xu & Kechadi (2024) — updated LIAR benchmark
    22,962 PolitiFact statements with six-way veracity labels.
    Used as English baseline for political disinformation detection.
    HyDMIS uses LIAR2 (not original LIAR) — larger and more recent.
    """
    path = os.path.join(data_dir, 'liar')
    print("Loading LIAR2...")

    dfs = []
    for split in ['train', 'test', 'validation']:
        filepath = f"{path}/{split}.csv"
        if os.path.exists(filepath):
            df = pd.read_csv(filepath, low_memory=False)
            df['split'] = split
            dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"  ✓ LIAR2: {len(combined):,} records | Political | English")

    return {
        'name': 'liar2',
        'data': combined,
        'n_samples': len(combined),
        'language': 'English',
        'domain': 'Political',
        'citation': 'Xu & Kechadi (2024)'
    }


if __name__ == '__main__':
    result = load_liar2()
    print(f"\nLIAR2 loaded: {result['n_samples']:,} records")
