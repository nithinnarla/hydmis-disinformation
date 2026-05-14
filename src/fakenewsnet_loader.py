"""
HyDMIS — FakeNewsNet Dataset Loader
Hybrid Disinformation Identification System

FakeNewsNet — 23,196 news articles (Shu et al., 2020)
PolitiFact and GossipCop political and celebrity news.
Content features only — English baseline news dataset.
"""

import pandas as pd
import os

DATA_DIR = 'data/raw'


def load_fakenewsnet(data_dir: str = DATA_DIR) -> dict:
    """
    Load FakeNewsNet — political and celebrity news.
    Source: Shu et al. (2020) Big Data
    23,196 records across PolitiFact and GossipCop sources.
    Content features only — social graph not available for
    low-resource community content so excluded by design.
    """
    path = os.path.join(data_dir, 'fakenewsnet')
    print("Loading FakeNewsNet...")

    dfs = []
    for f, label in [
        ('politifact_fake.csv', 1),
        ('politifact_real.csv', 0),
        ('gossipcop_fake.csv', 1),
        ('gossipcop_real.csv', 0)
    ]:
        df = pd.read_csv(f"{path}/{f}")
        df['label'] = label
        df['source'] = f.replace('.csv', '')
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"  ✓ FakeNewsNet: {len(combined):,} records | News | English")

    return {
        'name': 'fakenewsnet',
        'data': combined,
        'n_samples': len(combined),
        'language': 'English',
        'domain': 'News',
        'citation': 'Shu et al. (2020)'
    }


if __name__ == '__main__':
    result = load_fakenewsnet()
    print(f"\nFakeNewsNet loaded: {result['n_samples']:,} records")
