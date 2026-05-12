"""
HyDMIS — TruthSeeker Dataset Loader
Hybrid Disinformation Identification System

TruthSeeker — 134,198 social media posts (Dadkhah et al., 2023)
Largest labeled social media fake news dataset in existence.
Primary English social media benchmark for HyDMIS.
"""

import pandas as pd
import os

DATA_DIR = 'data/raw'


def load_truthseeker(data_dir: str = DATA_DIR) -> dict:
    """
    Load TruthSeeker dataset — social media fake news.
    Source: Dadkhah et al. (2023) IEEE TCSS
    134,198 Twitter/X posts — largest labeled social media
    fake news dataset in existence. Primary English benchmark.
    """
    path = os.path.join(data_dir, 'truthseeker')
    print("Loading TruthSeeker...")

    df = pd.read_csv(
        f"{path}/Truth_Seeker_Model_Dataset.csv",
        low_memory=False
    )
    print(f"  ✓ TruthSeeker: {len(df):,} records | Social Media | English")

    return {
        'name': 'truthseeker',
        'data': df,
        'n_samples': len(df),
        'language': 'English',
        'domain': 'Social Media',
        'citation': 'Dadkhah et al. (2023)'
    }


if __name__ == '__main__':
    result = load_truthseeker()
    print(f"\nTruthSeeker loaded: {result['n_samples']:,} records")
