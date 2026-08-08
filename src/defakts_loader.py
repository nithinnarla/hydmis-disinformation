"""
HyDMIS, DeFaktS Dataset Loader
Hybrid Disinformation Identification System

DeFaktS, 105,855 German Twitter posts (Ashraf et al., 2024)
Fine-grained disinformation annotations across elections, climate, health.
Primary German-language social media dataset for HyDMIS.
"""

import pandas as pd
import json
import os

DATA_DIR = 'data/raw'


def load_defakts(data_dir: str = DATA_DIR) -> dict:
    """
    Load DeFaktS, German Twitter disinformation.
    Source: Ashraf et al. (2024) LREC-COLING
    105,855 German Twitter/X posts with fine-grained
    disinformation annotations across elections, climate,
    and health topics. BERT-based models show strong results.
    """
    path = os.path.join(data_dir, 'defakts')
    print("Loading DeFaktS...")

    records = []
    with open(f"{path}/DefaktS_Twitter_DS.jsonl", 'r') as f:
        for line in f:
            records.append(json.loads(line.strip()))

    df = pd.DataFrame(records)
    print(f"   DeFaktS: {len(df):,} records | Social Media | German")

    return {
        'name': 'defakts',
        'data': df,
        'n_samples': len(df),
        'language': 'German',
        'domain': 'Social Media',
        'citation': 'Ashraf et al. (2024)'
    }


if __name__ == '__main__':
    result = load_defakts()
    print(f"\nDeFaktS loaded: {result['n_samples']:,} records")
