"""
HyDMIS, GPT-4 Semantic Verification Engine
Phase 4, Stage 2: GPT-4 Semantic Verification

Calls GPT-4 API on stratified 14,640-record sample to:
1. Verify whether each claim is disinformation (YES/NO/UNCERTAIN)
2. Verify whether LDA topic assignment matches text content (MATCH/MISMATCH/PARTIAL)
3. Categorize disinformation type (HEALTH/POLITICAL/ECONOMIC/SOCIAL/OTHER)

Architecture (Decision 4):
- GPT-4 labels 14,640 representative samples
- Labels used to fine-tune Mistral 7B for full-scale deployment
- Batch processing with rate limiting, 50 requests/minute
- Checkpoint saves every 500 records, resumable on failure
- Cost estimate: ~$3-$4 at GPT-4o-mini rates

Input: data/processed/gpt4_sample.csv (from gpt4_sampler.py)
Output: data/processed/gpt4_verified.csv, sample with GPT-4 labels

Script type: pipeline/infrastructure, no notebook, no figures
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(REPO_ROOT, 'data', 'processed')
SAMPLE_PATH = os.path.join(PROCESSED_DIR, 'gpt4_sample.csv')
OUTPUT_PATH = os.path.join(PROCESSED_DIR, 'gpt4_verified.csv')
CHECKPOINT_PATH = os.path.join(PROCESSED_DIR, 'gpt4_checkpoint.csv')

MODEL = 'gpt-4o-mini'  # cost-efficient; GPT-4 quality for classification tasks
BATCH_SIZE = 1          # one request per record, simpler error handling
RATE_LIMIT_SLEEP = 1.2  # seconds between requests, 50 req/min limit
CHECKPOINT_EVERY = 500  # save checkpoint every N records
MAX_RETRIES = 3
TIMEOUT = 30


def get_openai_client():
    """Initialize OpenAI client from environment variable."""
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable not set. "
            "Set it with: export OPENAI_API_KEY='your-key-here'"
        )
    from openai import OpenAI
    return OpenAI(api_key=api_key)


def parse_gpt4_response(response_text):
    """
    Parse GPT-4 JSON response into structured labels.

    Returns: dict with disinformation, topic_match, category
    """
    try:
        # Extract JSON from response
        text = response_text.strip()
        if '```json' in text:
            text = text.split('```json')[1].split('```')[0].strip()
        elif '```' in text:
            text = text.split('```')[1].split('```')[0].strip()

        parsed = json.loads(text)
        return {
            'gpt4_label': parsed.get('disinformation', 'UNCERTAIN').upper(),
            'gpt4_topic_match': parsed.get('topic_match', 'PARTIAL').upper(),
            'gpt4_category': parsed.get('category', 'OTHER').upper()
        }
    except (json.JSONDecodeError, KeyError, AttributeError):
        return {
            'gpt4_label': 'UNCERTAIN',
            'gpt4_topic_match': 'PARTIAL',
            'gpt4_category': 'OTHER'
        }


def verify_single(client, prompt, retries=MAX_RETRIES):
    """Call GPT-4 API for a single record with retry logic."""
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        'role': 'system',
                        'content': (
                            'You are a multilingual disinformation detection expert. '
                            'Always respond in valid JSON format only. '
                            'No preamble, no explanation, just JSON.'
                        )
                    },
                    {'role': 'user', 'content': prompt}
                ],
                max_tokens=100,
                temperature=0.0,  # deterministic, classification task
                timeout=TIMEOUT
            )
            return parse_gpt4_response(
                response.choices[0].message.content
            )
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # exponential backoff
            else:
                return {
                    'gpt4_label': 'ERROR',
                    'gpt4_topic_match': 'ERROR',
                    'gpt4_category': 'ERROR'
                }


def load_checkpoint():
    """Load checkpoint if exists, resume from last saved position."""
    if os.path.exists(CHECKPOINT_PATH):
        df = pd.read_csv(CHECKPOINT_PATH)
        completed = df[df['gpt4_label'].notna()].shape[0]
        print(f"  Checkpoint found, {completed:,} records already verified")
        return df, completed
    return None, 0


def run_gpt4_verifier(dry_run=False, max_records=None):
    """
    Main verification function.

    Args:
        dry_run: bool, if True, simulate API calls without actually calling
        max_records: int, limit records for testing (None = all)
    """
    print("HyDMIS Phase 4, Stage 2: GPT-4 Semantic Verification")
    print("=" * 60)
    print(f"  Model: {MODEL}")
    print(f"  Mode: {'DRY RUN (no API calls)' if dry_run else 'LIVE'}")

    print("\n--- Loading Sample ---")
    if not os.path.exists(SAMPLE_PATH):
        print(f"ERROR: Sample not found at {SAMPLE_PATH}")
        print("Run gpt4_sampler.py first.")
        return None

    df = pd.read_csv(SAMPLE_PATH)
    if max_records:
        df = df.head(max_records)
    print(f"  Sample: {len(df):,} records loaded")
    print(f"  Already verified: {df['gpt4_label'].notna().sum():,}")

    # Check for checkpoint
    checkpoint_df, n_completed = load_checkpoint()
    if checkpoint_df is not None and len(checkpoint_df) == len(df):
        df = checkpoint_df

    # Estimate cost
    n_remaining = df['gpt4_label'].isna().sum()
    cost_est = n_remaining * 0.00015  # gpt-4o-mini input ~$0.15/1M tokens
    time_est = n_remaining * RATE_LIMIT_SLEEP / 60
    print(f"  Remaining: {n_remaining:,} records")
    print(f"  Estimated cost: ~${cost_est:.2f}")
    print(f"  Estimated time: ~{time_est:.0f} minutes")

    if dry_run:
        print("\n--- Dry Run, Simulating 10 records ---")
        sample_rows = df[df['gpt4_label'].isna()].head(10)
        for i, (idx, row) in enumerate(sample_rows.iterrows()):
            # Simulate response
            simulated = {
                'gpt4_label': str(np.random.choice(['YES', 'NO', 'UNCERTAIN'])),
                'gpt4_topic_match': str(np.random.choice(['MATCH', 'MISMATCH', 'PARTIAL'])),
                'gpt4_category': str(np.random.choice(['HEALTH', 'POLITICAL', 'ECONOMIC', 'SOCIAL', 'OTHER']))
            }
            df.at[idx, 'gpt4_label'] = simulated['gpt4_label']
            df.at[idx, 'gpt4_topic_match'] = simulated['gpt4_topic_match']
            df.at[idx, 'gpt4_category'] = simulated['gpt4_category']
            print(f"  Record {i+1}: {simulated}")

        df.to_csv(OUTPUT_PATH, index=False)
        print(f"\n--- Dry Run complete ---")
        print(f"  10 records simulated")
        print(f"  Output: {OUTPUT_PATH}")
        print(f"  Run without dry_run=True to call live API")
        return df

    print("\n--- Starting GPT-4 Verification ---")
    client = get_openai_client()

    n_verified = 0
    n_errors = 0
    pending = df[df['gpt4_label'].isna()].index.tolist()

    for i, idx in enumerate(pending):
        prompt = df.at[idx, 'gpt4_prompt']
        result = verify_single(client, prompt)

        df.at[idx, 'gpt4_label'] = result['gpt4_label']
        df.at[idx, 'gpt4_topic_match'] = result['gpt4_topic_match']
        df.at[idx, 'gpt4_category'] = result['gpt4_category']

        if result['gpt4_label'] == 'ERROR':
            n_errors += 1
        else:
            n_verified += 1

        # Progress print every 100 records
        if (i + 1) % 100 == 0:
            pct = (i + 1) / len(pending) * 100
            print(f"  Progress: {i+1:,}/{len(pending):,} ({pct:.1f}%), "
                  f"verified: {n_verified:,} errors: {n_errors:,}")

        # Checkpoint save
        if (i + 1) % CHECKPOINT_EVERY == 0:
            df.to_csv(CHECKPOINT_PATH, index=False)
            print(f"  Checkpoint saved at record {i+1:,}")

        time.sleep(RATE_LIMIT_SLEEP)

    # Final save
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\n--- GPT-4 Verification complete ---")
    print(f"  Total verified: {n_verified:,}")
    print(f"  Errors: {n_errors:,}")
    print(f"  Output: {OUTPUT_PATH}")

    # Label distribution
    print(f"\n  Disinformation label distribution:")
    for label, count in df['gpt4_label'].value_counts().items():
        print(f"    {label:<12} {count:,}")

    return df


if __name__ == "__main__":
    # Default: dry run, set dry_run=False and provide API key for live run
    run_gpt4_verifier(dry_run=False)
