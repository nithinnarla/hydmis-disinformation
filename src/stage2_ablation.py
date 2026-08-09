"""
HyDMIS - Stage 2 Ablation
Phase 4 - Stage 2: Quantifying GPT-4 Semantic Verification's Value Over LDA Alone

Answers: does knowing a record's LDA topic (Stage 1) meaningfully predict its
GPT-4-verified veracity label (Stage 2), or does GPT-4 verification add
discriminative signal that topic modeling alone cannot provide?

Method: for each LDA topic, compute the majority GPT-4 label's share of
records in that topic. Average this across topics and compare against the
naive baseline (guessing the single most common label overall, ignoring
topic entirely). A small improvement over baseline means LDA topics do not
meaningfully separate veracity classes - confirming GPT-4 verification is
necessary, not redundant with Stage 1.

Data note: only 11,294 of 14,640 GPT-4-verified records (77.1%) have a real
LDA topic assignment. This gap is expected, not a bug: lda_pipeline.py
subsamples TruthSeeker and DeFaktS to SAMPLE_SIZE=50,000 (out of 134,198 and
105,855 full records respectively) for LDA training speed, while
gpt4_sampler.py draws its own independent stratified sample from the full
datasets. The two samples, drawn independently with random_state=42 from
datasets of different sizes, overlap only partially by chance. LIAR2,
FakeNewsNet, and NewsPolyML LDA ran on their full datasets and show
near-complete (98-100%) match rates; TruthSeeker and DeFaktS, LDA-subsampled,
show ~45% match rates - consistent with this explanation.

Pipeline/infrastructure script - no notebook (no figures; single quantitative
result, matches methodology_decisions.md Decision 11 documentation pattern).
"""

import os
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_PATH = os.path.join(REPO_ROOT, "data", "processed", "gpt4_sample.csv")
VERIFIED_PATH = os.path.join(REPO_ROOT, "data", "processed", "gpt4_verified.csv")
ASSIGNMENTS_PATH = os.path.join(REPO_ROOT, "data", "processed", "lda_topic_assignments.csv")
OUTPUT_PATH = os.path.join(REPO_ROOT, "data", "processed", "gpt4_verified_with_lda.csv")


def build_joined_dataset() -> pd.DataFrame:
    """Join GPT-4 verified labels (via reproducible sample re-generation) to real LDA topics."""
    new_sample = pd.read_csv(SAMPLE_PATH)
    old_verified = pd.read_csv(VERIFIED_PATH)
    assignments = pd.read_csv(ASSIGNMENTS_PATH)

    if len(new_sample) != len(old_verified):
        raise ValueError(
            f"Sample size mismatch: gpt4_sample.csv has {len(new_sample)} rows, "
            f"gpt4_verified.csv has {len(old_verified)}. Cannot safely reattach labels - "
            f"the seed=42 reproducibility this join depends on may be broken."
        )
    if not (new_sample["text"].values == old_verified["text"].values).all():
        raise ValueError(
            "Text mismatch between gpt4_sample.csv and gpt4_verified.csv - "
            "cannot safely reattach GPT-4 labels by row position. Re-verify "
            "seed=42 reproducibility before using this join."
        )

    final = new_sample[["text", "dataset", "language", "veracity", "record_id"]].copy()
    final["gpt4_label"] = old_verified["gpt4_label"].values
    final["gpt4_topic_match"] = old_verified["gpt4_topic_match"].values
    final["gpt4_category"] = old_verified["gpt4_category"].values

    merged = final.merge(
        assignments[["dataset", "record_id", "lda_topic_id"]],
        on=["dataset", "record_id"], how="left"
    )
    return merged


def run_stage2_ablation():
    print("HyDMIS - Stage 2 Ablation: LDA Topic vs GPT-4 Semantic Verification")
    print("=" * 65)

    merged = build_joined_dataset()
    merged.to_csv(OUTPUT_PATH, index=False)
    print(f"  Joined dataset saved: {OUTPUT_PATH}")
    print(f"  Total GPT-4-verified records: {len(merged):,}")

    df = merged[merged["lda_topic_id"].notna()].copy()
    df["lda_topic_id"] = df["lda_topic_id"].astype(int)
    coverage = len(df) / len(merged)
    print(f"  Records with real LDA topic match: {len(df):,} ({coverage:.1%})")
    print(f"  (Gap explained by LDA's SAMPLE_SIZE=50,000 cap on TruthSeeker/DeFaktS - see docstring)")

    print("\n--- Match rate by dataset ---")
    for ds in sorted(merged["dataset"].unique()):
        sub = merged[merged["dataset"] == ds]
        matched = sub["lda_topic_id"].notna().sum()
        print(f"  {ds:<15} {matched:,}/{len(sub):,} ({matched/len(sub):.1%})")

    print("\n--- Cross-tab: LDA topic vs GPT-4 label ---")
    ct = pd.crosstab(df["lda_topic_id"], df["gpt4_label"])
    print(ct)

    ct_pct = ct.div(ct.sum(axis=1), axis=0)
    majority_share = ct_pct.max(axis=1)
    overall_majority = df["gpt4_label"].value_counts(normalize=True).max()
    mean_majority = majority_share.mean()
    improvement = mean_majority - overall_majority

    print("\n--- Ablation Result ---")
    print(f"  Baseline (no topic info, guess most common label overall): {overall_majority:.1%}")
    print(f"  Mean majority-label share WITH topic known: {mean_majority:.1%}")
    print(f"  Improvement from knowing LDA topic: {improvement*100:+.1f} percentage points")
    print()
    if improvement < 0.10:
        print(f"  CONCLUSION: LDA topic is a WEAK predictor of veracity ({improvement*100:+.1f}pp over baseline).")
        print(f"  This confirms GPT-4 semantic verification (Stage 2) adds discriminative value")
        print(f"  that topic modeling alone (Stage 1) cannot provide - consistent with")
        print(f"  lda_topic_clusters.py's qualitative finding: 'Topic-veracity alignment: partial")
        print(f"  - LDA captures content domains not veracity.'")
    else:
        print(f"  CONCLUSION: LDA topic provides meaningful predictive signal ({improvement*100:+.1f}pp).")

    print("\n--- Stage 2 Ablation complete ---")

    return merged, majority_share, overall_majority


if __name__ == "__main__":
    run_stage2_ablation()
