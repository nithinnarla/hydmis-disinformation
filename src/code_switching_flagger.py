"""
HyDMIS - Code-Switching Detection Flag
Phase 4 - Stage 2: Flag records with possible language mixing

Uses langdetect's confidence score as an imperfect signal for possible
code-switching or ambiguous-language content. Records below a confidence
threshold are flagged for downstream awareness (e.g. in mBERT tokenization
or cross-lingual evaluation), not definitively classified as code-switched.

KNOWN LIMITATION (validated manually before building this script):
langdetect's single-language confidence score is not a reliable code-switching
detector. Tested against two constructed mixed-language examples:
- English/German mix: langdetect scored en=0.9999 (high confidence, single
  language) - completely missed the code-switching.
- English/Spanish mix: langdetect scored fr=0.57 as top guess (wrong language
  entirely) - low confidence correctly signaled "something is off" but did
  not correctly identify either true language present.
This means: a LOW confidence score is a useful (if imperfect) signal that a
record may need manual review or cross-lingual-aware handling downstream.
A HIGH confidence score does NOT guarantee the text is genuinely monolingual --
it can still miss real code-switching, as demonstrated above.

This script is a screening flag, not a definitive code-switching classifier.

Pipeline/infrastructure script - no notebook.
"""

import os
import warnings
import pandas as pd
from langdetect import detect_langs, DetectorFactory, LangDetectException
warnings.filterwarnings("ignore")

DetectorFactory.seed = 42

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(REPO_ROOT, "data", "processed", "gpt4_verified.csv")
OUTPUT_PATH = os.path.join(REPO_ROOT, "data", "processed", "code_switching_flags.csv")
CONFIDENCE_THRESHOLD = 0.90


def detect_confidence(text: str) -> dict:
    """Return top-language and confidence, or None values on detection failure."""
    try:
        result = detect_langs(str(text))
        return {
            "detected_top_lang": result[0].lang,
            "detected_confidence": round(result[0].prob, 4),
            "detection_error": None,
        }
    except LangDetectException as e:
        return {
            "detected_top_lang": None,
            "detected_confidence": None,
            "detection_error": str(e),
        }


def run_code_switching_flagger():
    print("HyDMIS - Code-Switching Detection Flag")
    print("=" * 50)
    print(f"  Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"  Known limitation: confidence-based detection can miss real")
    print(f"  code-switching (validated on constructed examples - see docstring)")
    print()

    df = pd.read_csv(INPUT_PATH)
    print(f"  Loaded {len(df):,} records from {INPUT_PATH}")

    results = []
    for i, text in enumerate(df["text"]):
        r = detect_confidence(text)
        results.append(r)
        if (i + 1) % 2000 == 0:
            print(f"  Progress: {i+1:,}/{len(df):,}")

    flag_df = pd.DataFrame(results)
    df = pd.concat([df.reset_index(drop=True), flag_df], axis=1)

    df["possible_code_switch"] = (
        (df["detected_confidence"].notna())
        & (df["detected_confidence"] < CONFIDENCE_THRESHOLD)
    )
    df["lang_mismatch"] = (
        (df["detected_top_lang"].notna())
        & (df["detected_top_lang"] != df["language"])
    )

    df.to_csv(OUTPUT_PATH, index=False)

    n_flagged = df["possible_code_switch"].sum()
    n_mismatch = df["lang_mismatch"].sum()
    n_errors = df["detection_error"].notna().sum()

    print()
    print("--- Results ---")
    print(f"  Total records: {len(df):,}")
    print(f"  Flagged as possible code-switch (<{CONFIDENCE_THRESHOLD} confidence): {n_flagged:,} ({n_flagged/len(df)*100:.1f}%)")
    print(f"  Detected language differs from labeled language: {n_mismatch:,} ({n_mismatch/len(df)*100:.1f}%)")
    print(f"  Detection errors (empty/invalid text): {n_errors:,}")
    print(f"  Saved: {OUTPUT_PATH}")
    print()
    print("--- Code-Switching Flag complete ---")
    print("  This is a screening flag for downstream awareness, not a")
    print("  definitive code-switching classification - see known limitation")
    print("  documented in this script's docstring and methodology_decisions.md")

    return df


if __name__ == "__main__":
    run_code_switching_flagger()
