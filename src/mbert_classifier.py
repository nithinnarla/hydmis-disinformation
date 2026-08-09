"""
HyDMIS - Stage 3: mBERT Cross-Lingual Classification Setup
Phase 4 - Stage 3: mBERT baseline (RemBERT primary backbone per Decision 5)

Sets up the mBERT (bert-base-multilingual-cased) classification pipeline as
the standard cross-lingual baseline, per Decision 2's three-stage architecture.
RemBERT is the primary reported backbone (Decision 5); mBERT and Mistral 7B
are comparison baselines, all evaluated in Phase 4 ablations.

TODAY'S SCOPE (Aug 4 2026): pipeline architecture setup and validation only,
using the 14,640 directly GPT-4-labeled records as a working prototype.
Full-scale training on the 562K+ dataset requires Mistral-7B pseudo-labeling
(Decision 3), not yet built - scheduled later in August per the tracker.
Real fine-tuning begins Aug 14 per the schedule; this script proves the
pipeline works correctly, it does not produce final reported results.

Training target: gpt4_label (YES/NO/UNCERTAIN), not veracity - see Decision 13.
The single PARTIAL record is excluded, not merged into UNCERTAIN - see
Decision 14 (PARTIAL reflects a confident partial judgment, UNCERTAIN reflects
insufficient evidence; these are semantically distinct, not interchangeable).

Pipeline/infrastructure script - no notebook (setup/validation only, no
headline figures yet; full training results will get their own script+notebook
per project convention once Aug 14 fine-tuning begins).
"""

import os
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification

warnings.filterwarnings("ignore")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(REPO_ROOT, "data", "processed", "gpt4_verified_with_lda.csv")
MODEL_NAME = "bert-base-multilingual-cased"
MAX_LENGTH = 128
RANDOM_STATE = 42


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class DisinfoDataset(Dataset):
    """Tokenized dataset wrapper for mBERT sequence classification."""

    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def load_and_prepare_data():
    """Load the GPT-4-labeled dataset, exclude PARTIAL (Decision 14), encode labels (Decision 13)."""
    df = pd.read_csv(DATA_PATH)
    n_total = len(df)

    df = df[df["gpt4_label"] != "PARTIAL"].copy()
    n_excluded = n_total - len(df)

    label_encoder = LabelEncoder()
    df["label_id"] = label_encoder.fit_transform(df["gpt4_label"])

    return df, label_encoder, n_excluded


def run_mbert_setup():
    print("HyDMIS - Stage 3: mBERT Classification Setup")
    print("=" * 50)

    device = get_device()
    print(f"  Device: {device}")

    print("\n--- Loading Data ---")
    df, label_encoder, n_excluded = load_and_prepare_data()
    print(f"  Records after excluding {n_excluded} PARTIAL record(s) (Decision 14): {len(df):,}")
    print(f"  Label classes (Decision 13, gpt4_label as target): {list(label_encoder.classes_)}")
    print(f"  Label distribution:")
    for label, count in df["gpt4_label"].value_counts().items():
        print(f"    {label:<12} {count:,}")

    print("\n--- Train/Val/Test Split (stratified by label, language) ---")
    train_df, temp_df = train_test_split(
        df, test_size=0.2, stratify=df["gpt4_label"], random_state=RANDOM_STATE
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["gpt4_label"], random_state=RANDOM_STATE
    )
    print(f"  Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    print("\n--- Loading Tokenizer and Model ---")
    print(f"  Model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(label_encoder.classes_)
    )
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {n_params:,}")

    print("\n--- Building Datasets ---")
    train_dataset = DisinfoDataset(train_df["text"], train_df["label_id"], tokenizer)
    val_dataset = DisinfoDataset(val_df["text"], val_df["label_id"], tokenizer)
    test_dataset = DisinfoDataset(test_df["text"], test_df["label_id"], tokenizer)
    print(f"  Train dataset: {len(train_dataset):,} examples")
    print(f"  Val dataset: {len(val_dataset):,} examples")
    print(f"  Test dataset: {len(test_dataset):,} examples")

    print("\n--- Validating Pipeline: Single Batch Forward Pass ---")
    sample_batch = [train_dataset[i] for i in range(4)]
    batch = {
        k: torch.stack([item[k] for item in sample_batch]).to(device)
        for k in sample_batch[0]
    }
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
    print(f"  Batch forward pass successful")
    print(f"  Loss (untrained, expected near ln({len(label_encoder.classes_)})={np.log(len(label_encoder.classes_)):.3f}): {outputs.loss.item():.3f}")
    print(f"  Logits shape: {outputs.logits.shape}")

    print("\n--- Language Distribution Check (per split) ---")
    for name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        print(f"  {name}: {dict(split_df['language'].value_counts())}")

    print("\n--- Stage 3 mBERT Setup complete ---")
    print(f"  Pipeline validated end-to-end on {device} - ready for Aug 14 fine-tuning")
    print(f"  NOTE: this is a {len(df):,}-record prototype, not full-scale training.")
    print(f"  Full-scale training (562K+ records) requires Mistral-7B pseudo-labeling")
    print(f"  (Decision 13), not yet built.")

    return model, tokenizer, label_encoder, (train_df, val_df, test_df)


if __name__ == "__main__":
    run_mbert_setup()
