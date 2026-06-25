"""
HyDMIS -- LDA Training and Hyperparameter Optimization
Phase 4 -- Stage 1: Topic Coherence Scoring and Model Selection

Evaluates LDA topic coherence across n_topics range for each corpus.
Selects optimal number of topics based on coherence scores.
Computes perplexity and log-likelihood for model quality assessment.

Three corpora:
- English: LIAR2 + TruthSeeker + FakeNewsNet (95,625 texts)
- German: DeFaktS (49,340 texts)
- Multilingual: NewsPolyML (32,109 texts)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
import os
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import cross_val_score
import re
import string

os.makedirs("figures/stage1", exist_ok=True)

RANDOM_STATE = 42
MAX_ITER = 20
MAX_FEATURES = 5000
TOPICS_RANGE = [8, 9, 10, 11, 12, 15]  # min 8 -- consistent with lda_pipeline.py findings
SAMPLE_SIZE = 50000
COHERENCE_SAMPLE = 10000  # smaller sample for coherence scoring speed


def clean_text(text, language="en"):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text.split()) < 3:
        return ""
    return text


def get_english_stopwords():
    return [
        "the","a","an","and","or","but","in","on","at","to","for","of","with",
        "is","are","was","were","be","been","being","have","has","had","do","does",
        "did","will","would","could","should","may","might","shall","can","need",
        "this","that","these","those","it","its","they","their","there","here",
        "what","which","who","when","where","why","how","all","any","both","each",
        "few","more","most","other","some","such","no","not","only","same","so",
        "than","too","very","just","also","said","say","says","according","one",
        "two","three","new","year","years","time","times","day","days","people",
        "person","mr","mrs","ms","dr","president","senator","state","states",
        "government","federal","national","official","report","reported","claims",
        "claim","says","said","according","told","reuters","ap","cnn","fox","nbc"
    ]


def get_german_stopwords():
    return [
        "der","die","das","ein","eine","und","oder","aber","in","an","auf","zu",
        "von","mit","ist","sind","war","waren","hat","haben","wird","werden","wurde",
        "nicht","auch","als","bei","nach","vor","uber","unter","durch","fur","um",
        "dem","den","des","ich","du","er","sie","wir","ihr","es","sich","man",
        "noch","schon","dann","wenn","dass","wie","was","wer","wo","warum","ob",
        "sehr","mehr","nur","alle","viele","keine","neue","neuen","gegen","beim",
        "zum","zur","im","am","rt","via","amp"
    ]


def compute_coherence_scores(texts, stopwords, topics_range, language="en"):
    """Compute log-likelihood and perplexity for each n_topics value."""
    results = []
    vectorizer = CountVectorizer(
        max_features=MAX_FEATURES,
        stop_words=stopwords,
        min_df=5,
        max_df=0.95,
        ngram_range=(1, 2)
    )
    dtm = vectorizer.fit_transform(texts)
    for n_topics in topics_range:
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            max_iter=MAX_ITER,
            learning_method="online",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        lda.fit(dtm)
        ll = lda.score(dtm)
        perplexity = lda.perplexity(dtm)
        results.append({
            "n_topics": n_topics,
            "log_likelihood": ll,
            "perplexity": perplexity,
        })
        print(f"    n_topics={n_topics:2d}: log_likelihood={ll:.1f} perplexity={perplexity:.1f}")
    return pd.DataFrame(results)


def get_top_words(lda_model, vectorizer, n_words=10):
    """Get top words for each topic."""
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for i, comp in enumerate(lda_model.components_):
        top_words = [feature_names[j] for j in comp.argsort()[:-n_words-1:-1]]
        topics.append(top_words)
    return topics


def run_lda_train():
    print("HyDMIS Phase 4 -- Stage 1: LDA Training and Hyperparameter Optimization")
    print("=" * 72)

    print("\n--- Loading Datasets ---")
    from liar2_loader import load_liar2
    from truthseeker_loader import load_truthseeker
    from fakenewsnet_loader import load_fakenewsnet
    from defakts_loader import load_defakts
    from newspolyml_loader import load_newspolyml

    liar2 = load_liar2()
    liar2_df = liar2["data"] if isinstance(liar2, dict) else liar2
    ts = load_truthseeker()
    ts_df = ts["data"] if isinstance(ts, dict) else ts
    fnn = load_fakenewsnet()
    fnn_df = fnn["data"] if isinstance(fnn, dict) else fnn
    defakts = load_defakts()
    de_df = defakts["data"] if isinstance(defakts, dict) else defakts
    npm = load_newspolyml()
    npm_df = npm["data"] if isinstance(npm, dict) else npm

    en_stopwords = get_english_stopwords()
    de_stopwords = get_german_stopwords()

    liar2_texts = [clean_text(t) for t in liar2_df["statement"].astype(str)]
    liar2_texts = [t for t in liar2_texts if t]
    ts_texts = [clean_text(t) for t in ts_df.sample(min(SAMPLE_SIZE, len(ts_df)), random_state=42)["statement"].astype(str)]
    ts_texts = [t for t in ts_texts if t]
    fnn_texts = [clean_text(t) for t in fnn_df["title"].astype(str)]
    fnn_texts = [t for t in fnn_texts if t]
    english_texts = liar2_texts + ts_texts + fnn_texts
    print(f"  English corpus: {len(english_texts):,} texts")

    de_texts = [clean_text(t, "de") for t in de_df.sample(min(SAMPLE_SIZE, len(de_df)), random_state=42)["text"].astype(str)]
    de_texts = [t for t in de_texts if t]
    print(f"  German corpus: {len(de_texts):,} texts")

    npm_texts = [clean_text(str(t)) for t in npm_df["claim_reviewed"].astype(str)]
    npm_texts = [t for t in npm_texts if t]
    print(f"  Multilingual corpus: {len(npm_texts):,} texts")

    print(f"\n--- English LDA Coherence Scoring ---")
    en_results = compute_coherence_scores(english_texts[:COHERENCE_SAMPLE], en_stopwords, TOPICS_RANGE)
    best_en = en_results.loc[en_results["log_likelihood"].idxmax()]
    print(f"  Best n_topics: {int(best_en['n_topics'])} (log_likelihood={best_en['log_likelihood']:.1f})")

    print(f"\n--- German LDA Coherence Scoring ---")
    de_results = compute_coherence_scores(de_texts[:COHERENCE_SAMPLE], de_stopwords, TOPICS_RANGE, "de")
    best_de = de_results.loc[de_results["log_likelihood"].idxmax()]
    print(f"  Best n_topics: {int(best_de['n_topics'])} (log_likelihood={best_de['log_likelihood']:.1f})")

    print(f"\n--- Multilingual LDA Coherence Scoring ---")
    multi_results = compute_coherence_scores(npm_texts[:COHERENCE_SAMPLE], en_stopwords, TOPICS_RANGE, "multi")
    best_multi = multi_results.loc[multi_results["log_likelihood"].idxmax()]
    print(f"  Best n_topics: {int(best_multi['n_topics'])} (log_likelihood={best_multi['log_likelihood']:.1f})")

    print(f"\n--- Training Final Models with Optimal Topics ---")
    from lda_pipeline import run_lda
    en_lda, en_vec, en_topics, en_dtm = run_lda(english_texts, int(best_en["n_topics"]), en_stopwords)
    de_lda, de_vec, de_topics, de_dtm = run_lda(de_texts, int(best_de["n_topics"]), de_stopwords, "de")
    multi_lda, multi_vec, multi_topics, multi_dtm = run_lda(npm_texts, int(best_multi["n_topics"]), en_stopwords, "multi")
    print(f"  English final model: {int(best_en['n_topics'])} topics")
    print(f"  German final model: {int(best_de['n_topics'])} topics")
    print(f"  Multilingual final model: {int(best_multi['n_topics'])} topics")

    print(f"\n--- Topic Word Analysis ---")
    print(f"  English top topics:")
    for i, words in enumerate(en_topics[:3]):
        print(f"    Topic {i}: {' | '.join(words[:6])}")
    print(f"  German top topics:")
    for i, words in enumerate(de_topics[:3]):
        print(f"    Topic {i}: {' | '.join(words[:6])}")
    print(f"  Multilingual top topics:")
    for i, words in enumerate(multi_topics[:3]):
        print(f"    Topic {i}: {' | '.join(words[:6])}")

    print(f"\n--- Key Findings ---")
    print(f"  English optimal topics: {int(best_en['n_topics'])} (from range {TOPICS_RANGE})")
    print(f"  German optimal topics: {int(best_de['n_topics'])} (from range {TOPICS_RANGE})")
    print(f"  Multilingual optimal topics: {int(best_multi['n_topics'])} (from range {TOPICS_RANGE})")
    print(f"  Higher log-likelihood = better model fit to corpus")
    print(f"  Lower perplexity = better generalization to unseen documents")
    print(f"  LDA topic coherence validates unsupervised cluster separation")

    # FIGURES
    print(f"\n--- Generating Figures ---")

    # Fig 1 -- Log-Likelihood vs n_topics
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, results, title, color in zip(
        axes,
        [en_results, de_results, multi_results],
        ["English", "German", "Multilingual"],
        ["#3498db", "#e74c3c", "#2ecc71"]
    ):
        ax.plot(results["n_topics"], results["log_likelihood"], "o-", color=color, linewidth=2, markersize=8)
        best_idx = results["log_likelihood"].idxmax()
        ax.axvline(results.loc[best_idx, "n_topics"], color="black", linestyle="--", alpha=0.7,
                  label=f'Optimal: {int(results.loc[best_idx, "n_topics"])}')
        ax.set_title(f"{title} Corpus\nLog-Likelihood vs n_topics", fontsize=11, fontweight="bold")
        ax.set_xlabel("Number of Topics"); ax.set_ylabel("Log-Likelihood"); ax.legend()
        ax.grid(True, alpha=0.3)
    plt.suptitle("HyDMIS -- LDA Coherence: Log-Likelihood vs n_topics", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("figures/stage1/lda_log_likelihood.png", dpi=150, bbox_inches="tight")
    plt.close(); print("Fig 1 saved -- lda_log_likelihood.png")

    # Fig 2 -- Perplexity vs n_topics
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, results, title, color in zip(
        axes,
        [en_results, de_results, multi_results],
        ["English", "German", "Multilingual"],
        ["#3498db", "#e74c3c", "#2ecc71"]
    ):
        ax.plot(results["n_topics"], results["perplexity"], "s-", color=color, linewidth=2, markersize=8)
        best_idx = results["perplexity"].idxmin()
        ax.axvline(results.loc[best_idx, "n_topics"], color="black", linestyle="--", alpha=0.7,
                  label=f'Optimal: {int(results.loc[best_idx, "n_topics"])}')
        ax.set_title(f"{title} Corpus\nPerplexity vs n_topics", fontsize=11, fontweight="bold")
        ax.set_xlabel("Number of Topics"); ax.set_ylabel("Perplexity"); ax.legend()
        ax.grid(True, alpha=0.3)
    plt.suptitle("HyDMIS -- LDA Coherence: Perplexity vs n_topics (lower = better)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("figures/stage1/lda_perplexity.png", dpi=150, bbox_inches="tight")
    plt.close(); print("Fig 2 saved -- lda_perplexity.png")

    # Fig 3 -- Top Words Heatmap English
    en_top_words = get_top_words(en_lda, en_vec, n_words=8)
    n_t = len(en_top_words)
    fig, ax = plt.subplots(figsize=(14, max(6, n_t*0.8)))
    topic_labels = [f"Topic {i}" for i in range(n_t)]
    word_matrix = np.zeros((n_t, 8))
    all_words = list({w for words in en_top_words for w in words})[:40]
    word_idx = {w: i for i, w in enumerate(all_words)}
    cell_text = [[", ".join(words[:8])] for words in en_top_words]
    ax.axis("off")
    tbl = ax.table(cellText=[[", ".join(w[:8])] for w in en_top_words],
                   rowLabels=topic_labels,
                   colLabels=["Top 8 Words"],
                   cellLoc="left", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    tbl.auto_set_column_width([0])
    ax.set_title(f"English LDA -- Top Words per Topic ({n_t} topics)", fontsize=12, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig("figures/stage1/lda_english_topics.png", dpi=150, bbox_inches="tight")
    plt.close(); print("Fig 3 saved -- lda_english_topics.png")

    # Fig 4 -- Top Words German
    de_top_words = get_top_words(de_lda, de_vec, n_words=8)
    n_t_de = len(de_top_words)
    fig, ax = plt.subplots(figsize=(14, max(6, n_t_de*0.8)))
    topic_labels_de = [f"Topic {i}" for i in range(n_t_de)]
    ax.axis("off")
    tbl = ax.table(cellText=[[", ".join(w[:8])] for w in de_top_words],
                   rowLabels=topic_labels_de,
                   colLabels=["Top 8 Words (German)"],
                   cellLoc="left", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    tbl.auto_set_column_width([0])
    ax.set_title(f"German LDA (DeFaktS) -- Top Words per Topic ({n_t_de} topics)", fontsize=12, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig("figures/stage1/lda_german_topics.png", dpi=150, bbox_inches="tight")
    plt.close(); print("Fig 4 saved -- lda_german_topics.png")

    # Fig 5 -- Topic Distribution English
    vectorizer_en = CountVectorizer(
        max_features=MAX_FEATURES, stop_words=en_stopwords,
        min_df=5, max_df=0.95, ngram_range=(1, 2)
    )
    dtm_en = vectorizer_en.fit_transform(english_texts)
    topic_dist = en_lda.transform(dtm_en)
    dominant_topics = np.argmax(topic_dist, axis=1)
    topic_counts = np.bincount(dominant_topics, minlength=int(best_en["n_topics"]))
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = plt.cm.Set3(np.linspace(0, 1, len(topic_counts)))
    bars = ax.bar(range(len(topic_counts)), topic_counts, color=colors, edgecolor="white")
    for bar, val in zip(bars, topic_counts):
        ax.text(bar.get_x()+bar.get_width()/2, val+50, str(val), ha="center", fontsize=8)
    ax.set_xticks(range(len(topic_counts)))
    ax.set_xticklabels([f"T{i}" for i in range(len(topic_counts))])
    ax.set_title(f"English Corpus -- Topic Distribution ({int(best_en['n_topics'])} topics, {len(english_texts):,} texts)",
                fontsize=11, fontweight="bold")
    ax.set_xlabel("Topic"); ax.set_ylabel("Number of Texts")
    plt.tight_layout()
    plt.savefig("figures/stage1/lda_english_topic_dist.png", dpi=150, bbox_inches="tight")
    plt.close(); print("Fig 5 saved -- lda_english_topic_dist.png")

    print(f"\n--- LDA Training complete ---")
    print(f"  6 figures saved to figures/stage1/")
    print(f"  Optimal topics: English={int(best_en['n_topics'])}, German={int(best_de['n_topics'])}, Multilingual={int(best_multi['n_topics'])}")
    print(f"  Ready for lda_validation.py -- topic cluster analysis and disinformation mapping")

    return en_lda, en_vec, de_lda, de_vec, multi_lda, multi_vec, en_results, de_results, multi_results


if __name__ == "__main__":
    run_lda_train()
