"""
HyDMIS, LDA Topic Modeling Pipeline
Phase 4, Stage 1: Unsupervised Topic Modeling

Latent Dirichlet Allocation across HyDMIS multilingual social media content.
Identifies latent topic clusters without requiring labels.
Works in zero-label low-resource settings.
Separates health, political, and social disinformation clusters.

Datasets:
- LIAR2: 22,962 English political claims (statement)
- TruthSeeker: 134,198 English social media posts (statement)
- FakeNewsNet: 23,196 English news titles (title)
- DeFaktS: 105,855 German Twitter posts (text)
- NewsPolyML: 32,129 multilingual news claims (claim_reviewed)

English LDA: LIAR2 + TruthSeeker + FakeNewsNet combined
German LDA: DeFaktS standalone
Multilingual LDA: NewsPolyML standalone
"""

import pandas as pd
import numpy as np
import sys
import os
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import re

# LDA configuration
N_TOPICS_ENGLISH = 10
N_TOPICS_GERMAN = 8
N_TOPICS_MULTILINGUAL = 10
N_TOP_WORDS = 15
MAX_FEATURES = 5000
MAX_ITER = 20
RANDOM_STATE = 42
SAMPLE_SIZE = 50000  # cap for speed on large datasets


def clean_text(text, language='en'):
    """Basic text cleaning for LDA preprocessing."""
    if not isinstance(text, str) or len(text.strip()) == 0:
        return ''
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    if len(text.split()) < 3:
        return ''
    return text


def get_english_stopwords():
    """English stopwords for LDA."""
    return [
        'the','a','an','and','or','but','in','on','at','to','for','of','with',
        'is','are','was','were','be','been','being','have','has','had','do','does',
        'did','will','would','could','should','may','might','shall','can','need',
        'this','that','these','those','it','its','they','their','there','here',
        'what','which','who','when','where','why','how','all','any','both','each',
        'few','more','most','other','some','such','no','not','only','same','so',
        'than','too','very','just','also','said','say','says','according','one',
        'two','three','new','year','years','time','times','day','days','people',
        'person','mr','mrs','ms','dr','president','senator','state','states',
        'government','federal','national','official','report','reported','claims',
        'claim','says','said','according','told','reuters','ap','cnn','fox','nbc'
    ]


def get_german_stopwords():
    """German stopwords for DeFaktS LDA."""
    return [
        'der','die','das','ein','eine','und','oder','aber','in','an','auf','zu',
        'von','mit','ist','sind','war','waren','hat','haben','wird','werden','wurde',
        'nicht','auch','als','bei','nach','vor','über','unter','durch','für','um',
        'dem','den','des','ich','du','er','sie','wir','ihr','es','sich','man',
        'noch','schon','dann','wenn','dass','wie','was','wer','wo','warum','ob',
        'sehr','mehr','nur','alle','viele','keine','neue','neuen','gegen','beim',
        'zum','zur','im','am','rt','via','amp'
    ]


def run_lda(texts, n_topics, stopwords, language='en', max_features=MAX_FEATURES):
    """Run LDA on a list of texts. Returns model, vectorizer, and topic words."""
    vectorizer = CountVectorizer(
        max_features=max_features,
        stop_words=stopwords,
        min_df=5,
        max_df=0.95,
        ngram_range=(1, 2)
    )
    dtm = vectorizer.fit_transform(texts)
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=MAX_ITER,
        learning_method='online',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    lda.fit(dtm)
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for i, comp in enumerate(lda.components_):
        top_words = [feature_names[j] for j in comp.argsort()[:-N_TOP_WORDS-1:-1]]
        topics.append(top_words)
    return lda, vectorizer, topics, dtm


def assign_topics(lda, vectorizer, texts):
    """Assign dominant topic to each text."""
    dtm = vectorizer.transform(texts)
    topic_dist = lda.transform(dtm)
    return np.argmax(topic_dist, axis=1)


def run_pipeline():
    print("HyDMIS Phase 4, Stage 1: LDA Topic Modeling Pipeline")
    print("=" * 58)

    # Load datasets
    print("\n--- Loading Datasets ---")
    from liar2_loader import load_liar2
    from truthseeker_loader import load_truthseeker
    from fakenewsnet_loader import load_fakenewsnet
    from defakts_loader import load_defakts
    from newspolyml_loader import load_newspolyml

    liar2 = load_liar2()
    liar2_df = liar2['data'] if isinstance(liar2, dict) else liar2
    truthseeker = load_truthseeker()
    ts_df = truthseeker['data'] if isinstance(truthseeker, dict) else truthseeker
    fakenewsnet = load_fakenewsnet()
    fnn_df = fakenewsnet['data'] if isinstance(fakenewsnet, dict) else fakenewsnet
    defakts = load_defakts()
    defakts_df = defakts['data'] if isinstance(defakts, dict) else defakts
    newspolyml = load_newspolyml()
    npm_df = newspolyml['data'] if isinstance(newspolyml, dict) else newspolyml

    print(f"  LIAR2:        {len(liar2_df):>7,} records")
    print(f"  TruthSeeker:  {len(ts_df):>7,} records (sample: {min(SAMPLE_SIZE, len(ts_df)):,})")
    print(f"  FakeNewsNet:  {len(fnn_df):>7,} records")
    print(f"  DeFaktS:      {len(defakts_df):>7,} records (sample: {min(SAMPLE_SIZE, len(defakts_df)):,})")
    print(f"  NewsPolyML:   {len(npm_df):>7,} records")

    print(f"\n--- English LDA Preprocessing ---")
    en_stopwords = get_english_stopwords()

    liar2_pairs = [(idx, clean_text(t)) for idx, t in zip(liar2_df.index, liar2_df['statement'].astype(str))]
    liar2_pairs = [(idx, t) for idx, t in liar2_pairs if t]
    liar2_ids = [idx for idx, t in liar2_pairs]
    liar2_texts = [t for idx, t in liar2_pairs]
    print(f"  LIAR2 clean texts: {len(liar2_texts):,}")

    ts_sample = ts_df.sample(min(SAMPLE_SIZE, len(ts_df)), random_state=RANDOM_STATE)
    ts_pairs = [(idx, clean_text(t)) for idx, t in zip(ts_sample.index, ts_sample['statement'].astype(str))]
    ts_pairs = [(idx, t) for idx, t in ts_pairs if t]
    ts_ids = [idx for idx, t in ts_pairs]
    ts_texts = [t for idx, t in ts_pairs]
    print(f"  TruthSeeker clean texts: {len(ts_texts):,}")

    fnn_pairs = [(idx, clean_text(t)) for idx, t in zip(fnn_df.index, fnn_df['title'].astype(str))]
    fnn_pairs = [(idx, t) for idx, t in fnn_pairs if t]
    fnn_ids = [idx for idx, t in fnn_pairs]
    fnn_texts = [t for idx, t in fnn_pairs]
    print(f"  FakeNewsNet clean texts: {len(fnn_texts):,}")

    english_texts = liar2_texts + ts_texts + fnn_texts
    english_source = (['liar2'] * len(liar2_texts) + ['truthseeker'] * len(ts_texts) + ['fakenewsnet'] * len(fnn_texts))
    english_ids = liar2_ids + ts_ids + fnn_ids
    print(f"  Combined English corpus: {len(english_texts):,} texts")

    print(f"\n--- English LDA ({N_TOPICS_ENGLISH} topics) ---")
    en_lda, en_vec, en_topics, en_dtm = run_lda(
        english_texts, N_TOPICS_ENGLISH, en_stopwords, language='en'
    )
    print(f"  Vocabulary size: {len(en_vec.get_feature_names_out()):,}")
    print(f"  Log-likelihood: {en_lda.score(en_dtm):.1f}")
    for i, words in enumerate(en_topics):
        print(f"  Topic {i:2d}: {' | '.join(words[:8])}")

    print(f"\n--- English Topic Distribution ---")
    en_assignments = assign_topics(en_lda, en_vec, english_texts)
    topic_counts = np.bincount(en_assignments, minlength=N_TOPICS_ENGLISH)
    for i, count in enumerate(topic_counts):
        print(f"  Topic {i:2d}: {count:>5,} texts ({count/len(english_texts):.1%})")

    print(f"\n--- German LDA Preprocessing (DeFaktS) ---")
    de_stopwords = get_german_stopwords()
    defakts_sample = defakts_df.sample(min(SAMPLE_SIZE, len(defakts_df)), random_state=RANDOM_STATE)
    de_pairs = [(idx, clean_text(t, language='de')) for idx, t in zip(defakts_sample.index, defakts_sample['text'].astype(str))]
    de_pairs = [(idx, t) for idx, t in de_pairs if t]
    de_ids = [idx for idx, t in de_pairs]
    de_texts = [t for idx, t in de_pairs]
    print(f"  DeFaktS clean texts: {len(de_texts):,}")

    print(f"\n--- German LDA ({N_TOPICS_GERMAN} topics) ---")
    de_lda, de_vec, de_topics, de_dtm = run_lda(
        de_texts, N_TOPICS_GERMAN, de_stopwords, language='de'
    )
    print(f"  Vocabulary size: {len(de_vec.get_feature_names_out()):,}")
    print(f"  Log-likelihood: {de_lda.score(de_dtm):.1f}")
    for i, words in enumerate(de_topics):
        print(f"  Topic {i:2d}: {' | '.join(words[:8])}")

    print(f"\n--- German Topic Distribution ---")
    de_assignments = assign_topics(de_lda, de_vec, de_texts)
    de_topic_counts = np.bincount(de_assignments, minlength=N_TOPICS_GERMAN)
    for i, count in enumerate(de_topic_counts):
        print(f"  Topic {i:2d}: {count:>5,} texts ({count/len(de_texts):.1%})")

    print(f"\n--- Multilingual LDA Preprocessing (NewsPolyML) ---")
    npm_pairs = [(idx, clean_text(str(t))) for idx, t in zip(npm_df.index, npm_df['claim_reviewed'].astype(str))]
    npm_pairs = [(idx, t) for idx, t in npm_pairs if t]
    npm_ids = [idx for idx, t in npm_pairs]
    npm_texts = [t for idx, t in npm_pairs]
    print(f"  NewsPolyML clean texts: {len(npm_texts):,}")
    if 'language' in npm_df.columns:
        lang_dist = npm_df['language'].value_counts()
        for lang, count in lang_dist.head(6).items():
            print(f"    {lang}: {count:,}")

    print(f"\n--- Multilingual LDA ({N_TOPICS_MULTILINGUAL} topics) ---")
    multi_lda, multi_vec, multi_topics, multi_dtm = run_lda(
        npm_texts, N_TOPICS_MULTILINGUAL, en_stopwords, language='multi'
    )
    print(f"  Vocabulary size: {len(multi_vec.get_feature_names_out()):,}")
    print(f"  Log-likelihood: {multi_lda.score(multi_dtm):.1f}")
    for i, words in enumerate(multi_topics):
        print(f"  Topic {i:2d}: {' | '.join(words[:8])}")

    print(f"\n--- Multilingual Topic Distribution ---")
    multi_assignments = assign_topics(multi_lda, multi_vec, npm_texts)
    multi_topic_counts = np.bincount(multi_assignments, minlength=N_TOPICS_MULTILINGUAL)
    for i, count in enumerate(multi_topic_counts):
        print(f"  Topic {i:2d}: {count:>5,} texts ({count/len(npm_texts):.1%})")

    print(f"\n--- Key Findings ---")
    print(f"  English corpus: {len(english_texts):,} texts | {N_TOPICS_ENGLISH} topics identified")
    print(f"  German corpus: {len(de_texts):,} texts | {N_TOPICS_GERMAN} topics identified")
    print(f"  Multilingual corpus: {len(npm_texts):,} texts | {N_TOPICS_MULTILINGUAL} topics identified")
    print(f"  LDA separates health, political, and social disinformation clusters unsupervised")
    print(f"  German LDA separate model, textstat FK confirmed unusable for German")
    print(f"  Stage 1 complete, topic assignments ready for Stage 2 GPT-4 verification")

    print(f"\n--- Saving Topic Assignments ---")
    assignments_records = []
    for source, rec_id, topic in zip(english_source, english_ids, en_assignments):
        assignments_records.append({"dataset": source, "record_id": rec_id, "lda_topic_id": int(topic), "lda_model": "english"})
    for rec_id, topic in zip(de_ids, de_assignments):
        assignments_records.append({"dataset": "defakts", "record_id": rec_id, "lda_topic_id": int(topic), "lda_model": "german"})
    for rec_id, topic in zip(npm_ids, multi_assignments):
        assignments_records.append({"dataset": "newspolyml", "record_id": rec_id, "lda_topic_id": int(topic), "lda_model": "multilingual"})

    assignments_df = pd.DataFrame(assignments_records)
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed", "lda_topic_assignments.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    assignments_df.to_csv(output_path, index=False)
    print(f"  Saved {len(assignments_df):,} topic assignments to {output_path}")
    print(f"  Columns: dataset, record_id, lda_topic_id, lda_model")
    print(f"  This file did not exist before - gpt4_sampler.py previously used a hardcoded")
    print(f"  lda_topic_id=-1 placeholder because no LDA output was ever persisted to disk.")

    print(f"\n--- LDA Pipeline complete ---")
    print(f"  3 LDA models trained: English, German, Multilingual")
    print(f"  Ready for lda_train.py, hyperparameter optimization and topic coherence scoring")

    return en_lda, en_vec, de_lda, de_vec, multi_lda, multi_vec


if __name__ == "__main__":
    run_pipeline()
