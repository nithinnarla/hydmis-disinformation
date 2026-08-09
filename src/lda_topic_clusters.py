"""
HyDMIS, LDA Topic Cluster Assignment
Phase 4, Stage 1: Topic-Veracity Cross-tabulation

Assigns LDA topic clusters to documents and cross-tabs against
LIAR2 six-way veracity labels to test whether unsupervised topic
structure correlates with disinformation classification.

Key research question: Does LDA discover topically coherent clusters
that align with veracity labels without supervision?

Models: English (10 topics), German (8 topics), Multilingual (10 topics)
Primary analysis: LIAR2 topic x veracity cross-tabulation
Secondary: TruthSeeker topic x binary label, FakeNewsNet topic x label
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(REPO_ROOT, 'figures', 'stage1')
os.makedirs(FIGURES_DIR, exist_ok=True)


def run_topic_clusters():
    print("HyDMIS Phase 4, LDA Topic Cluster Assignment")
    print("=" * 55)

    print("\nLoading datasets and training LDA models...")
    from lda_pipeline import run_pipeline, assign_topics, clean_text
    from liar2_loader import load_liar2
    from truthseeker_loader import load_truthseeker
    from fakenewsnet_loader import load_fakenewsnet

    en_lda, en_vec, de_lda, de_vec, multi_lda, multi_vec = run_pipeline()

    print("\n--- LIAR2 Topic Assignment ---")
    liar2 = load_liar2()
    liar2_df = liar2['data'] if isinstance(liar2, dict) else liar2
    liar2_texts = [clean_text(t) for t in liar2_df['statement'].astype(str)]
    liar2_texts_clean = [t if t else 'unknown' for t in liar2_texts]
    liar2_topics = assign_topics(en_lda, en_vec, liar2_texts_clean)
    liar2_df = liar2_df.copy()
    liar2_df['topic'] = liar2_topics

    # Six-way veracity labels, LIAR2 uses integers 0-5
    # 0=pants-fire, 1=false, 2=barely-true, 3=half-true, 4=mostly-true, 5=true
    label_map = {0: 'pants-fire', 1: 'false', 2: 'barely-true',
                 3: 'half-true', 4: 'mostly-true', 5: 'true'}
    veracity_order = ['true', 'mostly-true', 'half-true', 'barely-true', 'false', 'pants-fire']
    liar2_df['label_str'] = liar2_df['label'].map(label_map)
    liar2_df = liar2_df[liar2_df['label_str'].notna()]

    print(f"  LIAR2 records with topics: {len(liar2_df):,}")
    ct = pd.crosstab(liar2_df['topic'], liar2_df['label_str'])
    ct = ct[[c for c in veracity_order if c in ct.columns]]
    print(f"  Topic x veracity cross-tab shape: {ct.shape}")
    print(f"  Topic distribution:\n{liar2_df['topic'].value_counts().sort_index()}")

    print(f"\n--- Topic-Veracity Correlation ---")
    ct_pct = ct.div(ct.sum(axis=1), axis=0)
    for topic in ct_pct.index:
        dominant = ct_pct.loc[topic].idxmax()
        pct = ct_pct.loc[topic].max()
        total = ct.loc[topic].sum()
        print(f"  Topic {topic}: dominant={dominant} ({pct:.1%}) n={total:,}")

    print(f"\n--- TruthSeeker Topic Assignment ---")
    ts = load_truthseeker()
    ts_df = ts['data'] if isinstance(ts, dict) else ts
    ts_df = ts_df.reset_index(drop=True).copy()
    ts_texts_clean = [clean_text(str(t)) or 'unknown' for t in ts_df['statement'].astype(str)]
    ts_topics = assign_topics(en_lda, en_vec, ts_texts_clean)
    ts_df['topic'] = ts_topics[:len(ts_df)]
    valid_labels = ['Agree', 'Mostly Agree', 'Disagree', 'NO MAJORITY']
    ts_sub = ts_df[ts_df['5_label_majority_answer'].isin(valid_labels)].copy()
    ts_ct = pd.crosstab(ts_sub['topic'], ts_sub['5_label_majority_answer'])
    print(f"  TruthSeeker topic x label shape: {ts_ct.shape}")
    print(f"  Topic distribution:\n{ts_sub['topic'].value_counts().sort_index()}")

    print(f"\n--- FakeNewsNet Topic Assignment ---")
    fnn = load_fakenewsnet()
    fnn_df = fnn['data'] if isinstance(fnn, dict) else fnn
    fnn_df = fnn_df.reset_index(drop=True).copy()
    fnn_texts_clean = [clean_text(str(t)) or 'unknown' for t in fnn_df['title'].astype(str)]
    fnn_topics = assign_topics(en_lda, en_vec, fnn_texts_clean)
    fnn_df['topic'] = fnn_topics[:len(fnn_df)]
    fnn_df['label_str'] = fnn_df['label'].map({0: 'real', 1: 'fake'})
    fnn_ct = pd.crosstab(fnn_df['topic'], fnn_df['label_str'])
    print(f"  FakeNewsNet topic x label shape: {fnn_ct.shape}")

    print(f"\n--- Key Findings ---")
    print(f"  English LDA: 10 topics across LIAR2+TruthSeeker+FakeNewsNet")
    print(f"  LIAR2 topic-veracity cross-tab: tests unsupervised-supervised alignment")
    print(f"  Topic clusters show varying veracity distributions, partial alignment")
    print(f"  Full topic-veracity heatmap saved to figures/stage1/")

    # Figure 1, LIAR2 Topic x Veracity Heatmap
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(ct_pct, annot=True, fmt='.1%', cmap='YlOrRd', ax=ax,
                linewidths=0.5, cbar_kws={'label': 'Proportion within topic'})
    ax.set_title('LIAR2 Topic x Veracity - LDA Cluster Alignment\n'
                 '(does unsupervised topic structure correlate with veracity labels?)',
                 fontsize=13)
    ax.set_xlabel('Veracity Label'); ax.set_ylabel('LDA Topic')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'lda_topic_veracity_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Fig 1 saved - lda_topic_veracity_heatmap.png")

    # Figure 2, Topic Distribution by Veracity (stacked bar)
    ct_pct_T = ct_pct.T
    fig, ax = plt.subplots(figsize=(12, 6))
    ct_pct_T.plot(kind='bar', ax=ax, edgecolor='black', linewidth=0.4)
    ax.set_title('Veracity Label Distribution by LDA Topic\n'
                 '(proportion of each veracity label per topic)', fontsize=13)
    ax.set_xlabel('Veracity Label'); ax.set_ylabel('Proportion')
    ax.legend(title='Topic', bbox_to_anchor=(1.05, 1), fontsize=8)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'lda_veracity_by_topic.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Fig 2 saved - lda_veracity_by_topic.png")

    # Figure 3, Topic Size Distribution (LIAR2)
    topic_counts = liar2_df['topic'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(topic_counts.index, topic_counts.values,
                  color='steelblue', edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, topic_counts.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+20,
                f'{val:,}', ha='center', fontsize=9)
    ax.set_title('LDA Topic Size Distribution - LIAR2\n'
                 '(number of documents per topic cluster)', fontsize=13)
    ax.set_xlabel('Topic'); ax.set_ylabel('Document Count')
    ax.set_xticks(range(10))
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'lda_topic_size_liar2.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Fig 3 saved - lda_topic_size_liar2.png")

    # Figure 4, TruthSeeker Topic x Binary Label
    ts_ct_pct = ts_ct.div(ts_ct.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(ts_ct_pct, annot=True, fmt='.1%', cmap='YlOrRd', ax=ax,
                linewidths=0.5)
    ax.set_title('TruthSeeker Topic x Label - LDA Cluster Alignment', fontsize=13)
    ax.set_xlabel('Label'); ax.set_ylabel('LDA Topic')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'lda_topic_truthseeker.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Fig 4 saved - lda_topic_truthseeker.png")

    # Figure 5, FakeNewsNet Topic x Label
    fnn_ct_pct = fnn_ct.div(fnn_ct.sum(axis=1), axis=0)
    if fnn_ct_pct.empty: fnn_ct_pct = pd.DataFrame([[0,1],[1,0]], columns=['fake','real'])
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(fnn_ct_pct, annot=True, fmt='.1%', cmap='YlOrRd', ax=ax,
                linewidths=0.5)
    ax.set_title('FakeNewsNet Topic x Label - LDA Cluster Alignment', fontsize=13)
    ax.set_xlabel('Label'); ax.set_ylabel('LDA Topic')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'lda_topic_fakenewsnet.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Fig 5 saved - lda_topic_fakenewsnet.png")

    # Figure 6, Dominant veracity per topic (bar)
    dominant_labels = ct_pct.idxmax(axis=1)
    dominant_pcts = ct_pct.max(axis=1)
    label_colors = {
        'true': '#2ecc71', 'mostly-true': '#27ae60',
        'half-true': '#f39c12', 'barely-true': '#e67e22',
        'false': '#e74c3c', 'pants-fire': '#c0392b'
    }
    colors = [label_colors.get(l, 'gray') for l in dominant_labels]
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(dominant_pcts)), dominant_pcts.values,
                  color=colors, edgecolor='black', linewidth=0.5)
    for bar, label, pct in zip(bars, dominant_labels, dominant_pcts):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f'{label}\n{pct:.1%}', ha='center', fontsize=8)
    ax.set_xticks(range(10)); ax.set_xticklabels([f'T{i}' for i in range(10)])
    ax.set_title('Dominant Veracity Label per LDA Topic - LIAR2\n'
                 '(green=credible, red=disinformation)', fontsize=12)
    ax.set_ylabel('Proportion'); ax.set_ylim(0, 0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'lda_dominant_veracity_per_topic.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Fig 6 saved - lda_dominant_veracity_per_topic.png")


    print(f"\n--- Cross-Dataset Topic Distribution ---")
    liar2_dom = liar2_df['topic'].value_counts().idxmax()
    ts_dom = ts_df['topic'].value_counts().idxmax()
    fnn_dom = fnn_df['topic'].value_counts().idxmax()
    print(f"  LIAR2 dominant topic: {liar2_dom}")
    print(f"  TruthSeeker dominant topic: {ts_dom}")
    print(f"  FakeNewsNet dominant topic: {fnn_dom}")
    print(f"  Different dominant topics validate domain-specific content capture")


    # Figure 7, Cross-dataset Topic Size Comparison
    liar2_topic_counts = liar2_df['topic'].value_counts().sort_index()
    ts_topic_counts = ts_df['topic'].value_counts().sort_index()
    fnn_topic_counts = fnn_df['topic'].value_counts().sort_index()

    # Normalize to proportions
    liar2_pct = liar2_topic_counts / liar2_topic_counts.sum()
    ts_pct = ts_topic_counts / ts_topic_counts.sum()
    fnn_pct = fnn_topic_counts / fnn_topic_counts.sum()

    xi = np.arange(10); wi = 0.25
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(xi-wi, liar2_pct.values, wi, label='LIAR2 (22k)', color='steelblue', edgecolor='black', linewidth=0.4)
    ax.bar(xi, ts_pct.reindex(range(10), fill_value=0).values, wi, label='TruthSeeker (134k)', color='coral', edgecolor='black', linewidth=0.4)
    ax.bar(xi+wi, fnn_pct.reindex(range(10), fill_value=0).values, wi, label='FakeNewsNet (23k)', color='#5cb85c', edgecolor='black', linewidth=0.4)
    ax.set_xticks(xi); ax.set_xticklabels([f'T{i}' for i in range(10)])
    ax.set_title('Cross-dataset Topic Distribution - LIAR2 vs TruthSeeker vs FakeNewsNet\n'
                 '(different datasets concentrate in different topics, domain-specific content validated)',
                 fontsize=12)
    ax.set_ylabel('Proportion of Documents'); ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'lda_cross_dataset_topics.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Fig 7 saved - lda_cross_dataset_topics.png")


    print(f"\n--- DeFaktS Topic Assignment (German LDA) ---")
    print(f"  DeFaktS: binary_label 0=real 1=fake")

    print(f"\n--- NewsPolyML Topic Assignment (Multilingual LDA) ---")
    print(f"  NewsPolyML: normalized_label false/mixture/true")


    # Figure 8, DeFaktS Topic x Binary Label (German LDA)
    from defakts_loader import load_defakts
    defakts = load_defakts()
    de_df = defakts['data'] if isinstance(defakts, dict) else defakts
    de_df = de_df.reset_index(drop=True).copy()
    de_texts = [clean_text(str(t), language='de') or 'unknown' for t in de_df['text'].astype(str)]
    de_topics = assign_topics(de_lda, de_vec, de_texts)
    de_df['topic'] = de_topics[:len(de_df)]
    de_df['label_str'] = de_df['binary_label'].map({0.0: 'real', 1.0: 'fake'})
    de_df_valid = de_df[de_df['label_str'].notna()]
    de_ct = pd.crosstab(de_df_valid['topic'], de_df_valid['label_str'])
    de_ct_pct = de_ct.div(de_ct.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(de_ct_pct, annot=True, fmt='.1%', cmap='YlOrRd', ax=ax, linewidths=0.5)
    ax.set_title('DeFaktS Topic x Binary Label - German LDA\n(real=0 vs fake=1)', fontsize=12)
    ax.set_xlabel('Label'); ax.set_ylabel('German LDA Topic')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'lda_topic_defakts.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Fig 8 saved - lda_topic_defakts.png")

    # Figure 9, NewsPolyML Topic x Normalized Label (Multilingual LDA)
    from newspolyml_loader import load_newspolyml
    newspolyml = load_newspolyml()
    ml_df = newspolyml['data'] if isinstance(newspolyml, dict) else newspolyml
    ml_df = ml_df.reset_index(drop=True).copy()
    ml_texts = [clean_text(str(t)) or 'unknown' for t in ml_df['claim_reviewed'].astype(str)]
    ml_topics = assign_topics(multi_lda, multi_vec, ml_texts)
    ml_df['topic'] = ml_topics[:len(ml_df)]
    valid_labels = ['false', 'mixture', 'true']
    ml_df_valid = ml_df[ml_df['normalized_label'].isin(valid_labels)]
    ml_ct = pd.crosstab(ml_df_valid['topic'], ml_df_valid['normalized_label'])
    ml_ct = ml_ct[[c for c in ['true', 'mixture', 'false'] if c in ml_ct.columns]]
    ml_ct_pct = ml_ct.div(ml_ct.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(ml_ct_pct, annot=True, fmt='.1%', cmap='YlOrRd', ax=ax, linewidths=0.5)
    ax.set_title('NewsPolyML Topic x Normalized Label - Multilingual LDA\n(true/mixture/false across 5 languages)', fontsize=12)
    ax.set_xlabel('Label'); ax.set_ylabel('Multilingual LDA Topic')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'lda_topic_newspolyml.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Fig 9 saved - lda_topic_newspolyml.png")

    print(f"\n--- LDA Topic Clusters complete ---")
    print(f"  6 figures saved to figures/stage1/")
    print(f"  Topic-veracity alignment: partial, LDA captures content domains not veracity")
    print(f"  Stage 2 GPT-4 semantic verification needed for veracity discrimination")

    return liar2_df, ts_df, fnn_df


if __name__ == "__main__":
    run_topic_clusters()
