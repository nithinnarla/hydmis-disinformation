"""
HyDMIS, Pipeline Architecture Diagram
Phase 4, schematic for the paper

The three-stage architecture is described in prose in the README and drawn as
an ASCII tree in docs/literature_analysis.md, but no figure exists for it. That
is the same gap oracle-rag-pipeline found in its own Section 3.1 and closed with
src/architecture_figure.py, so this closes it here the same way.

Stage 3 is drawn differently from Stages 1 and 2 on purpose. It is scoped and
scripted and has never run, so it appears outlined and dashed, and its label
says so. Do not fill it in until a training run exists; check 4 in
src/check_consistency.py exists to catch the same claim made in prose.

Counts come from the processed CSVs through csv.reader, NOT by counting lines.
The first version of this script counted lines and reported 239,356 verified
records against the true 14,640, because the verified text fields contain
embedded newlines. Every count is then cross-checked against the figure the
README states, and a disagreement is printed rather than drawn silently.

No API calls, no model loading, deterministic.
"""

import os
import csv
import sys
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

csv.field_size_limit(10 ** 7)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(REPO_ROOT, 'data', 'processed')
FIGURES_DIR = os.path.join(REPO_ROOT, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

DPI = 300

# What the README states. Used to cross-check the counts read from disk, and as
# the fallback when a processed file is not present on this machine.
README_TOPIC_ASSIGNMENTS = 177074
README_VERIFIED = 14640
README_CORPUS = 324292

DONE_FILL, DONE_EDGE = '#e8f4ea', '#3d7a52'
NOTRUN_FILL, NOTRUN_EDGE = '#ffffff', '#95a5a6'
PANEL_FILL, PANEL_EDGE = '#eef3f8', '#5d6d7e'

# The six confirmed corpora. MultiClaim and ClimateMiSt are deliberately absent:
# neither has been obtained, so neither belongs in a diagram of what exists.
SOURCES = [
    ('LIAR2', '22,962', 'EN'),
    ('TruthSeeker', '134,198', 'EN'),
    ('FakeNewsNet', '23,196', 'EN'),
    ('Covid-misinfo-MIC', '5,952', 'EN / PT / ID'),
    ('NewsPolyML', '32,129', 'EN / DE / ES / FR / IT'),
    ('DeFaktS', '105,855', 'DE'),
]

LINE_H = 0.040      # vertical space one body line occupies, axis units
TITLE_H = 0.062     # space the bold box title occupies
PAD = 0.030         # padding inside a box, top and bottom


def count_rows(filename):
    """True CSV row count, or None when the file is not on this machine.

    Counts records through csv.reader rather than counting newlines, because
    the verified text fields hold embedded newlines and a naive line count
    over-reports by an order of magnitude.
    """
    path = os.path.join(PROCESSED, filename)
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8', errors='replace', newline='') as fh:
        return max(sum(1 for _ in csv.reader(fh)) - 1, 0)


def resolve(filename, readme_value, label, notes):
    """Read a count, fall back to the README, and report any disagreement."""
    found = count_rows(filename)
    if found is None:
        notes.append('%s: %s absent, using the README figure %s'
                     % (label, filename, format(readme_value, ',')))
        return readme_value, False
    if found != readme_value:
        notes.append('%s: data/processed says %s but the README says %s. '
                     'Drawing the value from disk; reconcile the two.'
                     % (label, format(found, ','), format(readme_value, ',')))
    return found, True


def box(ax, x, y, w, title, lines, fill, edge, dashed=False):
    """Draw a box sized to its own content, so text cannot overflow it."""
    h = TITLE_H + len(lines) * LINE_H + PAD
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle='round,pad=0.010,rounding_size=0.018',
        facecolor=fill, edgecolor=edge, linewidth=1.6,
        linestyle='--' if dashed else '-', zorder=2))
    ax.text(x + w / 2, y + h - PAD / 2, title, ha='center', va='top',
            fontsize=10.2, fontweight='bold', color=edge, zorder=3)
    for i, line in enumerate(lines):
        ax.text(x + w / 2, y + h - TITLE_H - i * LINE_H, line, ha='center',
                va='top', fontsize=8.2, color='#2c3e50', zorder=3)
    return h


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=14,
        linewidth=1.4, color='#5d6d7e', zorder=1))


def draw(notes):
    topics, t_disk = resolve('lda_topic_assignments.csv',
                             README_TOPIC_ASSIGNMENTS, 'Stage 1', notes)
    verified, v_disk = resolve('gpt4_verified.csv',
                               README_VERIFIED, 'Stage 2', notes)

    fig, ax = plt.subplots(figsize=(13.0, 8.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    ax.text(0.5, 0.982, 'HyDMIS: hybrid three-stage disinformation detection',
            ha='center', va='top', fontsize=13.5, fontweight='bold',
            color='#1b2631')
    ax.text(0.5, 0.945,
            'Low-resource language performance as the primary evaluation '
            'target, not an aggregate metric',
            ha='center', va='top', fontsize=9.0, color='#566573')

    # --- corpus panel, full width, sources laid out inside it on two rows ---
    panel_y, panel_h = 0.700, 0.205
    ax.add_patch(FancyBboxPatch(
        (0.035, panel_y), 0.930, panel_h,
        boxstyle='round,pad=0.010,rounding_size=0.018',
        facecolor=PANEL_FILL, edgecolor=PANEL_EDGE, linewidth=1.6, zorder=2))
    ax.text(0.5, panel_y + panel_h - 0.014,
            'Corpus: %s records confirmed across six public sources, seven '
            'languages' % format(README_CORPUS, ','),
            ha='center', va='top', fontsize=10.2, fontweight='bold',
            color=PANEL_EDGE, zorder=3)

    # Three columns, two rows, left-aligned at generous spacing so no two
    # entries can collide the way they did in the first version.
    col_x = [0.075, 0.385, 0.700]
    row_y = [panel_y + panel_h - 0.072, panel_y + panel_h - 0.118]
    for i, (name, n, langs) in enumerate(SOURCES):
        ax.text(col_x[i % 3], row_y[i // 3], '%s  %s' % (name, n),
                ha='left', va='center', fontsize=8.4, color='#2c3e50',
                fontweight='medium', zorder=3)
        ax.text(col_x[i % 3], row_y[i // 3] - 0.026, langs,
                ha='left', va='center', fontsize=7.4, color='#7f8c8d',
                zorder=3)
    ax.text(0.5, panel_y + 0.016,
            'MultiClaim and ClimateMiSt are not obtained and are excluded',
            ha='center', va='bottom', fontsize=7.6, style='italic',
            color='#7f8c8d', zorder=3)

    arrow(ax, 0.170, 0.698, 0.170, 0.632)

    # --- three stages ---
    stage_y, stage_w = 0.360, 0.284
    h1 = box(ax, 0.035, stage_y, stage_w, 'Stage 1: LDA topic modeling',
             ['unsupervised, needs no labels',
              '%s records assigned a topic' % format(topics, ','),
              'topic count chosen by coherence',
              'ablation: topic alone adds only',
              '3.6 points over the baseline'],
             DONE_FILL, DONE_EDGE)

    box(ax, 0.358, stage_y, stage_w, 'Stage 2: GPT-4 verification',
        ['semantic veracity judgement',
         '%s records verified, 0 errors' % format(verified, ','),
         'handles code-switching and',
         'cultural context',
         'carries the discriminative signal'],
        DONE_FILL, DONE_EDGE)

    box(ax, 0.681, stage_y, stage_w, 'Stage 3: cross-lingual classifier',
        ['mBERT, XLM-R, RemBERT, Mistral',
         'community-weighted loss',
         'all three backbones to be ablated',
         'scoped and scripted,',
         'NOT YET RUN'],
        NOTRUN_FILL, NOTRUN_EDGE, dashed=True)

    mid = stage_y + h1 / 2
    arrow(ax, 0.321, mid, 0.356, mid)
    arrow(ax, 0.644, mid, 0.679, mid)

    ax.text(0.823, stage_y - 0.018,
            'dashed because no training run exists yet',
            ha='center', va='top', fontsize=7.4, style='italic',
            color='#95a5a6')

    # --- evaluation ---
    box(ax, 0.185, 0.085, 0.630, 'Evaluation, planned',
        ['F1 macro and weighted, precision and recall per class',
         'accuracy reported separately by language resource level',
         'rather than as a single multilingual aggregate'],
        PANEL_FILL, PANEL_EDGE)
    arrow(ax, 0.500, 0.355, 0.500, 0.300)

    src = ('record counts read from data/processed'
           if (t_disk or v_disk)
           else 'record counts from the README; processed files absent here')
    ax.text(0.5, 0.030, src, ha='center', va='bottom', fontsize=7.2,
            color='#95a5a6')

    out = os.path.join(FIGURES_DIR, 'hydmis_architecture.png')
    plt.savefig(out, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    return out


def main():
    print('HyDMIS pipeline architecture diagram')
    print('=' * 58)
    notes = []
    out = draw(notes)
    for n in notes:
        print('  NOTE  %s' % n)
    print('  Stage 3 drawn as not-yet-run, by design')
    print('  saved %s at %d dpi' % (os.path.relpath(out, REPO_ROOT), DPI))
    return 0


if __name__ == '__main__':
    sys.exit(main())
