"""
HyDMIS, Repository Consistency Checks
Phase 4, pre-commit gate

Follows the same idea as the checkers in fape-fairness-ml and oracle-rag-pipeline:
every check here exists because the corresponding defect was actually found in
this repository, so each one is a regression test rather than a hypothetical.

Two severities. A correctness problem means something in the repository states
an untruth, and it exits non-zero. A pending item means work that is not done
yet, which is reported but only fatal under --submission.

Run before every commit:
    python3 src/check_consistency.py

Wire it in with a symlink:
    ln -s ../../../tools/pre-commit .git/hooks/pre-commit
"""

import os
import re
import sys
import json
import glob

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The confirmed corpus, as the README's own summary line states it. The 562K
# figure that appears in older planning notes depends on a Mistral-7B
# pseudo-labeling step that has never run, and MultiClaim's 234,000 records
# depend on a Zenodo access request that is still open.
CONFIRMED_RECORDS = '324,292'
CONFIRMED_DATASETS = 6
PENDING_DATASETS = ('MultiClaim', 'ClimateMiSt')

# Languages actually present across the six confirmed corpora: LIAR2,
# TruthSeeker and FakeNewsNet are English, Covid-misinfo is EN/PT/ID,
# NewsPolyML is EN/DE/ES/FR/IT, DeFaktS is German. Seven distinct languages.
# Written out because a downstream description claimed four for months, which
# understated the corpus.
CONFIRMED_LANGUAGES = 7

# Prose files. Data CSVs are excluded on purpose and must stay excluded:
# they hold gpt-4o-mini's own generated text, and editing a dash out of
# recorded model output would falsify the data.
PROSE_EXTS = ('.md', '.py', '.txt', '.ipynb')
PROSE_EXTRA = ('requirements.txt', '.gitignore')
SKIP_DIRS = {'.git', 'venv', '.ipynb_checkpoints', '__pycache__',
             '.code-review-graph', '.pytest_cache'}

# Characters that do not belong in this repository's prose. Arrows are
# deliberately absent from this set: HyDMIS uses them as pipeline notation
# ("LDA (Stage 1) -> GPT-4") and as evolution chains in the literature
# analysis, which is meaningful rather than decorative, and the same
# convention holds in fape-fairness-ml's results tables.
# Built with chr() rather than written literally, so this file contains no
# non-ASCII bytes of its own for check 1 to trip over.
BANNED = {
    'em dash': chr(0x2014),
    'en dash': chr(0x2013),
    'non-breaking space': chr(0x00a0),
    'curly apostrophe': chr(0x2019),
    'curly open quote': chr(0x201c),
    'curly close quote': chr(0x201d),
}

problems = []   # correctness: the repository says something untrue
pending = []    # work not finished yet


def prose_files():
    for root, dirs, files in os.walk(REPO_ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fn in files:
            path = os.path.join(root, fn)
            rel = os.path.relpath(path, REPO_ROOT)
            if rel.startswith('data' + os.sep) or rel.endswith('.csv'):
                continue
            # The checker names the strings it hunts for, so it must not scan
            # itself; otherwise check 1 and check 2 both fire on this file.
            if os.path.abspath(path) == os.path.abspath(__file__):
                continue
            if fn.endswith(PROSE_EXTS) or fn in PROSE_EXTRA:
                yield rel, path


def read(rel):
    with open(os.path.join(REPO_ROOT, rel), encoding='utf-8') as fh:
        return fh.read()


def check_1_typography():
    """requirements.txt carried four em dashes in its section comments until
    Sep 27 2026, the same defect oracle-rag-pipeline found in round 4, and
    methodology_decisions.md carried an en dash inside a citation page range.
    Neither was caught because this repository had no gate at all."""
    for rel, path in prose_files():
        try:
            text = open(path, encoding='utf-8').read()
        except (UnicodeDecodeError, OSError):
            continue
        for name, char in BANNED.items():
            if char in text:
                problems.append('typography: %s contains %d %s' % (
                    rel, text.count(char), name))


def check_2_record_count():
    """The confirmed total is 324,292. A figure of 387K circulated for months
    and matched nothing here: not the confirmed total, not the component sums,
    not the 562K full dataset."""
    readme = read('README.md')
    if CONFIRMED_RECORDS not in readme:
        problems.append('record count: README.md no longer states the '
                        'confirmed total %s' % CONFIRMED_RECORDS)
    # 562K and 387K must never appear as a current-state claim.
    for bad in ('387K', '387,'):
        for rel, path in prose_files():
            if rel.endswith('.ipynb'):
                continue
            try:
                text = open(path, encoding='utf-8').read()
            except (UnicodeDecodeError, OSError):
                continue
            if bad in text:
                problems.append('record count: %s claims %s, which traces to '
                                'no figure in this repository' % (rel, bad))


def check_3_pending_datasets_marked():
    """MultiClaim and ClimateMiSt are not obtained. Wherever they appear they
    must be marked pending, or the corpus reads as larger than it is."""
    readme = read('README.md')
    for name in PENDING_DATASETS:
        if name not in readme:
            continue
        window = readme[max(0, readme.find(name) - 200):readme.find(name) + 200]
        if 'ending' not in window and 'equest' not in window:
            problems.append('pending datasets: %s appears in README.md '
                            'without being marked pending' % name)


def check_4_stage3_not_claimed_as_run():
    """The defect this check exists for reached downstream descriptions of this
    work before anyone noticed. Stage 3, the mBERT/XLM-R/RemBERT/Mistral
    classification with community-weighted loss, is scoped and scripted and
    has NEVER RUN. No script in src/ contains a training loop for it. So the
    repository must not describe classification accuracy, F1 by language
    resource level, or the community-weighted loss as achieved results.
    Delete this check only when Stage 3 has actually run."""
    trainers = []
    for path in glob.glob(os.path.join(REPO_ROOT, 'src', '*.py')):
        if os.path.basename(path) == os.path.basename(__file__):
            continue
        text = open(path, encoding='utf-8').read()
        if re.search(r'loss\.backward|Trainer\(|\.fit\(|optimizer\.step', text):
            trainers.append(os.path.relpath(path, REPO_ROOT))
    stage3_has_run = bool([t for t in trainers if 'lda' not in t.lower()])

    if not stage3_has_run:
        readme = read('README.md')
        if 'not yet run' not in readme:
            problems.append('stage 3: no training loop exists in src/, so '
                            'README.md must still say the classification '
                            'stage is "not yet run"')
        # A claim of finished cross-lingual accuracy would be untrue.
        for rel, _ in prose_files():
            if not rel.endswith('.md'):
                continue
            text = read(rel)
            for m in re.finditer(r'(?i)(we (?:find|show|achieve)|our results show|'
                                 r'accuracy drops hardest|F1 of \d)', text):
                line = text[:m.start()].count('\n') + 1
                problems.append('stage 3: %s:%d reads as a finished result '
                                '("%s") while Stage 3 has not run'
                                % (rel, line, m.group(0)))
    else:
        pending.append('stage 3: a training loop now exists in %s, so check 4 '
                       'should be retired and the results claims re-enabled'
                       % ', '.join(trainers))


def check_5_language_count():
    """Four languages were claimed downstream when the six confirmed corpora
    carry seven: English, German, Spanish, French, Italian, Portuguese,
    Indonesian.
    The README's 15+ figure depends on MultiClaim's 39, which is not obtained."""
    readme = read('README.md')
    for m in re.finditer(r'(\d+)\+?\s+languages', readme):
        claimed = int(m.group(1))
        window = readme[max(0, m.start() - 250):m.end() + 250]
        if claimed > CONFIRMED_LANGUAGES and 'ending' not in window \
                and 'MultiClaim' not in window:
            line = readme[:m.start()].count('\n') + 1
            problems.append('languages: README.md:%d claims %d languages '
                            'against %d confirmed, with nothing nearby '
                            'marking the rest as pending'
                            % (line, claimed, CONFIRMED_LANGUAGES))


def check_6_scripts_compile():
    """Cheap, and it has caught a syntax error mid-refactor before."""
    import py_compile
    for path in glob.glob(os.path.join(REPO_ROOT, 'src', '*.py')):
        try:
            py_compile.compile(path, doraise=True)
        except py_compile.PyCompileError as exc:
            problems.append('compile: %s does not compile (%s)'
                            % (os.path.relpath(path, REPO_ROOT), exc.msg.strip()))


def check_7_notebooks_valid():
    """A truncated write leaves a notebook that opens nowhere."""
    for path in glob.glob(os.path.join(REPO_ROOT, 'notebooks', '*.ipynb')):
        try:
            json.load(open(path, encoding='utf-8'))
        except (json.JSONDecodeError, OSError) as exc:
            problems.append('notebook: %s is not valid JSON (%s)'
                            % (os.path.relpath(path, REPO_ROOT), exc))


def check_8_figure_orphans():
    """A figure drawn by nothing cannot be reproduced. Reported rather than
    fatal, since exploratory plots legitimately accumulate during EDA."""
    drawn = set()
    for path in (glob.glob(os.path.join(REPO_ROOT, 'src', '*.py'))
                 + glob.glob(os.path.join(REPO_ROOT, 'notebooks', '*.ipynb'))):
        text = open(path, encoding='utf-8', errors='replace').read()
        drawn.update(re.findall(r"([A-Za-z0-9_\-]+\.png)", text))
    orphans = []
    for path in glob.glob(os.path.join(REPO_ROOT, 'figures', '**', '*.png'),
                          recursive=True):
        if os.path.basename(path) not in drawn:
            orphans.append(os.path.relpath(path, REPO_ROOT))
    if orphans:
        pending.append('figures: %d of the committed figures are named by no '
                       'script or notebook, first few: %s'
                       % (len(orphans), ', '.join(sorted(orphans)[:3])))


CHECKS = [
    check_1_typography,
    check_2_record_count,
    check_3_pending_datasets_marked,
    check_4_stage3_not_claimed_as_run,
    check_5_language_count,
    check_6_scripts_compile,
    check_7_notebooks_valid,
    check_8_figure_orphans,
]


def main():
    submission = '--submission' in sys.argv
    print('HyDMIS consistency checks')
    print('=' * 62)
    for check in CHECKS:
        check()

    if problems:
        print('\n%d correctness problem(s):\n' % len(problems))
        for p in problems:
            print('  - %s' % p)
    if pending:
        print('\n%d item(s) pending:\n' % len(pending))
        for p in pending:
            print('  - %s' % p)
    if not problems and not pending:
        print('\nEverything matches.')

    print('\n%d checks: %s.%s' % (
        len(CHECKS),
        'no correctness problems' if not problems
        else '%d correctness problem(s)' % len(problems),
        ' %d pending.' % len(pending) if pending else ''))

    if problems or (submission and pending):
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
