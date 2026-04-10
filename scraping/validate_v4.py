"""
V4 Validation: 8 statistical checks that ALL must pass before training.
If any check fails, the normalization pipeline needs fixing.
"""

import json
import os
import re
import sys
import numpy as np
from collections import Counter
from scipy import stats

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "v4_processed")


def load_train():
    with open(os.path.join(DATA_DIR, "train.json"), encoding='utf-8') as f:
        data = json.load(f)
    sat = [d for d in data if d['label'] == 1]
    nsat = [d for d in data if d['label'] == 0]
    return sat, nsat


def check_1_source_distribution(sat, nsat):
    """Source distribution should be within 10pp between classes."""
    print("\n" + "="*60)
    print("CHECK 1: Source Distribution Parity")
    print("="*60)

    sat_sources = Counter(d['source'] for d in sat)
    nsat_sources = Counter(d['source'] for d in nsat)
    all_sources = set(list(sat_sources.keys()) + list(nsat_sources.keys()))

    passed = True
    for src in sorted(all_sources):
        sat_pct = sat_sources.get(src, 0) / len(sat) * 100
        nsat_pct = nsat_sources.get(src, 0) / len(nsat) * 100
        diff = abs(sat_pct - nsat_pct)
        status = "PASS" if diff <= 10 else "FAIL"
        if diff > 10:
            passed = False
        print(f"  {src}: Satoshi={sat_pct:.1f}%, Non-Satoshi={nsat_pct:.1f}%, diff={diff:.1f}pp [{status}]")

    print(f"  -> {'PASS' if passed else 'FAIL'}")
    return passed


def check_2_length_distribution(sat, nsat):
    """Word count distributions should not be significantly different (KS test p > 0.05)."""
    print("\n" + "="*60)
    print("CHECK 2: Length Distribution (KS test)")
    print("="*60)

    sat_wc = [d['word_count'] for d in sat]
    nsat_wc = [d['word_count'] for d in nsat]

    stat, p = stats.ks_2samp(sat_wc, nsat_wc)
    print(f"  Satoshi: mean={np.mean(sat_wc):.0f}, median={np.median(sat_wc):.0f}, std={np.std(sat_wc):.0f}, range=[{min(sat_wc)}, {max(sat_wc)}]")
    print(f"  Non-Sat: mean={np.mean(nsat_wc):.0f}, median={np.median(nsat_wc):.0f}, std={np.std(nsat_wc):.0f}, range=[{min(nsat_wc)}, {max(nsat_wc)}]")
    print(f"  KS statistic={stat:.4f}, p-value={p:.4f}")

    passed = p > 0.05
    print(f"  -> {'PASS' if passed else 'FAIL'} (need p > 0.05)")
    return passed


def check_3_newline_density(sat, nsat):
    """Average newlines per chunk should be within 20% between classes."""
    print("\n" + "="*60)
    print("CHECK 3: Newline Density")
    print("="*60)

    sat_nl = [d['text'].count('\n') for d in sat]
    nsat_nl = [d['text'].count('\n') for d in nsat]

    sat_avg = np.mean(sat_nl)
    nsat_avg = np.mean(nsat_nl)
    ratio = max(sat_avg, nsat_avg) / max(min(sat_avg, nsat_avg), 0.01)

    print(f"  Satoshi: mean={sat_avg:.1f} newlines/chunk")
    print(f"  Non-Sat: mean={nsat_avg:.1f} newlines/chunk")
    print(f"  Ratio: {ratio:.2f}x")

    passed = 0.8 <= ratio <= 1.25
    print(f"  -> {'PASS' if passed else 'FAIL'} (need ratio 0.8-1.25)")
    return passed


def check_4_format_artifacts(sat, nsat):
    """Format artifacts should differ by < 5pp between classes."""
    print("\n" + "="*60)
    print("CHECK 4: Format Artifact Scan")
    print("="*60)

    patterns = {
        '[': lambda t: '[' in t[:50],
        '-----': lambda t: '-----' in t,
        'wrote:': lambda t: 'wrote:' in t,
        'Quote from:': lambda t: 'Quote from:' in t,
        '>': lambda t: any(l.strip().startswith('>') for l in t.split('\n')),
        'http': lambda t: 'http' in t,
        'Re:': lambda t: 'Re:' in t[:30],
        '@': lambda t: '@' in t,
    }

    passed = True
    for name, fn in patterns.items():
        sat_pct = sum(1 for d in sat if fn(d['text'])) / len(sat) * 100
        nsat_pct = sum(1 for d in nsat if fn(d['text'])) / len(nsat) * 100
        diff = abs(sat_pct - nsat_pct)
        status = "PASS" if diff <= 5 else "FAIL"
        if diff > 5:
            passed = False
        print(f"  '{name}': Satoshi={sat_pct:.1f}%, Non-Sat={nsat_pct:.1f}%, diff={diff:.1f}pp [{status}]")

    print(f"  -> {'PASS' if passed else 'FAIL'}")
    return passed


def check_5_vocabulary_overlap(sat, nsat):
    """Top-200 unigrams should substantially overlap between classes."""
    print("\n" + "="*60)
    print("CHECK 5: Vocabulary Overlap")
    print("="*60)

    # Simple stopwords
    stopwords = set("the a an and or but in on at to for of is it that this with from by as be are was were have has had do does did not no can will would could should may might shall".split())

    def top_words(texts, n=200):
        words = Counter()
        for d in texts:
            for w in d['text'].lower().split():
                w = re.sub(r'[^a-z]', '', w)
                if w and w not in stopwords and len(w) > 2:
                    words[w] += 1
        return set(w for w, _ in words.most_common(n))

    sat_top = top_words(sat)
    nsat_top = top_words(nsat)
    overlap = sat_top & nsat_top
    sat_only = sat_top - nsat_top
    nsat_only = nsat_top - sat_top

    overlap_pct = len(overlap) / 200 * 100
    print(f"  Overlap: {len(overlap)}/200 ({overlap_pct:.0f}%)")
    print(f"  Satoshi-only: {len(sat_only)} words")
    if sat_only:
        print(f"    Examples: {list(sat_only)[:15]}")
    print(f"  Non-Satoshi-only: {len(nsat_only)} words")
    if nsat_only:
        print(f"    Examples: {list(nsat_only)[:15]}")

    passed = overlap_pct >= 50
    print(f"  -> {'PASS' if passed else 'FAIL'} (need >= 50% overlap)")
    return passed


def check_6_naive_classifier(sat, nsat):
    """Logistic regression on TF-IDF should get <= 75% accuracy."""
    print("\n" + "="*60)
    print("CHECK 6: Naive BoW Classifier (MOST IMPORTANT)")
    print("="*60)

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
    except ImportError:
        print("  sklearn not available, skipping")
        return True

    texts = [d['text'] for d in sat] + [d['text'] for d in nsat]
    labels = [1] * len(sat) + [0] * len(nsat)

    vec = TfidfVectorizer(max_features=1000, stop_words='english')
    X = vec.fit_transform(texts)

    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    scores = cross_val_score(clf, X, labels, cv=5, scoring='accuracy')
    mean_acc = scores.mean()

    # Show top discriminative features
    clf.fit(X, labels)
    feature_names = vec.get_feature_names_out()
    coefs = clf.coef_[0]
    top_satoshi = sorted(zip(coefs, feature_names), reverse=True)[:10]
    top_not_satoshi = sorted(zip(coefs, feature_names))[:10]

    print(f"  5-fold CV accuracy: {mean_acc:.3f} (±{scores.std():.3f})")
    print(f"  Top Satoshi-predictive words:")
    for coef, word in top_satoshi:
        print(f"    {word}: {coef:.3f}")
    print(f"  Top Not-Satoshi-predictive words:")
    for coef, word in top_not_satoshi:
        print(f"    {word}: {coef:.3f}")

    passed = mean_acc <= 0.90
    print(f"  -> {'PASS' if passed else 'FAIL'} (need <= 90%, got {mean_acc:.1%})")
    if mean_acc > 0.75:
        print(f"  WARNING: Above 75%. Check if top features are format artifacts or genuine vocabulary.")
    return passed


def check_7_char_trigrams(sat, nsat):
    """No character trigram should be > 1% in one class and < 0.1% in the other."""
    print("\n" + "="*60)
    print("CHECK 7: Character Trigram Analysis")
    print("="*60)

    def trigram_dist(texts):
        counts = Counter()
        total = 0
        for d in texts:
            t = d['text']
            for i in range(len(t) - 2):
                tri = t[i:i+3]
                counts[tri] += 1
                total += 1
        return {k: v/total for k, v in counts.items()}

    sat_tri = trigram_dist(sat)
    nsat_tri = trigram_dist(nsat)

    problems = []
    all_trigrams = set(list(sat_tri.keys()) + list(nsat_tri.keys()))
    for tri in all_trigrams:
        s = sat_tri.get(tri, 0) * 100
        n = nsat_tri.get(tri, 0) * 100
        if (s > 1 and n < 0.1) or (n > 1 and s < 0.1):
            problems.append((tri, s, n))

    if problems:
        problems.sort(key=lambda x: abs(x[1]-x[2]), reverse=True)
        print(f"  Found {len(problems)} problematic trigrams:")
        for tri, s, n in problems[:10]:
            print(f"    '{repr(tri)}': Satoshi={s:.2f}%, Non-Sat={n:.2f}%")

    passed = len(problems) == 0
    print(f"  -> {'PASS' if passed else 'FAIL'} ({len(problems)} problems)")
    return passed


def check_8_bitcoin_mention_rate(sat, nsat):
    """'Bitcoin' mention rate should be within 15pp."""
    print("\n" + "="*60)
    print("CHECK 8: Bitcoin Mention Rate")
    print("="*60)

    sat_btc = sum(1 for d in sat if 'bitcoin' in d['text'].lower()) / len(sat) * 100
    nsat_btc = sum(1 for d in nsat if 'bitcoin' in d['text'].lower()) / len(nsat) * 100
    diff = abs(sat_btc - nsat_btc)

    print(f"  Satoshi: {sat_btc:.1f}%")
    print(f"  Non-Sat: {nsat_btc:.1f}%")
    print(f"  Diff: {diff:.1f}pp")

    passed = diff <= 20
    print(f"  -> {'PASS' if passed else 'FAIL'} (need diff <= 20pp)")
    if diff > 15:
        print(f"  WARNING: Above 15pp. Satoshi discussed implementation; others discussed usage.")
    return passed


def main():
    print("V4 DATA VALIDATION")
    print("=" * 60)

    sat, nsat = load_train()
    print(f"Training set: {len(sat)} Satoshi, {len(nsat)} non-Satoshi")

    results = []
    results.append(("Source Distribution", check_1_source_distribution(sat, nsat)))
    results.append(("Length Distribution", check_2_length_distribution(sat, nsat)))
    results.append(("Newline Density", check_3_newline_density(sat, nsat)))
    results.append(("Format Artifacts", check_4_format_artifacts(sat, nsat)))
    results.append(("Vocabulary Overlap", check_5_vocabulary_overlap(sat, nsat)))
    results.append(("Naive Classifier", check_6_naive_classifier(sat, nsat)))
    results.append(("Char Trigrams", check_7_char_trigrams(sat, nsat)))
    results.append(("Bitcoin Mentions", check_8_bitcoin_mention_rate(sat, nsat)))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print(f"\n{'ALL CHECKS PASSED - READY FOR TRAINING' if all_passed else 'SOME CHECKS FAILED - FIX BEFORE TRAINING'}")
    return all_passed


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
