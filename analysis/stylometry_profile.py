"""
Satoshi Stylometry Analysis

Step 1: Profile Satoshi's writing fingerprint across multiple dimensions
Step 2: Build a classifier using the most discriminative features
Step 3: Evaluate on held-out data

Uses:
- Function word frequencies (Burrows Delta)
- Character n-grams
- Punctuation profile
- Sentence-level statistics
- pybiber for Biber's linguistic dimensions
"""

import json
import os
import re
import sys
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "v5_clean")

# ============================================================
# FEATURE EXTRACTION
# ============================================================

# Common English function words (topic-independent)
FUNCTION_WORDS = [
    'the', 'a', 'an', 'and', 'or', 'but', 'if', 'then', 'that', 'which',
    'who', 'whom', 'this', 'these', 'those', 'is', 'are', 'was', 'were',
    'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
    'will', 'would', 'could', 'should', 'may', 'might', 'shall', 'can',
    'not', 'no', 'nor', 'so', 'yet', 'also', 'just', 'only', 'very',
    'still', 'already', 'even', 'now', 'then', 'here', 'there', 'where',
    'when', 'how', 'what', 'why', 'all', 'each', 'every', 'both',
    'few', 'more', 'most', 'other', 'some', 'such', 'than', 'too',
    'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'to',
    'into', 'through', 'about', 'after', 'before', 'between', 'under',
    'above', 'up', 'down', 'out', 'off', 'over', 'again', 'further',
    'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
    'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their',
    'myself', 'yourself', 'himself', 'itself', 'ourselves', 'themselves',
]


def extract_handcrafted_features(text):
    """Extract a vector of stylometric features from a single text."""
    words = text.split()
    word_count = len(words)
    if word_count == 0:
        return {}

    words_lower = [w.lower() for w in words]
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    features = {}

    # --- FUNCTION WORD FREQUENCIES ---
    word_freq = Counter(words_lower)
    for fw in FUNCTION_WORDS:
        features[f'fw_{fw}'] = word_freq.get(fw, 0) / word_count

    # --- PUNCTUATION PROFILE ---
    features['punct_comma_rate'] = text.count(',') / word_count
    features['punct_semicolon_rate'] = text.count(';') / word_count
    features['punct_colon_rate'] = text.count(':') / word_count
    features['punct_exclamation_rate'] = text.count('!') / word_count
    features['punct_question_rate'] = text.count('?') / word_count
    features['punct_dash_rate'] = text.count('-') / word_count
    features['punct_paren_rate'] = (text.count('(') + text.count(')')) / word_count
    features['punct_quote_rate'] = (text.count('"') + text.count("'")) / word_count
    features['punct_ellipsis_rate'] = text.count('...') / word_count

    # --- SENTENCE STATISTICS ---
    if sentences:
        sent_lengths = [len(s.split()) for s in sentences]
        features['sent_mean_length'] = np.mean(sent_lengths)
        features['sent_std_length'] = np.std(sent_lengths) if len(sent_lengths) > 1 else 0
        features['sent_max_length'] = max(sent_lengths)
        features['sent_min_length'] = min(sent_lengths) if sent_lengths else 0
        features['sent_count'] = len(sentences)
    else:
        features['sent_mean_length'] = 0
        features['sent_std_length'] = 0
        features['sent_max_length'] = 0
        features['sent_min_length'] = 0
        features['sent_count'] = 0

    # --- WORD LENGTH STATISTICS ---
    word_lengths = [len(w) for w in words]
    features['word_mean_length'] = np.mean(word_lengths)
    features['word_std_length'] = np.std(word_lengths)
    features['word_long_rate'] = sum(1 for l in word_lengths if l > 6) / word_count
    features['word_short_rate'] = sum(1 for l in word_lengths if l <= 3) / word_count

    # --- VOCABULARY RICHNESS ---
    unique_words = set(words_lower)
    features['vocab_type_token_ratio'] = len(unique_words) / word_count
    # Hapax legomena (words used exactly once)
    features['vocab_hapax_ratio'] = sum(1 for w, c in word_freq.items() if c == 1) / word_count

    # --- CONTRACTION PATTERNS ---
    contractions = re.findall(r"\w+n't|\w+'[a-z]{1,2}", text.lower())
    features['contraction_rate'] = len(contractions) / word_count

    # --- CAPITALIZATION ---
    features['cap_all_caps_rate'] = sum(1 for w in words if w.isupper() and len(w) > 1) / word_count
    features['cap_initial_caps_rate'] = sum(1 for w in words if w[0].isupper() and not w.isupper()) / word_count

    # --- PARAGRAPH STRUCTURE ---
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    features['para_count'] = len(paragraphs)
    features['para_mean_length'] = np.mean([len(p.split()) for p in paragraphs]) if paragraphs else 0

    # --- HEDGING / CERTAINTY MARKERS ---
    hedge_words = ['think', 'believe', 'probably', 'maybe', 'perhaps', 'seems',
                   'likely', 'unlikely', 'might', 'could', 'would', 'should',
                   'guess', 'suppose', 'assume', 'possible', 'possibly']
    certainty_words = ['definitely', 'certainly', 'always', 'never', 'must',
                       'obviously', 'clearly', 'exactly', 'absolutely']
    features['hedge_rate'] = sum(word_freq.get(w, 0) for w in hedge_words) / word_count
    features['certainty_rate'] = sum(word_freq.get(w, 0) for w in certainty_words) / word_count

    # --- FIRST PERSON USAGE ---
    features['first_person_rate'] = sum(word_freq.get(w, 0) for w in ['i', 'me', 'my', 'mine', 'myself']) / word_count
    features['first_person_plural_rate'] = sum(word_freq.get(w, 0) for w in ['we', 'us', 'our', 'ours', 'ourselves']) / word_count
    features['second_person_rate'] = sum(word_freq.get(w, 0) for w in ['you', 'your', 'yours', 'yourself']) / word_count

    return features


def extract_char_ngrams(texts, n=3, max_features=200):
    """Extract character n-gram features."""
    vec = TfidfVectorizer(analyzer='char', ngram_range=(n, n), max_features=max_features)
    X = vec.fit_transform(texts)
    return X, vec.get_feature_names_out()


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():
    # Load data
    with open(os.path.join(DATA_DIR, "satoshi_clean.json"), encoding='utf-8') as f:
        sat_all = json.load(f)
    with open(os.path.join(DATA_DIR, "non_satoshi_clean.json"), encoding='utf-8') as f:
        nsat_all = json.load(f)

    # Filter to texts >= 50 words (need enough text for features)
    sat = [d for d in sat_all if d['word_count'] >= 50]
    nsat = [d for d in nsat_all if d['word_count'] >= 50]

    print(f"Satoshi texts (>= 50 words): {len(sat)}")
    print(f"Non-Satoshi texts (>= 50 words): {len(nsat)}")

    # ============================================================
    # PART 1: PROFILE SATOSHI'S WRITING
    # ============================================================
    print("\n" + "="*60)
    print("PART 1: SATOSHI'S WRITING FINGERPRINT")
    print("="*60)

    # Extract features for all texts
    sat_features = [extract_handcrafted_features(d['text']) for d in sat]
    nsat_features = [extract_handcrafted_features(d['text']) for d in nsat]

    # Compare means across key features
    feature_names = sorted(sat_features[0].keys())
    print(f"\nFeatures with largest differences between Satoshi and others:\n")
    print(f"{'Feature':<35s} {'Satoshi':>10s} {'Others':>10s} {'Diff':>10s} {'Ratio':>8s}")
    print("-" * 75)

    diffs = []
    for feat in feature_names:
        sat_vals = [f[feat] for f in sat_features]
        nsat_vals = [f[feat] for f in nsat_features]
        sat_mean = np.mean(sat_vals)
        nsat_mean = np.mean(nsat_vals)
        diff = sat_mean - nsat_mean
        ratio = sat_mean / nsat_mean if nsat_mean > 0.0001 else float('inf')
        diffs.append((feat, sat_mean, nsat_mean, diff, ratio))

    # Sort by absolute difference
    diffs.sort(key=lambda x: abs(x[3]), reverse=True)
    for feat, sm, nm, d, r in diffs[:30]:
        print(f"{feat:<35s} {sm:>10.4f} {nm:>10.4f} {d:>+10.4f} {r:>8.2f}x")

    # ============================================================
    # PART 2: BUILD CLASSIFIERS
    # ============================================================
    print("\n" + "="*60)
    print("PART 2: CLASSIFIER EVALUATION")
    print("="*60)

    # Prepare data
    all_texts = [d['text'] for d in sat] + [d['text'] for d in nsat]
    all_labels = [1] * len(sat) + [0] * len(nsat)
    all_labels = np.array(all_labels)

    # Feature set 1: Handcrafted features
    print("\n--- Feature Set 1: Handcrafted Stylometric Features ---")
    all_hand_features = sat_features + nsat_features
    feature_matrix = np.array([[f[k] for k in feature_names] for f in all_hand_features])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, clf in [
        ("Logistic Regression", LogisticRegression(max_iter=1000, class_weight='balanced')),
        ("Random Forest", RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)),
        ("SVM (RBF)", SVC(kernel='rbf', class_weight='balanced', probability=True)),
        ("Gradient Boosting", GradientBoostingClassifier(n_estimators=200, random_state=42)),
    ]:
        pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
        scores = cross_val_score(pipe, feature_matrix, all_labels, cv=cv, scoring='f1')
        print(f"  {name:<25s} F1 = {scores.mean():.3f} (±{scores.std():.3f})")

    # Feature set 2: Character trigrams
    print("\n--- Feature Set 2: Character Trigrams (TF-IDF) ---")
    X_char, char_names = extract_char_ngrams(all_texts, n=3, max_features=300)

    for name, clf in [
        ("Logistic Regression", LogisticRegression(max_iter=1000, class_weight='balanced')),
        ("SVM (RBF)", SVC(kernel='rbf', class_weight='balanced')),
    ]:
        pipe = Pipeline([('scaler', StandardScaler(with_mean=False)), ('clf', clf)])
        scores = cross_val_score(pipe, X_char, all_labels, cv=cv, scoring='f1')
        print(f"  {name:<25s} F1 = {scores.mean():.3f} (±{scores.std():.3f})")

    # Feature set 3: Function words only (Burrows Delta style)
    print("\n--- Feature Set 3: Function Words Only ---")
    fw_indices = [feature_names.index(f'fw_{fw}') for fw in FUNCTION_WORDS if f'fw_{fw}' in feature_names]
    fw_matrix = feature_matrix[:, fw_indices]

    for name, clf in [
        ("Logistic Regression", LogisticRegression(max_iter=1000, class_weight='balanced')),
        ("SVM (RBF)", SVC(kernel='rbf', class_weight='balanced')),
    ]:
        pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
        scores = cross_val_score(pipe, fw_matrix, all_labels, cv=cv, scoring='f1')
        print(f"  {name:<25s} F1 = {scores.mean():.3f} (±{scores.std():.3f})")

    # Feature set 4: Combined (handcrafted + char trigrams)
    print("\n--- Feature Set 4: Combined (Handcrafted + Char Trigrams) ---")
    from scipy.sparse import hstack, csr_matrix
    X_combined = hstack([csr_matrix(feature_matrix), X_char])

    for name, clf in [
        ("Logistic Regression", LogisticRegression(max_iter=1000, class_weight='balanced')),
        ("SVM (RBF)", SVC(kernel='rbf', class_weight='balanced')),
        ("Gradient Boosting", GradientBoostingClassifier(n_estimators=200, random_state=42)),
    ]:
        pipe = Pipeline([('scaler', StandardScaler(with_mean=False)), ('clf', clf)])
        scores = cross_val_score(pipe, X_combined, all_labels, cv=cv, scoring='f1')
        print(f"  {name:<25s} F1 = {scores.mean():.3f} (±{scores.std():.3f})")

    # ============================================================
    # PART 3: BEST MODEL - FULL EVALUATION
    # ============================================================
    print("\n" + "="*60)
    print("PART 3: BEST MODEL EVALUATION")
    print("="*60)

    # Train on 80%, test on 20%
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        feature_matrix, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )

    best_clf = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', GradientBoostingClassifier(n_estimators=200, random_state=42))
    ])
    best_clf.fit(X_train, y_train)
    y_pred = best_clf.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['not_satoshi', 'satoshi']))
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  Predicted:  not_satoshi  satoshi")
    print(f"  not_satoshi:    {cm[0][0]:>5d}    {cm[0][1]:>5d}")
    print(f"  satoshi:        {cm[1][0]:>5d}    {cm[1][1]:>5d}")

    # Feature importance (from the gradient boosting model)
    if hasattr(best_clf.named_steps['clf'], 'feature_importances_'):
        importances = best_clf.named_steps['clf'].feature_importances_
        top_features = sorted(zip(importances, feature_names), reverse=True)[:20]
        print("\nTop 20 most important features:")
        for imp, name in top_features:
            print(f"  {name:<35s} {imp:.4f}")

    # ============================================================
    # PART 4: TEST ON KNOWN TEXTS
    # ============================================================
    print("\n" + "="*60)
    print("PART 4: TEST ON KNOWN TEXTS")
    print("="*60)

    # Train on ALL data for final model
    final_clf = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', GradientBoostingClassifier(n_estimators=200, random_state=42))
    ])
    final_clf.fit(feature_matrix, all_labels)

    test_texts = [
        ("Satoshi (zombie farms email)", "There would be many smaller zombie farms that are not big enough to overpower the network, and they could still make money by generating bitcoins. The smaller farms are then the honest nodes. The more smaller farms resort to generating bitcoins, the higher the bar gets to overpower the network, making larger farms also too small to overpower it so that they may as well generate bitcoins too. According to the long tail theory, the small, medium and merely large farms put together should add up to a lot more than the biggest zombie farm."),
        ("Satoshi (SHA-256)", "SHA-256 is very strong. It's not like the incremental step from MD5 to SHA1. It can last several decades unless there's some massive breakthrough attack. If SHA-256 became completely broken, I think we could come to some agreement about what the honest block chain was before the trouble started, lock that in and continue from there with a new hash function."),
        ("Satoshi (whitepaper intro)", "A purely peer-to-peer version of electronic cash would allow online payments to be sent directly from one party to another without going through a financial institution. Digital signatures provide part of the solution, but the main benefits are lost if a trusted third party is still required to prevent double-spending. We propose a solution to the double-spending problem using a peer-to-peer network."),
        ("Non-Satoshi (generic)", "I think the real question is whether Bitcoin can scale to handle millions of transactions per day. The current block size limit seems like it will be a problem eventually. We need to figure out a solution before it becomes critical."),
        ("Hal Finney", "Bitcoin seems to be a very promising idea. I like the idea of basing security on the assumption that the CPU power of honest participants outweighs that of the attacker. It is a very modern notion that exploits the power of the long tail."),
    ]

    for name, text in test_texts:
        features = extract_handcrafted_features(text)
        fv = np.array([[features[k] for k in feature_names]])
        prob = final_clf.predict_proba(fv)[0]
        print(f"  {name:<40s} -> Satoshi: {prob[1]:.3f}  Not-Satoshi: {prob[0]:.3f}")


if __name__ == "__main__":
    main()
