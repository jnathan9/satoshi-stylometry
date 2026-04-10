"""
Full Stylometry Classifier: Handcrafted features + Biber features + Character n-grams

Three feature families combined:
1. Handcrafted (function words, punctuation, sentence stats, contractions, etc.)
2. Biber's 67 linguistic dimensions (via pybiber)
3. Character n-grams (2-4 grams)
"""

import json
import os
import re
import sys
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.sparse import hstack, csr_matrix
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "v5_clean")

# ============================================================
# FEATURE EXTRACTION: HANDCRAFTED
# ============================================================

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
    'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
    'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their',
]

def extract_handcrafted(text):
    words = text.split()
    wc = len(words)
    if wc == 0: return {}
    words_lower = [w.lower() for w in words]
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    freq = Counter(words_lower)
    f = {}
    for fw in FUNCTION_WORDS:
        f[f'fw_{fw}'] = freq.get(fw, 0) / wc
    f['punct_comma'] = text.count(',') / wc
    f['punct_semicolon'] = text.count(';') / wc
    f['punct_colon'] = text.count(':') / wc
    f['punct_excl'] = text.count('!') / wc
    f['punct_quest'] = text.count('?') / wc
    f['punct_dash'] = text.count('-') / wc
    f['punct_paren'] = (text.count('(') + text.count(')')) / wc
    f['punct_quote'] = (text.count('"') + text.count("'")) / wc
    f['punct_ellipsis'] = text.count('...') / wc
    if sentences:
        sl = [len(s.split()) for s in sentences]
        f['sent_mean'] = np.mean(sl)
        f['sent_std'] = np.std(sl) if len(sl) > 1 else 0
        f['sent_count'] = len(sentences) / wc * 100  # per 100 words
    else:
        f['sent_mean'] = f['sent_std'] = f['sent_count'] = 0
    wl = [len(w) for w in words]
    f['word_mean_len'] = np.mean(wl)
    f['word_long_rate'] = sum(1 for l in wl if l > 6) / wc
    f['word_short_rate'] = sum(1 for l in wl if l <= 3) / wc
    f['vocab_ttr'] = len(set(words_lower)) / wc
    f['vocab_hapax'] = sum(1 for w, c in freq.items() if c == 1) / wc
    contractions = re.findall(r"\w+n't|\w+'[a-z]{1,2}", text.lower())
    f['contraction_rate'] = len(contractions) / wc
    paras = [p for p in text.split('\n\n') if p.strip()]
    f['para_count'] = len(paras) / wc * 100
    f['first_person'] = sum(freq.get(w, 0) for w in ['i', 'me', 'my', 'mine', 'myself']) / wc
    f['first_person_pl'] = sum(freq.get(w, 0) for w in ['we', 'us', 'our', 'ours']) / wc
    f['second_person'] = sum(freq.get(w, 0) for w in ['you', 'your', 'yours']) / wc
    hedge = ['think', 'believe', 'probably', 'maybe', 'perhaps', 'seems', 'likely', 'might', 'could', 'would', 'guess', 'suppose']
    f['hedge_rate'] = sum(freq.get(w, 0) for w in hedge) / wc
    certain = ['definitely', 'certainly', 'always', 'never', 'must', 'obviously', 'clearly']
    f['certainty_rate'] = sum(freq.get(w, 0) for w in certain) / wc
    return f


# ============================================================
# FEATURE EXTRACTION: BIBER
# ============================================================

def extract_biber_features(texts, doc_ids):
    """Extract Biber features for a list of texts using pybiber."""
    print("  Extracting Biber features (this takes a minute)...", flush=True)
    from pybiber import CorpusProcessor, biber
    import polars as pl
    import spacy

    nlp = spacy.load('en_core_web_sm')
    cp = CorpusProcessor()

    # Process in batches to avoid memory issues
    batch_size = 200
    all_results = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_ids = doc_ids[i:i+batch_size]

        df = pl.DataFrame({'doc_id': batch_ids, 'text': batch_texts})
        try:
            parsed = cp.process_corpus(df, nlp)
            result = biber(parsed)
            all_results.append(result)
        except Exception as e:
            print(f"  Batch {i//batch_size} error: {e}", flush=True)
            continue

        if (i // batch_size + 1) % 5 == 0:
            print(f"  Processed {i+batch_size}/{len(texts)} texts", flush=True)

    if not all_results:
        return None, []

    # Combine batches
    import polars as pl
    combined = pl.concat(all_results)

    # Convert to numpy, excluding doc_id
    feature_cols = [c for c in combined.columns if c != 'doc_id']
    matrix = combined.select(feature_cols).to_numpy().astype(float)

    print(f"  -> {matrix.shape[1]} Biber features extracted for {matrix.shape[0]} texts", flush=True)
    return matrix, feature_cols


# ============================================================
# MAIN
# ============================================================

def main():
    with open(os.path.join(DATA_DIR, "satoshi_clean.json"), encoding='utf-8') as f:
        sat_all = json.load(f)
    with open(os.path.join(DATA_DIR, "non_satoshi_clean.json"), encoding='utf-8') as f:
        nsat_all = json.load(f)

    # Filter to >= 50 words
    sat = [d for d in sat_all if d['word_count'] >= 50]
    nsat = [d for d in nsat_all if d['word_count'] >= 50]
    print(f"Satoshi: {len(sat)} texts | Non-Satoshi: {len(nsat)} texts")

    all_texts = [d['text'] for d in sat] + [d['text'] for d in nsat]
    all_labels = np.array([1]*len(sat) + [0]*len(nsat))
    doc_ids = [f"sat_{i}" for i in range(len(sat))] + [f"nsat_{i}" for i in range(len(nsat))]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ============================================================
    # Extract all feature families
    # ============================================================

    # 1. Handcrafted
    print("\nExtracting handcrafted features...", flush=True)
    hand_features = [extract_handcrafted(t) for t in all_texts]
    hand_names = sorted(hand_features[0].keys())
    X_hand = np.array([[f[k] for k in hand_names] for f in hand_features])
    print(f"  -> {X_hand.shape[1]} handcrafted features")

    # 2. Character n-grams (2-4)
    print("Extracting character n-grams...", flush=True)
    char_vec = TfidfVectorizer(analyzer='char', ngram_range=(2, 4), max_features=500)
    X_char = char_vec.fit_transform(all_texts)
    print(f"  -> {X_char.shape[1]} char n-gram features")

    # 3. Biber features
    biber_matrix, biber_names = extract_biber_features(all_texts, doc_ids)

    # ============================================================
    # Evaluate each feature family + combinations
    # ============================================================
    print("\n" + "="*60)
    print("CLASSIFIER COMPARISON (5-fold CV, F1 score)")
    print("="*60)

    feature_sets = {
        "Handcrafted only": csr_matrix(X_hand),
        "Char n-grams only": X_char,
    }

    if biber_matrix is not None and biber_matrix.shape[0] == len(all_texts):
        # Replace NaN with 0
        biber_clean = np.nan_to_num(biber_matrix, nan=0.0)
        feature_sets["Biber only"] = csr_matrix(biber_clean)
        feature_sets["Handcrafted + Biber"] = hstack([csr_matrix(X_hand), csr_matrix(biber_clean)])
        feature_sets["Handcrafted + Char n-grams"] = hstack([csr_matrix(X_hand), X_char])
        feature_sets["Biber + Char n-grams"] = hstack([csr_matrix(biber_clean), X_char])
        feature_sets["ALL COMBINED"] = hstack([csr_matrix(X_hand), csr_matrix(biber_clean), X_char])
    else:
        feature_sets["Handcrafted + Char n-grams"] = hstack([csr_matrix(X_hand), X_char])

    classifiers = [
        ("Logistic Regression", LogisticRegression(max_iter=2000, class_weight='balanced', C=1.0)),
        ("SVM (RBF)", SVC(kernel='rbf', class_weight='balanced', C=10.0, gamma='scale')),
        ("Gradient Boosting", GradientBoostingClassifier(n_estimators=300, max_depth=4, random_state=42)),
        ("Random Forest", RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42)),
    ]

    best_score = 0
    best_config = None

    for fs_name, X in feature_sets.items():
        print(f"\n--- {fs_name} ({X.shape[1]} features) ---")
        for clf_name, clf in classifiers:
            pipe = Pipeline([('scaler', StandardScaler(with_mean=False)), ('clf', clf)])
            scores = cross_val_score(pipe, X, all_labels, cv=cv, scoring='f1')
            mean_f1 = scores.mean()
            print(f"  {clf_name:<25s} F1 = {mean_f1:.3f} (±{scores.std():.3f})")
            if mean_f1 > best_score:
                best_score = mean_f1
                best_config = (fs_name, clf_name, clf, X)

    # ============================================================
    # Best model: full evaluation
    # ============================================================
    print(f"\n{'='*60}")
    print(f"BEST: {best_config[0]} + {best_config[1]} (F1={best_score:.3f})")
    print(f"{'='*60}")

    X_best = best_config[3]
    X_train, X_test, y_train, y_test = train_test_split(
        X_best, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )

    best_pipe = Pipeline([
        ('scaler', StandardScaler(with_mean=False)),
        ('clf', best_config[2])
    ])
    best_pipe.fit(X_train, y_train)
    y_pred = best_pipe.predict(X_test)

    print("\nTest Set Results:")
    print(classification_report(y_test, y_pred, target_names=['not_satoshi', 'satoshi']))
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:")
    print(f"  not_satoshi: {cm[0][0]:>5d} correct, {cm[0][1]:>3d} false positive")
    print(f"  satoshi:     {cm[1][0]:>5d} missed,  {cm[1][1]:>3d} correct")

    # ============================================================
    # Test on known texts
    # ============================================================
    print(f"\n{'='*60}")
    print("TEST ON KNOWN TEXTS")
    print(f"{'='*60}")

    # Re-fit on all data
    best_pipe.fit(X_best, all_labels)

    test_texts = [
        ("Satoshi (zombie farms email)", "There would be many smaller zombie farms that are not big enough to overpower the network, and they could still make money by generating bitcoins. The smaller farms are then the honest nodes. The more smaller farms resort to generating bitcoins, the higher the bar gets to overpower the network, making larger farms also too small to overpower it so that they may as well generate bitcoins too. According to the long tail theory, the small, medium and merely large farms put together should add up to a lot more than the biggest zombie farm."),
        ("Satoshi (SHA-256 post)", "SHA-256 is very strong. It's not like the incremental step from MD5 to SHA1. It can last several decades unless there's some massive breakthrough attack. If SHA-256 became completely broken, I think we could come to some agreement about what the honest block chain was before the trouble started, lock that in and continue from there with a new hash function."),
        ("Satoshi (whitepaper)", "A purely peer-to-peer version of electronic cash would allow online payments to be sent directly from one party to another without going through a financial institution. Digital signatures provide part of the solution, but the main benefits are lost if a trusted third party is still required to prevent double-spending. We propose a solution to the double-spending problem using a peer-to-peer network."),
        ("Satoshi (Malmi email style)", "Thanks for starting that topic, your understanding of bitcoin is spot on. Some of their responses were rather Neanderthal, although I guess they're so used to being anti-fiat-money that anything short of gold isn't good enough. They don't understand that it doesn't matter what backs the money, what matters is who controls the supply."),
        ("Non-Satoshi (generic)", "I think the real question is whether Bitcoin can scale to handle millions of transactions per day. The current block size limit seems like it will be a problem eventually. We need to figure out a solution before it becomes critical."),
        ("Hal Finney", "Bitcoin seems to be a very promising idea. I like the idea of basing security on the assumption that the CPU power of honest participants outweighs that of the attacker. It is a very modern notion that exploits the power of the long tail."),
        ("Craig Wright style", "I am Satoshi Nakamoto. I created Bitcoin. The evidence is clear and I have the keys to prove it. Anyone who disagrees is simply wrong and doesn't understand the technology I invented."),
    ]

    for name, text in test_texts:
        # Extract same features as training
        hf = extract_handcrafted(text)
        hv = np.array([[hf[k] for k in hand_names]])
        cv_text = char_vec.transform([text])

        if biber_matrix is not None and best_config[0] in ['Biber only', 'Handcrafted + Biber', 'Biber + Char n-grams', 'ALL COMBINED']:
            # Extract biber for this text
            import polars as pl
            from pybiber import CorpusProcessor, biber as biber_fn
            import spacy
            nlp = spacy.load('en_core_web_sm')
            cp = CorpusProcessor()
            bdf = pl.DataFrame({'doc_id': ['test'], 'text': [text]})
            parsed = cp.process_corpus(bdf, nlp)
            br = biber_fn(parsed)
            bv = br.select([c for c in br.columns if c != 'doc_id']).to_numpy().astype(float)
            bv = np.nan_to_num(bv, nan=0.0)

            if 'ALL COMBINED' in best_config[0]:
                fv = hstack([csr_matrix(hv), csr_matrix(bv), cv_text])
            elif 'Handcrafted + Biber' in best_config[0]:
                fv = hstack([csr_matrix(hv), csr_matrix(bv)])
            elif 'Biber + Char' in best_config[0]:
                fv = hstack([csr_matrix(bv), cv_text])
            else:
                fv = csr_matrix(bv)
        elif 'Char' in best_config[0] and 'Hand' in best_config[0]:
            fv = hstack([csr_matrix(hv), cv_text])
        elif 'Char' in best_config[0]:
            fv = cv_text
        else:
            fv = csr_matrix(hv)

        prob = best_pipe.predict_proba(fv)[0] if hasattr(best_pipe.named_steps['clf'], 'predict_proba') else None
        pred = best_pipe.predict(fv)[0]
        label = "SATOSHI" if pred == 1 else "NOT SATOSHI"
        prob_str = f" (prob={prob[1]:.3f})" if prob is not None else ""
        print(f"  {name:<35s} -> {label}{prob_str}")


if __name__ == "__main__":
    main()
