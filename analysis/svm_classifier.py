"""
Satoshi SVM Classifier: Handcrafted + Biber features with SVM (best combo from grid search).
Focused version — train, evaluate, test on known texts.
"""

import json
import os
import re
import numpy as np
from collections import Counter
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "v5_clean")

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
        f['sent_count'] = len(sentences) / wc * 100
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


def extract_biber_batch(texts, doc_ids):
    """Extract Biber features for texts."""
    from pybiber import CorpusProcessor, biber
    import polars as pl
    import spacy
    nlp = spacy.load('en_core_web_sm')
    cp = CorpusProcessor()

    batch_size = 500
    all_results = []
    for i in range(0, len(texts), batch_size):
        df = pl.DataFrame({
            'doc_id': doc_ids[i:i+batch_size],
            'text': texts[i:i+batch_size]
        })
        parsed = cp.process_corpus(df, nlp)
        result = biber(parsed)
        all_results.append(result)

    combined = pl.concat(all_results)
    feature_cols = [c for c in combined.columns if c != 'doc_id']
    matrix = np.nan_to_num(combined.select(feature_cols).to_numpy().astype(float), nan=0.0)
    return matrix, feature_cols


def main():
    with open(os.path.join(DATA_DIR, "satoshi_clean.json"), encoding='utf-8') as f:
        sat_all = json.load(f)
    with open(os.path.join(DATA_DIR, "non_satoshi_clean.json"), encoding='utf-8') as f:
        nsat_all = json.load(f)

    sat = [d for d in sat_all if d['word_count'] >= 50]
    nsat = [d for d in nsat_all if d['word_count'] >= 50]
    print(f"Satoshi: {len(sat)} | Non-Satoshi: {len(nsat)}", flush=True)

    all_texts = [d['text'] for d in sat] + [d['text'] for d in nsat]
    all_labels = np.array([1]*len(sat) + [0]*len(nsat))
    doc_ids = [f"s{i}" for i in range(len(sat))] + [f"n{i}" for i in range(len(nsat))]

    # Extract features
    print("Extracting handcrafted features...", flush=True)
    hand = [extract_handcrafted(t) for t in all_texts]
    hand_names = sorted(hand[0].keys())
    X_hand = np.array([[f[k] for k in hand_names] for f in hand])

    print("Extracting Biber features...", flush=True)
    X_biber, biber_names = extract_biber_batch(all_texts, doc_ids)

    X = np.hstack([X_hand, X_biber])
    all_feature_names = hand_names + biber_names
    print(f"Total features: {X.shape[1]} ({len(hand_names)} handcrafted + {len(biber_names)} Biber)", flush=True)

    # Cross-validation
    print("\n5-fold CV...", flush=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipe = Pipeline([('scaler', StandardScaler()), ('svm', SVC(kernel='rbf', class_weight='balanced', C=10.0, gamma='scale', probability=True))])
    scores = cross_val_score(pipe, X, all_labels, cv=cv, scoring='f1')
    print(f"F1 = {scores.mean():.3f} (±{scores.std():.3f})", flush=True)

    # Train/test split evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, all_labels, test_size=0.2, random_state=42, stratify=all_labels)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    print(f"\nTest set ({len(X_test)} samples):", flush=True)
    print(classification_report(y_test, y_pred, target_names=['not_satoshi', 'satoshi']))
    cm = confusion_matrix(y_test, y_pred)
    print(f"  not_satoshi: {cm[0][0]} correct, {cm[0][1]} false positive")
    print(f"  satoshi:     {cm[1][0]} missed,  {cm[1][1]} correct")

    # Train on ALL data for final model
    print("\nTraining final model on all data...", flush=True)
    pipe.fit(X, all_labels)

    # Test on known texts
    print(f"\n{'='*60}", flush=True)
    print("KNOWN TEXT PREDICTIONS", flush=True)
    print(f"{'='*60}", flush=True)

    test_texts = [
        ("Satoshi (zombie farms)", "There would be many smaller zombie farms that are not big enough to overpower the network, and they could still make money by generating bitcoins. The smaller farms are then the honest nodes. The more smaller farms resort to generating bitcoins, the higher the bar gets to overpower the network, making larger farms also too small to overpower it so that they may as well generate bitcoins too. According to the long tail theory, the small, medium and merely large farms put together should add up to a lot more than the biggest zombie farm."),
        ("Satoshi (SHA-256)", "SHA-256 is very strong. It's not like the incremental step from MD5 to SHA1. It can last several decades unless there's some massive breakthrough attack. If SHA-256 became completely broken, I think we could come to some agreement about what the honest block chain was before the trouble started, lock that in and continue from there with a new hash function."),
        ("Satoshi (whitepaper)", "A purely peer-to-peer version of electronic cash would allow online payments to be sent directly from one party to another without going through a financial institution. Digital signatures provide part of the solution, but the main benefits are lost if a trusted third party is still required to prevent double-spending. We propose a solution to the double-spending problem using a peer-to-peer network."),
        ("Satoshi (Malmi email)", "Thanks for starting that topic, your understanding of bitcoin is spot on. Some of their responses were rather Neanderthal, although I guess they're so used to being anti-fiat-money that anything short of gold isn't good enough. They don't understand that it doesn't matter what backs the money, what matters is who controls the supply."),
        ("Non-Satoshi (generic)", "I think the real question is whether Bitcoin can scale to handle millions of transactions per day. The current block size limit seems like it will be a problem eventually. We need to figure out a solution before it becomes critical."),
        ("Hal Finney", "Bitcoin seems to be a very promising idea. I like the idea of basing security on the assumption that the CPU power of honest participants outweighs that of the attacker. It is a very modern notion that exploits the power of the long tail."),
        ("Craig Wright", "I am Satoshi Nakamoto. I created Bitcoin. The evidence is clear and I have the keys to prove it. Anyone who disagrees is simply wrong and doesn't understand the technology I invented."),
    ]

    # Extract features for each test text
    from pybiber import CorpusProcessor, biber
    import polars as pl
    import spacy
    nlp = spacy.load('en_core_web_sm')
    cp = CorpusProcessor()

    for name, text in test_texts:
        hf = extract_handcrafted(text)
        hv = np.array([[hf[k] for k in hand_names]])

        bdf = pl.DataFrame({'doc_id': ['t'], 'text': [text]})
        parsed = cp.process_corpus(bdf, nlp)
        br = biber(parsed)
        bv = np.nan_to_num(br.select([c for c in br.columns if c != 'doc_id']).to_numpy().astype(float), nan=0.0)

        fv = np.hstack([hv, bv])
        prob = pipe.predict_proba(fv)[0]
        verdict = "SATOSHI" if prob[1] > 0.5 else "NOT SATOSHI"
        bar = "█" * int(prob[1] * 20) + "░" * (20 - int(prob[1] * 20))
        print(f"  {name:<30s} {bar} {prob[1]:.1%} -> {verdict}", flush=True)


if __name__ == "__main__":
    main()
