"""Score real candidate texts through the Satoshi classifier."""

import json, os, re, numpy as np
from collections import Counter
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "v5_clean")
CAND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "candidates")

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

# Train classifier on Satoshi vs non-Satoshi
with open(os.path.join(DATA_DIR, 'satoshi_clean.json')) as fp:
    sat = [d for d in json.load(fp) if d['word_count'] >= 50]
with open(os.path.join(DATA_DIR, 'non_satoshi_clean.json')) as fp:
    nsat = [d for d in json.load(fp) if d['word_count'] >= 50]

all_texts = [d['text'] for d in sat] + [d['text'] for d in nsat]
all_labels = np.array([1]*len(sat) + [0]*len(nsat))
hand = [extract_handcrafted(t) for t in all_texts]
hand_names = sorted(hand[0].keys())
X = np.array([[f[k] for k in hand_names] for f in hand])

pipe = Pipeline([('scaler', StandardScaler()), ('svm', SVC(kernel='rbf', class_weight='balanced', C=10.0, gamma='scale', probability=True))])
pipe.fit(X, all_labels)
print(f"Trained on {len(sat)} Satoshi + {len(nsat)} non-Satoshi texts\n")

# Also score Satoshi's own texts as baseline
sat_scores = []
for d in sat:
    hf = extract_handcrafted(d['text'])
    fv = np.array([[hf[k] for k in hand_names]])
    prob = pipe.predict_proba(fv)[0][1]
    sat_scores.append(prob)

print(f"Satoshi baseline (trained texts): mean={np.mean(sat_scores):.1%}, median={np.median(sat_scores):.1%}")
print(f"  25th pct: {np.percentile(sat_scores, 25):.1%}, 75th pct: {np.percentile(sat_scores, 75):.1%}\n")

# Score each candidate
print("=" * 70)
print("CANDIDATE SCORES (real scraped texts)")
print("=" * 70)

for candidate_file in sorted(os.listdir(CAND_DIR)):
    if not candidate_file.endswith('.json'):
        continue
    name = candidate_file.replace('.json', '').replace('_', ' ').title()
    with open(os.path.join(CAND_DIR, candidate_file)) as fp:
        texts = json.load(fp)

    # Filter to 50+ word texts
    texts = [t for t in texts if len(t.split()) >= 50]
    if not texts:
        print(f"\n{name}: No texts with 50+ words")
        continue

    scores = []
    for text in texts:
        hf = extract_handcrafted(text)
        fv = np.array([[hf[k] for k in hand_names]])
        prob = pipe.predict_proba(fv)[0][1]
        scores.append(prob)

    mean_score = np.mean(scores)
    median_score = np.median(scores)
    pct_above_50 = sum(1 for s in scores if s > 0.5) / len(scores) * 100

    bar = chr(9608) * int(mean_score * 30) + chr(9617) * (30 - int(mean_score * 30))
    print(f"\n{name} ({len(texts)} texts, {sum(len(t.split()) for t in texts):,} words)")
    print(f"  Mean:   {bar} {mean_score:.1%}")
    print(f"  Median: {median_score:.1%} | Texts scoring >50%: {pct_above_50:.0f}%")
    print(f"  Range:  {min(scores):.1%} - {max(scores):.1%}")

    # Show highest-scoring text snippet
    best_idx = np.argmax(scores)
    print(f"  Highest ({scores[best_idx]:.1%}): \"{texts[best_idx][:120]}...\"")

if __name__ == "__main__":
    pass
