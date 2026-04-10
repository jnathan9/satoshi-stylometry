"""Score Satoshi candidates: Adam Back, Nick Szabo, Hal Finney, Craig Wright, Gavin Andresen."""

import json, os, re, numpy as np
from collections import Counter
from sklearn.svm import SVC
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

# Load and train
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

# ============================================================
# Candidates
# ============================================================

candidates = [
    # Known Satoshi
    ("Satoshi (SHA-256 post)",
     "SHA-256 is very strong. It's not like the incremental step from MD5 to SHA1. It can last several decades unless there's some massive breakthrough attack. If SHA-256 became completely broken, I think we could come to some agreement about what the honest block chain was before the trouble started, lock that in and continue from there with a new hash function."),

    ("Satoshi (zombie farms)",
     "There would be many smaller zombie farms that are not big enough to overpower the network, and they could still make money by generating bitcoins. The smaller farms are then the honest nodes. The more smaller farms resort to generating bitcoins, the higher the bar gets to overpower the network, making larger farms also too small to overpower it so that they may as well generate bitcoins too."),

    ("Satoshi (Malmi email)",
     "Thanks for starting that topic, your understanding of bitcoin is spot on. Some of their responses were rather Neanderthal, although I guess they're so used to being anti-fiat-money that anything short of gold isn't good enough. They don't understand that it doesn't matter what backs the money, what matters is who controls the supply."),

    # Adam Back
    ("Adam Back (hashcash)",
     "Hashcash was designed as a proof of work system. The idea is that you can use computational work as a form of postage to prevent spam. Each email would need to include a proof of work token that takes some amount of CPU time to compute. This makes bulk spam economically infeasible while being negligible cost for legitimate senders."),

    ("Adam Back (technical)",
     "The key insight is that we can use hash collisions as proof of computational work. Finding a partial hash collision with k leading zero bits requires on average 2^k hash operations. This is easy to verify but hard to produce, which gives us the asymmetry needed for a useful proof of work system."),

    # Nick Szabo
    ("Nick Szabo (bit gold)",
     "Bit gold is my proposal for a protocol whereby unforgeable costly bits could be created online with minimal dependence on trusted third parties. These bits would possess properties similar to those of gold: mainly that their value derives from the costliness of their creation rather than from any trusted third party."),

    ("Nick Szabo (smart contracts)",
     "The idea of smart contracts goes back to my 1994 paper. A smart contract is a computerized transaction protocol that executes the terms of a contract. The general objectives of smart contract design are to satisfy common contractual conditions, minimize exceptions both malicious and accidental, and minimize the need for trusted intermediaries."),

    # Hal Finney
    ("Hal Finney (Bitcoin review)",
     "Bitcoin seems to be a very promising idea. I like the idea of basing security on the assumption that the CPU power of honest participants outweighs that of the attacker. It is a very modern notion that exploits the power of the long tail."),

    ("Hal Finney (RPOW)",
     "Reusable proofs of work extend the concept of hashcash by making the tokens reusable. Instead of each token being used once and discarded, RPOW tokens can be exchanged for new ones of equal value. This is accomplished through a trusted computing platform that can verify and reissue tokens."),

    # Craig Wright
    ("Craig Wright",
     "I am Satoshi Nakamoto. I created Bitcoin. The evidence is clear and I have the keys to prove it. Anyone who disagrees is simply wrong and does not understand the technology I invented."),

    # Gavin Andresen
    ("Gavin Andresen",
     "I think the Bitcoin network can handle a lot more transactions than it is handling right now. The one megabyte block size limit is an arbitrary limit that was put in place as a temporary measure. We should increase it to allow Bitcoin to grow and serve more users."),

    # Wei Dai
    ("Wei Dai (b-money)",
     "I am fascinated by Tim May's crypto-anarchy. Unlike the communities traditionally associated with the word anarchy, in a crypto-anarchy the government is not temporarily destroyed but permanently forbidden and permanently unnecessary. It is a community where the threat of violence is impotent because violence is impossible, and violence is impossible because its participants cannot be linked to their true names or physical locations."),
]

print("SATOSHI CANDIDATE SCORES")
print("=" * 65)
for name, text in candidates:
    hf = extract_handcrafted(text)
    fv = np.array([[hf[k] for k in hand_names]])
    prob = pipe.predict_proba(fv)[0]
    bar = chr(9608) * int(prob[1] * 20) + chr(9617) * (20 - int(prob[1] * 20))
    print(f"  {name:<30s} {bar} {prob[1]:.1%}")
