"""
Satlock — Satoshi Nakamoto Writing Style Analyzer

Stylometry-based analysis of writing style similarity to Satoshi Nakamoto.
Uses handcrafted linguistic features (function words, punctuation patterns,
sentence structure, contractions, hedging) trained on 565 verified Satoshi
texts and 2,873 contemporary Bitcoin community texts.
"""

import json
import os
import re
import numpy as np
from collections import Counter
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import gradio as gr
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Feature extraction
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

def extract_features(text):
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

# ============================================================
# Load training data and train model
# ============================================================

# Training data is bundled with the Space
SAT_DATA = "satoshi_clean.json"
NSAT_DATA = "non_satoshi_clean.json"

with open(SAT_DATA, encoding='utf-8') as fp:
    sat = [d for d in json.load(fp) if d['word_count'] >= 50]
with open(NSAT_DATA, encoding='utf-8') as fp:
    nsat = [d for d in json.load(fp) if d['word_count'] >= 50]

all_texts = [d['text'] for d in sat] + [d['text'] for d in nsat]
all_labels = np.array([1]*len(sat) + [0]*len(nsat))
hand = [extract_features(t) for t in all_texts]
FEATURE_NAMES = sorted(hand[0].keys())
X = np.array([[f[k] for k in FEATURE_NAMES] for f in hand])

MODEL = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', class_weight='balanced', C=10.0, gamma='scale', probability=True))
])
MODEL.fit(X, all_labels)

# ============================================================
# Inference
# ============================================================

def analyze(text):
    if not text or len(text.split()) < 20:
        return "Please enter at least 20 words for meaningful analysis.", ""

    word_count = len(text.split())
    features = extract_features(text)
    fv = np.array([[features[k] for k in FEATURE_NAMES]])
    prob = MODEL.predict_proba(fv)[0][1]
    score_pct = prob * 100

    # Build score bar
    bar_filled = int(prob * 30)
    bar = "=" * bar_filled + "-" * (30 - bar_filled)

    # Verdict
    if score_pct >= 70:
        verdict = "Strong stylistic match with Satoshi's writing"
        color = "#2d8a4e"
    elif score_pct >= 40:
        verdict = "Moderate stylistic similarity"
        color = "#c4a000"
    elif score_pct >= 15:
        verdict = "Weak similarity - some shared patterns"
        color = "#cc7000"
    else:
        verdict = "Does not match Satoshi's writing style"
        color = "#888"

    # Main result
    result = f"""## Satoshi Score: {score_pct:.1f}%

**[{bar}]**

**{verdict}**

*Based on {word_count} words analyzed across 126 stylometric dimensions.*
"""

    # Style fingerprint breakdown
    sat_features = [extract_features(d['text']) for d in sat[:100]]
    user_f = features

    comparisons = [
        ("Contraction rate", "contraction_rate", "don't, it's, can't, etc."),
        ("Sentence length", "sent_mean", "avg words per sentence"),
        ("First person (I/me/my)", "first_person", "self-referencing"),
        ("Conditional 'if'", "fw_if", "conditional reasoning"),
        ("Hedging", "hedge_rate", "think, believe, probably, etc."),
        ("Comma rate", "punct_comma", "clause-separating commas"),
        ("Parentheses", "punct_paren", "parenthetical asides"),
        ("Second person (you/your)", "second_person", "addressing the reader"),
        ("Vocabulary richness", "vocab_ttr", "unique words / total words"),
        ("Word length", "word_mean_len", "avg characters per word"),
        ("Dash rate", "punct_dash", "use of hyphens/dashes"),
        ("Certainty markers", "certainty_rate", "definitely, always, never, etc."),
    ]

    details = "### Style fingerprint\n\n"
    details += "| Feature | Your text | Satoshi avg | Match |\n|---|---|---|---|\n"
    for label, key, note in comparisons:
        user_val = user_f.get(key, 0)
        sat_avg = np.mean([f.get(key, 0) for f in sat_features])
        if sat_avg > 0.0001:
            ratio = user_val / sat_avg
            if 0.7 <= ratio <= 1.3:
                icon = "Close"
            elif 0.4 <= ratio <= 1.6:
                icon = "~"
            else:
                icon = "Far"
        else:
            icon = "Close" if user_val < 0.001 else "Far"
        details += f"| {label} | {user_val:.4f} | {sat_avg:.4f} | {icon} |\n"

    details += "\n*Close = within 30% of Satoshi's average. Features measured per word.*"

    return result, details


# ============================================================
# Gradio UI
# ============================================================

EXAMPLES = [
    ["SHA-256 is very strong. It's not like the incremental step from MD5 to SHA1. It can last several decades unless there's some massive breakthrough attack. If SHA-256 became completely broken, I think we could come to some agreement about what the honest block chain was before the trouble started, lock that in and continue from there with a new hash function."],
    ["I think the real question is whether Bitcoin can scale to handle millions of transactions per day. The current block size limit seems like it will be a problem eventually. We need to figure out a solution before it becomes critical."],
    ["A purely peer-to-peer version of electronic cash would allow online payments to be sent directly from one party to another without going through a financial institution. Digital signatures provide part of the solution, but the main benefits are lost if a trusted third party is still required to prevent double-spending."],
]

with gr.Blocks(
    title="Satlock",
    theme=gr.themes.Base(primary_hue="amber", neutral_hue="stone"),
    css="""
    .main-header { text-align: center; margin-bottom: 0.5em; }
    .main-header h1 { font-size: 2.5em; margin-bottom: 0; }
    .subtitle { text-align: center; color: #666; margin-bottom: 1.5em; }
    """
) as demo:
    gr.HTML("""
        <div class="main-header"><h1>Satlock</h1></div>
        <p class="subtitle">
            How closely does a piece of writing match Satoshi Nakamoto's style?<br>
            Stylometry trained on 565 verified Satoshi texts vs 2,873 contemporary Bitcoin community texts.
        </p>
    """)

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Paste text to analyze",
                placeholder="Enter a paragraph or more (50+ words recommended)...",
                lines=8,
            )
            submit_btn = gr.Button("Analyze", variant="primary", size="lg")

        with gr.Column(scale=1):
            result_output = gr.Markdown(label="Score")

    details_output = gr.Markdown(label="Style Fingerprint")

    gr.Examples(examples=EXAMPLES, inputs=text_input)

    submit_btn.click(fn=analyze, inputs=text_input, outputs=[result_output, details_output])
    text_input.submit(fn=analyze, inputs=text_input, outputs=[result_output, details_output])

    gr.HTML("""
        <div style="margin-top: 2em; padding-top: 1em; border-top: 1px solid #ddd; color: #888; font-size: 0.85em; text-align: center;">
            <p><strong>How it works:</strong> Extracts 126 stylometric features (function word frequencies, punctuation patterns,
            sentence structure, contraction rates, hedging language) and scores them against Satoshi's writing profile using an SVM classifier.
            Trained on clean, decontaminated data from the same BitcoinTalk threads Satoshi participated in.</p>
            <p><strong>Key Satoshi markers:</strong> 62% more contractions, 31% less first-person "I", shorter sentences, more conditional "if" reasoning.</p>
            <p>This tool measures <em>writing style similarity</em>, not identity. A high score means stylistically similar, not authored by Satoshi.</p>
            <p><a href="https://github.com/jnathan9/satoshi-stylometry">GitHub</a></p>
        </div>
    """)

demo.launch()
