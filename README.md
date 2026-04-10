# Satlock

A stylometry tool that measures how closely a piece of writing matches Satoshi Nakamoto's writing style. Uses classical linguistic feature analysis — not neural networks — trained on the largest known corpus of Satoshi's public and private writings.

## Try it

**[Live Demo on HuggingFace Spaces](https://huggingface.co/spaces/thestalwart/satlock)** — Paste any text and see how it scores.

## What makes Satoshi's writing distinctive

We extracted 126 stylometric features from 565 verified Satoshi texts and compared them to 2,873 texts from his contemporaries in the early Bitcoin community. The most distinctive markers:

| Feature | Satoshi | Others | Difference |
|---|---|---|---|
| **Contraction rate** ("don't", "it's") | 3.0% | 1.8% | 62% more |
| **First person ("I/me/my")** | 1.2% | 1.8% | 31% less |
| **Sentence length** | 13.9 words | 17.3 words | Shorter |
| **Use of "if"** | 1.1% | 0.7% | 62% more |
| **Paragraph count** | 2.9/post | 1.8/post | 64% more |

Satoshi writes shorter sentences, uses more contractions, reasons conditionally with "if", and refers to himself less than his peers. He explains systems rather than sharing personal opinions.

## Candidate comparison

We scored real scraped writings from commonly named Satoshi candidates:

| Writer | Mean Score | Texts Analyzed |
|---|---|---|
| **Satoshi Nakamoto** | **89.6%** | **417** |
| Wei Dai | 20.4% | 10 |
| Hal Finney | 16.3% | 112 |
| Gavin Andresen | 14.4% | 151 |
| Nick Szabo | 9.2% | 57 |

None of the candidates score close to Satoshi's baseline. Nick Szabo — often cited as a leading candidate — scores the lowest, with a much more academic writing style.

## Data

**Satoshi corpus: 565 texts, 75,939 words** across 8 sources:

| Source | Texts | Words |
|---|---|---|
| BitcoinTalk posts | 394 | 38,457 |
| Martti Malmi private emails | 136 | 22,706 |
| Cryptography mailing list | 18 | 5,140 |
| Bitcoin whitepaper | 1 | 3,413 |
| Wei Dai emails | 2 | 2,244 |
| Hal Finney emails | 1 | 1,743 |
| Bitcoin-list emails | 9 | 1,320 |
| P2P Foundation | 4 | 916 |

**Non-Satoshi corpus: 2,873 texts, 261,836 words** from 536 authors in the early Bitcoin community, scraped from the exact same 258 BitcoinTalk threads Satoshi participated in (eliminating topic as a confound).

All texts cleaned through the same pipeline: metadata stripped, quoted text removed, signatures removed, URLs removed, deduped, cross-contamination checked.

## Method

SVM classifier on 126 handcrafted stylometric features:
- Function word frequencies (100+ common words)
- Punctuation profile (commas, semicolons, dashes, parentheses, etc.)
- Sentence statistics (length, variance, count)
- Vocabulary richness (type-token ratio, hapax legomena)
- Contraction patterns
- Person reference rates (first/second/third person)
- Hedging and certainty markers

Cross-validated F1: **0.68** on clean, decontaminated data (no format shortcuts — validated by 8 statistical checks including a naive BoW baseline test).

## Project structure

```
satoshi-stylometry/
├── data/
│   ├── v5_clean/               # Final cleaned data
│   │   ├── satoshi_clean.json  # 565 Satoshi texts
│   │   └── non_satoshi_clean.json # 2,873 non-Satoshi texts
│   ├── candidates/             # Scraped candidate writings
│   └── raw/                    # Raw scraped data
├── scraping/
│   ├── scrape_all_satoshi.py   # All Satoshi sources (NI + Malmi + whitepaper + emails)
│   ├── scrape_v4_non_satoshi.py # Same-thread non-Satoshi collection
│   ├── scrape_candidates.py    # Candidate writings
│   ├── clean_v5.py             # Universal cleaning pipeline
│   ├── normalize_v4.py         # Normalization + chunking
│   └── validate_v4.py          # 8 statistical validation checks
├── analysis/
│   ├── stylometry_profile.py   # Satoshi writing fingerprint
│   ├── svm_classifier.py       # SVM classifier (best model)
│   ├── score_real_candidates.py # Candidate scoring on real texts
│   └── full_classifier.py      # Full feature family comparison
├── space/
│   └── app.py                  # Satlock Gradio app
└── training/
    └── train_modal.py          # Earlier BERT experiments (superseded)
```

## Caveats

This tool detects **writing style similarity**, not identity. A high score means stylistically similar to Satoshi's known writings — it does not prove authorship. The classifier works best on 50+ word passages about technical topics. It may be less reliable on very different domains or very short texts.

The F1 of 0.68 is honest — achieved on clean data with no format contamination. Earlier BERT-based versions achieved 99%+ but were exploiting metadata artifacts, not actual writing style.

## License

MIT
