# Satoshi Stylometry

A BERT-based stylometry classifier that identifies whether a piece of text was written by Satoshi Nakamoto, trained on the complete corpus of Satoshi's public writings.

## Try it

**[Live Demo on HuggingFace Spaces](https://huggingface.co/spaces/thestalwart/satoshi-stylometry)** - Paste any text and get a Satoshi probability score.

## How it works

We fine-tuned [ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) as a binary classifier (Satoshi vs. not-Satoshi) on the complete public record of Satoshi Nakamoto's writings.

### Data

**Satoshi corpus (572 texts):**
- 539 BitcoinTalk forum posts (2009-2010)
- 18 Cryptography mailing list emails (2008-2009)
- 11 Bitcoin-list emails (2009-2010)
- 4 P2P Foundation posts (2009)

All sourced from [satoshi.nakamotoinstitute.org](https://satoshi.nakamotoinstitute.org/).

**Non-Satoshi corpus (1,546 texts):**
- 761 posts from the Cryptography mailing list (same era, same community)
- 785 BitcoinTalk forum posts from other early users

All texts were cleaned (metadata headers removed, quoted text stripped) and chunked into 50-400 word pieces, yielding 452 Satoshi chunks and 1,304 non-Satoshi chunks.

### Training

- **Model:** ModernBERT-base (149M parameters)
- **GPU:** NVIDIA A10G on [Modal](https://modal.com)
- **Epochs:** 10
- **Training time:** ~4 minutes
- **Class-weighted loss** to handle the 1:3 class imbalance

### Results

**Golden held-out set (350 texts never seen during training):**

| Metric | Score |
|--------|-------|
| **Accuracy** | 99.1% (347/350) |
| **Satoshi recall** | 97.8% (88/90) |
| **Non-Satoshi recall** | 99.6% (259/260) |
| **F1** | 0.983 |

**Confusion matrix:**

| | Predicted Not-Satoshi | Predicted Satoshi |
|---|:---:|:---:|
| **Actual Not-Satoshi** | 259 | 1 |
| **Actual Satoshi** | 2 | 88 |

**Confidence distribution:**
- Satoshi texts: mean probability 0.978 (very confident)
- Non-Satoshi texts: mean probability 0.004 (very confident it's NOT Satoshi)

Only 3 misclassifications:
- 2 very short/generic Satoshi posts that lacked his distinctive style
- 1 non-Satoshi post about building Bitcoin on FreeBSD that mimicked his technical voice

### Caveats

This model detects **writing style**, not identity. A high score means "this text is stylistically similar to Satoshi's known writings." It does not prove authorship. The model was trained on crypto/technical forum posts, so it may be less reliable on text from very different domains.

## Project structure

```
satoshi-stylometry/
├── data/
│   ├── raw/                    # Original scraped data
│   │   ├── satoshi_raw.json    # 572 Satoshi writings
│   │   └── non_satoshi_raw.json # 1,546 non-Satoshi writings
│   └── processed/              # Cleaned, chunked, and split
│       ├── train.json          # 1,056 training chunks
│       ├── val.json            # 175 validation chunks
│       ├── test.json           # 175 test chunks
│       ├── golden.json         # 350 held-out golden set
│       ├── golden_satoshi.json # 90 held-out Satoshi texts
│       └── golden_non_satoshi.json # 260 held-out non-Satoshi texts
├── scraping/
│   ├── scrape_satoshi.py       # Scrape from Nakamoto Institute
│   ├── scrape_non_satoshi_fast.py # Concurrent mailing list + forum scraper
│   └── normalize_and_split.py  # Text cleaning, chunking, train/test splits
└── training/
    ├── train_modal.py          # Modal GPU training script
    └── upload_to_modal.py      # Upload data to Modal volume
```

## Reproducing

1. Install dependencies: `pip install beautifulsoup4 requests`
2. Scrape data: `python scraping/scrape_satoshi.py && python scraping/scrape_non_satoshi_fast.py`
3. Normalize: `python scraping/normalize_and_split.py`
4. Upload to Modal: `python training/upload_to_modal.py`
5. Train: `modal run training/train_modal.py`

## License

MIT
