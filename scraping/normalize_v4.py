"""
V4 Normalization: Universal cleaning function for ALL text.

After normalize_text(), you cannot tell whether the original came from
a forum, mailing list, or email client. The only information remaining
is the actual words and sentences the author chose to write.
"""

import json
import os
import re
import random
import unicodedata
from collections import Counter

random.seed(42)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
OUT_DIR = os.path.join(DATA_DIR, "v4_processed")
os.makedirs(OUT_DIR, exist_ok=True)

MIN_WORDS = 50
MAX_WORDS = 400


def normalize_text(raw_text):
    """
    Universal text normalization. ALL text (Satoshi and non-Satoshi,
    from any platform) passes through this SAME function.
    """

    text = raw_text

    # Step 0: Basic encoding cleanup
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = text.replace('\xa0', ' ')  # non-breaking space
    text = text.replace('\u200b', '')  # zero-width space
    text = text.replace('\u200c', '')  # zero-width non-joiner
    text = text.replace('\ufeff', '')  # BOM

    # Step 1: Strip metadata headers (line by line)
    lines = text.split('\n')
    cleaned_lines = []
    in_header = True

    for line in lines:
        stripped = line.strip()

        # Skip empty lines at the start
        if in_header and not stripped:
            continue

        # Skip metadata header lines
        if in_header:
            skip = False
            # Email headers
            for prefix in ['From:', 'To:', 'Cc:', 'Subject:', 'Date:', 'Message-ID:',
                           'References:', 'In-Reply-To:', 'Content-Type:', 'Content-Transfer',
                           'MIME-Version:', 'X-', 'Return-Path:', 'Received:', 'Delivered-To:']:
                if stripped.startswith(prefix):
                    skip = True
                    break
            # Nakamoto Institute chrome
            if stripped in ('BitcoinTalk', 'P2P Foundation', 'Cryptography Mailing List',
                           'Bitcoin-list', 'P2P Research', 'View original', 'View in thread'):
                skip = True
            # Standalone author names
            if stripped.lower() in ('satoshi', 'satoshi nakamoto'):
                skip = True
            # Standalone numbers (post numbers)
            if re.match(r'^\d{1,4}$', stripped):
                skip = True
            # Date-only lines
            if re.match(r'^(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d', stripped):
                skip = True
            if re.match(r'^\d{4}-\d{2}-\d{2}', stripped):
                skip = True
            if 'UTC' in stripped and len(stripped) < 50:
                skip = True
            # "Re: title" lines (email/forum subjects)
            if re.match(r'^Re:\s', stripped) and len(stripped) < 120:
                skip = True
            # Title-like lines at the very start (before any content)
            if len(cleaned_lines) == 0 and len(stripped) < 80 and not any(c in stripped for c in '.?!,;:'):
                skip = True

            if skip:
                continue
            else:
                in_header = False

        cleaned_lines.append(line)

    text = '\n'.join(cleaned_lines)

    # Step 2: Remove quoted/reply text
    lines = text.split('\n')
    unquoted = []
    for line in lines:
        stripped = line.strip()
        # Email-style quotes
        if stripped.startswith('>'):
            continue
        # "On DATE, PERSON wrote:" attributions
        if re.match(r'^On .+ wrote:\s*$', stripped):
            continue
        # "PERSON wrote:" (short name)
        if re.match(r'^.{1,40} wrote:\s*$', stripped):
            continue
        # "Quote from: USER on DATE" forum quote blocks
        if re.match(r'^Quote from:', stripped):
            continue
        # Standalone "Quote"
        if stripped == 'Quote':
            continue
        unquoted.append(line)
    text = '\n'.join(unquoted)

    # Step 3: Remove signatures
    # Look for standard sig separator
    sig_match = re.search(r'\n-- ?\n', text)
    if sig_match:
        text = text[:sig_match.start()]
    # Also remove "---" separators that often precede sigs
    sig_match = re.search(r'\n---+\s*\n', text)
    if sig_match:
        # Only cut if it's in the last 30% of the text
        if sig_match.start() > len(text) * 0.7:
            text = text[:sig_match.start()]

    # Step 3b: Remove Nakamoto Institute chrome (can appear ANYWHERE in text)
    ni_patterns = [
        'View original', 'View in thread', 'Previous', 'Next',
        'Back to emails', 'Back to posts',
        'Satoshi Nakamoto Institute',
    ]
    lines = text.split('\n')
    text = '\n'.join(line for line in lines if line.strip() not in ni_patterns)
    for pattern in ni_patterns:
        text = text.replace(pattern, '')

    # Step 3c: Remove NI metadata blocks that survived header stripping
    # Pattern: "From:\nsatoshi\nSubject:\nTITLE\nDate:\nDATE"
    text = re.sub(r'From:\s*\n\s*satoshi\s*\n', '', text, flags=re.IGNORECASE)
    text = re.sub(r'From:\s*\n\s*Satoshi Nakamoto\s*\n', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Subject:\s*\n', '', text)
    text = re.sub(r'Date:\s*\n', '', text)
    # Remove standalone line numbers (NI post numbers)
    text = re.sub(r'^\d{1,4}\s*$', '', text, flags=re.MULTILINE)
    # Remove standalone date lines anywhere
    text = re.sub(r'^(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\w+ \d{1,2}, \d{4} at \d{1,2}:\d{2}:\d{2} UTC\s*$', '', text, flags=re.MULTILINE)

    # Step 4: Remove URLs
    text = re.sub(r'https?://\S+', '', text)

    # Step 5: Remove code blocks
    # Forum "Code:" blocks
    text = re.sub(r'Code:\s*\n.*?(?=\n\n|\Z)', '', text, flags=re.DOTALL)
    # Indented code blocks (4+ spaces at start, consecutive lines)
    text = re.sub(r'(?:\n {4,}\S.*){2,}', '', text)

    # Step 6: Remove bracketed annotations < 40 chars
    text = re.sub(r'\[.{1,38}\]', '', text)

    # Step 7: Normalize whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)       # Collapse excess newlines
    text = re.sub(r' {2,}', ' ', text)             # Collapse spaces
    lines = text.split('\n')
    lines = [line.strip() for line in lines]       # Strip each line
    # Remove empty lines that are just whitespace artifacts
    # Keep at most one blank line between paragraphs
    cleaned = []
    prev_blank = False
    for line in lines:
        if not line:
            if not prev_blank:
                cleaned.append('')
                prev_blank = True
        else:
            cleaned.append(line)
            prev_blank = False
    text = '\n'.join(cleaned)
    text = text.strip()

    # Step 8: Normalize punctuation
    # Curly quotes to straight
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    # Em/en dash to hyphen
    text = text.replace('\u2013', '-').replace('\u2014', '-')
    # Ellipsis character to dots
    text = text.replace('\u2026', '...')

    return text


def chunk_text(text, min_words=MIN_WORDS, max_words=MAX_WORDS):
    """Split text into chunks by paragraph boundaries."""
    words = text.split()
    if len(words) <= max_words:
        if len(words) >= min_words:
            return [text]
        return []

    chunks = []
    paragraphs = re.split(r'\n\n+', text)
    current = []
    current_wc = 0

    for para in paragraphs:
        pw = len(para.split())
        if pw == 0:
            continue
        if current_wc + pw > max_words and current_wc >= min_words:
            chunks.append('\n\n'.join(current))
            current = [para]
            current_wc = pw
        else:
            current.append(para)
            current_wc += pw

    if current_wc >= min_words:
        chunks.append('\n\n'.join(current))

    return chunks


def main():
    # Load V4 raw data
    sat_path = os.path.join(RAW_DIR, "v4_satoshi.json")
    nsat_path = os.path.join(RAW_DIR, "v4_non_satoshi.json")

    with open(sat_path, encoding='utf-8') as f:
        satoshi_raw = json.load(f)
    with open(nsat_path, encoding='utf-8') as f:
        non_satoshi_raw = json.load(f)

    print(f"Raw Satoshi: {len(satoshi_raw)}")
    print(f"Raw Non-Satoshi: {len(non_satoshi_raw)}")

    # Normalize and chunk - SAME function for both
    def process(items, label, label_name):
        chunks = []
        for item in items:
            cleaned = normalize_text(item['text'])
            for i, chunk in enumerate(chunk_text(cleaned)):
                chunks.append({
                    "text": chunk,
                    "label": label,
                    "label_name": label_name,
                    "source": item['source'],
                    "author": item.get('author', 'unknown'),
                    "word_count": len(chunk.split()),
                })
        return chunks

    sat_chunks = process(satoshi_raw, 1, "satoshi")
    nsat_chunks = process(non_satoshi_raw, 0, "not_satoshi")

    print(f"\nSatoshi chunks: {len(sat_chunks)}")
    print(f"Non-Satoshi chunks: {len(nsat_chunks)}")

    # Source distribution
    def show_sources(chunks, name):
        sources = Counter(c['source'] for c in chunks)
        total = len(chunks)
        print(f"\n{name} source distribution:")
        for s, c in sorted(sources.items(), key=lambda x: -x[1]):
            print(f"  {s}: {c} ({c/total*100:.1f}%)")

    show_sources(sat_chunks, "Satoshi")
    show_sources(nsat_chunks, "Non-Satoshi")

    # Shuffle
    random.shuffle(sat_chunks)
    random.shuffle(nsat_chunks)

    # Stratified split: 60% train, 10% val, 10% test, 20% golden
    def split_data(chunks):
        n = len(chunks)
        n_golden = max(20, int(n * 0.20))
        n_test = max(15, int(n * 0.10))
        n_val = max(15, int(n * 0.10))
        return (
            chunks[n_golden + n_test + n_val:],  # train
            chunks[n_golden + n_test:n_golden + n_test + n_val],  # val
            chunks[n_golden:n_golden + n_test],  # test
            chunks[:n_golden],  # golden
        )

    sat_train, sat_val, sat_test, sat_golden = split_data(sat_chunks)
    nsat_train, nsat_val, nsat_test, nsat_golden = split_data(nsat_chunks)

    splits = {
        "train": sat_train + nsat_train,
        "val": sat_val + nsat_val,
        "test": sat_test + nsat_test,
        "golden": sat_golden + nsat_golden,
        "golden_satoshi": sat_golden,
        "golden_non_satoshi": nsat_golden,
    }

    for name in ["train", "val", "test", "golden"]:
        random.shuffle(splits[name])

    # Save
    for name, data in splits.items():
        path = os.path.join(OUT_DIR, f"{name}.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        n_sat = sum(1 for d in data if d['label'] == 1)
        n_nsat = sum(1 for d in data if d['label'] == 0)
        print(f"\n{name}: {len(data)} (satoshi={n_sat}, not_satoshi={n_nsat})")

    # Show samples
    print("\n=== SAMPLE SATOSHI (after normalization) ===")
    print(sat_chunks[0]['text'][:300])
    print("\n=== SAMPLE NON-SATOSHI (after normalization) ===")
    print(nsat_chunks[0]['text'][:300])


if __name__ == "__main__":
    main()
