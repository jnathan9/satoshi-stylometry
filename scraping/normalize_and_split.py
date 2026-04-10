"""
Normalize and chunk all scraped texts into similar-sized pieces.
Create train/val/test splits with golden held-out sets.

Strategy:
- Clean metadata headers from text
- Chunk longer texts into ~200-400 word pieces
- Keep shorter texts as-is if they meet minimum length
- Balance classes (satoshi vs non-satoshi)
- Hold out golden test sets for evaluation
"""

import json
import os
import re
import random
from collections import Counter

random.seed(42)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
OUT_DIR = os.path.join(DATA_DIR, "processed")
os.makedirs(OUT_DIR, exist_ok=True)

# Target chunk size
MIN_WORDS = 50
MAX_WORDS = 400
TARGET_WORDS = 250


def clean_satoshi_text(text):
    """Remove metadata headers that the scraper captured."""
    # Remove header blocks like "BitcoinTalk\nRe: Title\nNumber\nFrom:\nsatoshi\nSubject:..."
    # Pattern: starts with source name, then title, then metadata
    lines = text.split('\n')
    cleaned_lines = []
    skip_until_content = True
    metadata_patterns = [
        r'^BitcoinTalk$',
        r'^P2P Foundation$',
        r'^Cryptography Mailing List$',
        r'^Bitcoin-list$',
        r'^P2P Research$',
        r'^From:$',
        r'^satoshi$',
        r'^Satoshi Nakamoto$',
        r'^Subject:$',
        r'^Date:$',
        r'^Re:',
        r'^\d+$',  # Standalone numbers (post numbers)
        r'^\w+ \d+, \d{4}',  # Date lines
        r'^Source$',
        r'^Posted',
    ]

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if not skip_until_content:
                cleaned_lines.append('')
            continue

        # Check if this is still metadata
        is_metadata = False
        if skip_until_content:
            for pattern in metadata_patterns:
                if re.match(pattern, stripped, re.IGNORECASE):
                    is_metadata = True
                    break
            # Also skip if it looks like a date/time
            if re.match(r'^(January|February|March|April|May|June|July|August|September|October|November|December)', stripped):
                is_metadata = True
            if re.match(r'^\d{4}-\d{2}-\d{2}', stripped):
                is_metadata = True
            if 'UTC' in stripped and len(stripped) < 40:
                is_metadata = True

        if is_metadata:
            continue
        else:
            skip_until_content = False
            cleaned_lines.append(line)

    result = '\n'.join(cleaned_lines).strip()

    # Remove "Quote from: ..." blocks (but keep the response)
    result = re.sub(r'Quote from:.*?(?=\n\n)', '', result, flags=re.DOTALL)
    # Remove standalone "Quote" lines
    result = re.sub(r'^Quote\s*$', '', result, flags=re.MULTILINE)

    return result.strip()


def clean_non_satoshi_text(text):
    """Clean non-Satoshi mailing list / forum texts."""
    # Remove email headers
    lines = text.split('\n')
    cleaned = []
    in_header = True
    for line in lines:
        stripped = line.strip()
        if in_header:
            # Skip typical email headers
            if any(stripped.startswith(h) for h in ['From:', 'To:', 'Cc:', 'Subject:', 'Date:', 'Message-ID:', 'References:', 'In-Reply-To:', 'Content-', 'MIME-', 'X-']):
                continue
            if stripped.startswith('-----'):
                in_header = False
                continue
            if not stripped:
                in_header = False
                continue
        cleaned.append(line)

    text = '\n'.join(cleaned).strip()

    # Remove signatures (lines after "-- " or "---")
    sig_match = re.search(r'\n-- ?\n', text)
    if sig_match:
        text = text[:sig_match.start()]

    # Remove quoted text
    lines = text.split('\n')
    original = []
    for line in lines:
        if line.strip().startswith('>'):
            continue
        if re.match(r'^On .+ wrote:$', line.strip()):
            continue
        original.append(line)

    return '\n'.join(original).strip()


def chunk_text(text, min_words=MIN_WORDS, max_words=MAX_WORDS):
    """Split text into chunks of roughly equal word count."""
    words = text.split()
    if len(words) <= max_words:
        if len(words) >= min_words:
            return [text]
        else:
            return []  # Too short

    chunks = []
    # Split into paragraphs first, then combine into target-sized chunks
    paragraphs = re.split(r'\n\n+', text)
    current_chunk = []
    current_word_count = 0

    for para in paragraphs:
        para_words = len(para.split())
        if para_words == 0:
            continue

        if current_word_count + para_words > max_words and current_word_count >= min_words:
            # Save current chunk and start new one
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = [para]
            current_word_count = para_words
        else:
            current_chunk.append(para)
            current_word_count += para_words

    # Don't forget the last chunk
    if current_word_count >= min_words:
        chunks.append('\n\n'.join(current_chunk))

    return chunks


def main():
    # Load raw data
    satoshi_path = os.path.join(RAW_DIR, "satoshi_raw.json")
    non_satoshi_path = os.path.join(RAW_DIR, "non_satoshi_raw.json")

    with open(satoshi_path, encoding='utf-8') as f:
        satoshi_raw = json.load(f)
    print(f"Loaded {len(satoshi_raw)} raw Satoshi items")

    with open(non_satoshi_path, encoding='utf-8') as f:
        non_satoshi_raw = json.load(f)
    print(f"Loaded {len(non_satoshi_raw)} raw non-Satoshi items")

    # Clean and chunk Satoshi texts
    satoshi_chunks = []
    for item in satoshi_raw:
        cleaned = clean_satoshi_text(item['text'])
        chunks = chunk_text(cleaned)
        for i, chunk in enumerate(chunks):
            satoshi_chunks.append({
                "text": chunk,
                "label": 1,  # 1 = Satoshi
                "label_name": "satoshi",
                "source": item['source'],
                "original_title": item.get('title', ''),
                "original_date": item.get('date', ''),
                "chunk_index": i,
                "word_count": len(chunk.split())
            })

    print(f"Satoshi: {len(satoshi_chunks)} chunks from {len(satoshi_raw)} items")

    # Clean and chunk non-Satoshi texts
    non_satoshi_chunks = []
    for item in non_satoshi_raw:
        cleaned = clean_non_satoshi_text(item['text'])
        chunks = chunk_text(cleaned)
        for i, chunk in enumerate(chunks):
            non_satoshi_chunks.append({
                "text": chunk,
                "label": 0,  # 0 = not Satoshi
                "label_name": "not_satoshi",
                "source": item['source'],
                "author": item.get('author', 'unknown'),
                "original_title": item.get('title', ''),
                "original_date": item.get('date', ''),
                "chunk_index": i,
                "word_count": len(chunk.split())
            })

    print(f"Non-Satoshi: {len(non_satoshi_chunks)} chunks from {len(non_satoshi_raw)} items")

    # Shuffle within each class
    random.shuffle(satoshi_chunks)
    random.shuffle(non_satoshi_chunks)

    # Word count stats
    sat_wc = [c['word_count'] for c in satoshi_chunks]
    nsat_wc = [c['word_count'] for c in non_satoshi_chunks]
    print(f"\nSatoshi word counts: min={min(sat_wc)}, max={max(sat_wc)}, mean={sum(sat_wc)/len(sat_wc):.0f}")
    print(f"Non-Satoshi word counts: min={min(nsat_wc)}, max={max(nsat_wc)}, mean={sum(nsat_wc)/len(nsat_wc):.0f}")

    # Create splits
    # Golden test: 20% of Satoshi (never seen during training)
    # Regular test: 10%
    # Val: 10%
    # Train: 60%

    n_sat = len(satoshi_chunks)
    n_sat_golden = max(50, int(n_sat * 0.20))
    n_sat_test = max(30, int(n_sat * 0.10))
    n_sat_val = max(30, int(n_sat * 0.10))
    n_sat_train = n_sat - n_sat_golden - n_sat_test - n_sat_val

    sat_golden = satoshi_chunks[:n_sat_golden]
    sat_test = satoshi_chunks[n_sat_golden:n_sat_golden + n_sat_test]
    sat_val = satoshi_chunks[n_sat_golden + n_sat_test:n_sat_golden + n_sat_test + n_sat_val]
    sat_train = satoshi_chunks[n_sat_golden + n_sat_test + n_sat_val:]

    # For non-Satoshi: match sizes, plus golden test
    n_nsat = len(non_satoshi_chunks)
    n_nsat_golden = max(50, int(n_nsat * 0.20))
    n_nsat_test = max(30, int(n_nsat * 0.10))
    n_nsat_val = max(30, int(n_nsat * 0.10))
    n_nsat_train = n_nsat - n_nsat_golden - n_nsat_test - n_nsat_val

    nsat_golden = non_satoshi_chunks[:n_nsat_golden]
    nsat_test = non_satoshi_chunks[n_nsat_golden:n_nsat_golden + n_nsat_test]
    nsat_val = non_satoshi_chunks[n_nsat_golden + n_nsat_test:n_nsat_golden + n_nsat_test + n_nsat_val]
    nsat_train = non_satoshi_chunks[n_nsat_golden + n_nsat_test + n_nsat_val:]

    # Combine splits
    train = sat_train + nsat_train
    val = sat_val + nsat_val
    test = sat_test + nsat_test
    golden = sat_golden + nsat_golden

    # Shuffle combined sets
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    # Don't shuffle golden - keep satoshi first, then non-satoshi for easy analysis

    # Save splits
    for name, data in [("train", train), ("val", val), ("test", test), ("golden", golden)]:
        path = os.path.join(OUT_DIR, f"{name}.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        n_sat_in = sum(1 for d in data if d['label'] == 1)
        n_nsat_in = sum(1 for d in data if d['label'] == 0)
        print(f"\n{name}: {len(data)} items (satoshi={n_sat_in}, not_satoshi={n_nsat_in})")

    # Also save golden satoshi and non-satoshi separately for easy analysis
    golden_satoshi_path = os.path.join(OUT_DIR, "golden_satoshi.json")
    golden_non_satoshi_path = os.path.join(OUT_DIR, "golden_non_satoshi.json")
    with open(golden_satoshi_path, 'w', encoding='utf-8') as f:
        json.dump(sat_golden, f, indent=2, ensure_ascii=False)
    with open(golden_non_satoshi_path, 'w', encoding='utf-8') as f:
        json.dump(nsat_golden, f, indent=2, ensure_ascii=False)

    print(f"\nGolden Satoshi: {len(sat_golden)} held-out texts")
    print(f"Golden Non-Satoshi: {len(nsat_golden)} held-out texts")

    # Author distribution in non-Satoshi data
    authors = Counter(c['author'] for c in non_satoshi_chunks if 'author' in c)
    print(f"\nTop 15 non-Satoshi authors:")
    for a, c in authors.most_common(15):
        print(f"  {a}: {c} chunks")


if __name__ == "__main__":
    main()
