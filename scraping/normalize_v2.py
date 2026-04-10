"""
V2 normalization: Fix metadata contamination.

Key changes:
1. Use ONLY BitcoinTalk posts for non-Satoshi (drop mailing list - different format/topic)
2. Strip "Quote from:" blocks aggressively from both classes
3. Remove all format artifacts so model must learn STYLE, not FORMAT
4. For Satoshi, use only BitcoinTalk posts too (drop emails - too few and different format)
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

MIN_WORDS = 50
MAX_WORDS = 400


def aggressive_clean(text):
    """Aggressively clean text to remove ALL formatting artifacts."""

    # Remove "Quote from: XXX on DATE" blocks and their content
    # These appear as: Quote from: username on Month DD, YYYY, HH:MM:SS AM/PM\n<quoted text>\n\n
    text = re.sub(r'Quote from:.*?(?=\n\n|\Z)', '', text, flags=re.DOTALL)
    text = re.sub(r'^Quote\s*$', '', text, flags=re.MULTILINE)

    # Remove lines that are just "Quote"
    text = re.sub(r'^\s*Quote\s*$', '', text, flags=re.MULTILINE)

    # Remove email-style quoted lines
    lines = text.split('\n')
    clean_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip quoted lines
        if stripped.startswith('>'):
            continue
        # Skip "On ... wrote:" lines
        if re.match(r'^On .+ wrote:$', stripped):
            continue
        # Skip "X wrote:" lines
        if re.match(r'^.{1,40} wrote:$', stripped):
            continue
        # Skip forum metadata lines
        if re.match(r'^(From|Subject|Date|Posted|Source|To|Cc|In-Reply-To|References|Message-ID|Content-Type|MIME-Version):', stripped):
            continue
        # Skip "BitcoinTalk" / "P2P Foundation" standalone lines
        if stripped in ('BitcoinTalk', 'P2P Foundation', 'Cryptography Mailing List', 'Bitcoin-list'):
            continue
        # Skip standalone numbers (post numbers)
        if re.match(r'^\d+$', stripped):
            continue
        # Skip date-only lines
        if re.match(r'^(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d', stripped):
            continue
        if re.match(r'^\d{4}-\d{2}-\d{2}', stripped):
            continue
        if 'UTC' in stripped and len(stripped) < 40:
            continue
        # Skip "satoshi" or "Satoshi Nakamoto" standalone author lines
        if stripped.lower() in ('satoshi', 'satoshi nakamoto'):
            continue
        # Skip "Re: ..." title lines at the start
        if stripped.startswith('Re: ') and len(stripped) < 100:
            continue
        # Skip "[Moderator" lines
        if stripped.startswith('[Moderator'):
            continue
        # Skip email artifacts
        if stripped.startswith('[EMAIL'):
            continue
        if stripped.startswith('-----'):
            continue

        clean_lines.append(line)

    text = '\n'.join(clean_lines)

    # Remove "Code:" blocks (forum code blocks)
    text = re.sub(r'Code:\s*\n.*?(?=\n\n|\Z)', '', text, flags=re.DOTALL)

    # Normalize whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)

    # Remove leading/trailing whitespace from each line
    lines = text.split('\n')
    text = '\n'.join(line.strip() for line in lines)

    return text.strip()


def chunk_text(text, min_words=MIN_WORDS, max_words=MAX_WORDS):
    words = text.split()
    if len(words) <= max_words:
        if len(words) >= min_words:
            return [text]
        else:
            return []

    chunks = []
    paragraphs = re.split(r'\n\n+', text)
    current_chunk = []
    current_word_count = 0

    for para in paragraphs:
        para_words = len(para.split())
        if para_words == 0:
            continue
        if current_word_count + para_words > max_words and current_word_count >= min_words:
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = [para]
            current_word_count = para_words
        else:
            current_chunk.append(para)
            current_word_count += para_words

    if current_word_count >= min_words:
        chunks.append('\n\n'.join(current_chunk))

    return chunks


def main():
    # Load raw data
    with open(os.path.join(RAW_DIR, "satoshi_raw.json"), encoding='utf-8') as f:
        satoshi_raw = json.load(f)
    with open(os.path.join(RAW_DIR, "non_satoshi_raw.json"), encoding='utf-8') as f:
        non_satoshi_raw = json.load(f)

    # FILTER: Only BitcoinTalk posts for both classes
    satoshi_btc = [d for d in satoshi_raw if 'bitcointalk' in d['source']]
    non_satoshi_btc = [d for d in non_satoshi_raw if d['source'] == 'bitcointalk']

    print(f"Satoshi BitcoinTalk posts: {len(satoshi_btc)}")
    print(f"Non-Satoshi BitcoinTalk posts: {len(non_satoshi_btc)}")

    # Clean and chunk
    satoshi_chunks = []
    for item in satoshi_btc:
        cleaned = aggressive_clean(item['text'])
        chunks = chunk_text(cleaned)
        for i, chunk in enumerate(chunks):
            satoshi_chunks.append({
                "text": chunk,
                "label": 1,
                "label_name": "satoshi",
                "source": "bitcointalk",
                "word_count": len(chunk.split())
            })

    non_satoshi_chunks = []
    for item in non_satoshi_btc:
        cleaned = aggressive_clean(item['text'])
        chunks = chunk_text(cleaned)
        for i, chunk in enumerate(chunks):
            non_satoshi_chunks.append({
                "text": chunk,
                "label": 0,
                "label_name": "not_satoshi",
                "source": "bitcointalk",
                "author": item.get('author', 'unknown'),
                "word_count": len(chunk.split())
            })

    print(f"\nSatoshi chunks: {len(satoshi_chunks)}")
    print(f"Non-Satoshi chunks: {len(non_satoshi_chunks)}")

    # Verify cleaning worked - check for format differences
    def check_artifacts(chunks, name):
        has_quote = sum(1 for c in chunks if 'Quote from:' in c['text'])
        has_wrote = sum(1 for c in chunks if 'wrote:' in c['text'])
        has_bracket = sum(1 for c in chunks if '[' in c['text'][:50])
        has_bitcoin = sum(1 for c in chunks if 'bitcoin' in c['text'].lower())
        avg_newlines = sum(c['text'].count('\n') for c in chunks) / max(len(chunks), 1)
        avg_len = sum(len(c['text']) for c in chunks) / max(len(chunks), 1)
        print(f"\n{name} format check:")
        print(f"  'Quote from:': {has_quote} ({has_quote/len(chunks)*100:.0f}%)")
        print(f"  'wrote:': {has_wrote} ({has_wrote/len(chunks)*100:.0f}%)")
        print(f"  Bracket in first 50: {has_bracket} ({has_bracket/len(chunks)*100:.0f}%)")
        print(f"  Mentions bitcoin: {has_bitcoin} ({has_bitcoin/len(chunks)*100:.0f}%)")
        print(f"  Avg newlines: {avg_newlines:.1f}")
        print(f"  Avg length: {avg_len:.0f} chars")

    check_artifacts(satoshi_chunks, "SATOSHI")
    check_artifacts(non_satoshi_chunks, "NON-SATOSHI")

    # Show samples
    print("\n=== SATOSHI SAMPLE ===")
    print(satoshi_chunks[0]['text'][:300])
    print("\n=== NON-SATOSHI SAMPLE ===")
    print(non_satoshi_chunks[0]['text'][:300])

    # Shuffle
    random.shuffle(satoshi_chunks)
    random.shuffle(non_satoshi_chunks)

    # Split: 60% train, 10% val, 10% test, 20% golden
    def split_data(chunks):
        n = len(chunks)
        n_golden = max(30, int(n * 0.20))
        n_test = max(20, int(n * 0.10))
        n_val = max(20, int(n * 0.10))
        golden = chunks[:n_golden]
        test = chunks[n_golden:n_golden + n_test]
        val = chunks[n_golden + n_test:n_golden + n_test + n_val]
        train = chunks[n_golden + n_test + n_val:]
        return train, val, test, golden

    sat_train, sat_val, sat_test, sat_golden = split_data(satoshi_chunks)
    nsat_train, nsat_val, nsat_test, nsat_golden = split_data(non_satoshi_chunks)

    train = sat_train + nsat_train
    val = sat_val + nsat_val
    test = sat_test + nsat_test
    golden = sat_golden + nsat_golden

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    # Save
    for name, data in [("train", train), ("val", val), ("test", test), ("golden", golden)]:
        path = os.path.join(OUT_DIR, f"{name}.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        n_sat = sum(1 for d in data if d['label'] == 1)
        n_nsat = sum(1 for d in data if d['label'] == 0)
        print(f"\n{name}: {len(data)} items (satoshi={n_sat}, not_satoshi={n_nsat})")

    # Save golden splits separately
    with open(os.path.join(OUT_DIR, "golden_satoshi.json"), 'w', encoding='utf-8') as f:
        json.dump(sat_golden, f, indent=2, ensure_ascii=False)
    with open(os.path.join(OUT_DIR, "golden_non_satoshi.json"), 'w', encoding='utf-8') as f:
        json.dump(nsat_golden, f, indent=2, ensure_ascii=False)

    print(f"\nGolden Satoshi: {len(sat_golden)}")
    print(f"Golden Non-Satoshi: {len(nsat_golden)}")


if __name__ == "__main__":
    main()
