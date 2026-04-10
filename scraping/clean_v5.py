"""
V5 Data Pipeline: Just clean words. Nothing else.

Goal: Two JSON files:
  - satoshi_clean.json: List of texts containing ONLY Satoshi's own words
  - non_satoshi_clean.json: List of texts containing ONLY non-Satoshi's own words

Same format, same structure, no headers, no metadata, no quotes from
other people, no formatting artifacts. Just prose.
"""

import json
import os
import re
import sys

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
OUT_DIR = os.path.join(DATA_DIR, "v5_clean")
os.makedirs(OUT_DIR, exist_ok=True)


def extract_own_words(raw_text, is_nakamoto_institute=False):
    """
    Extract ONLY the author's own words from a raw post.
    Remove everything that isn't original prose by the author.
    """

    text = raw_text

    # === STEP 1: Encoding normalization ===
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = text.replace('\xa0', ' ')
    text = text.replace('\u200b', '').replace('\u200c', '').replace('\ufeff', '')

    # === STEP 2: Strip Nakamoto Institute wrapper ===
    if is_nakamoto_institute:
        # The format is always:
        # "SourceName\nTitle\nNumber\nFrom:\nauthor\nSubject:\nTitle\nDate:\nDateString\n[actual content]\nView original\nView in thread"
        lines = text.split('\n')

        # Find where the actual content starts by skipping the header block
        content_start = 0
        found_date_line = False
        for i, line in enumerate(lines):
            s = line.strip()
            # The date line containing "UTC" marks the end of the header
            if 'UTC' in s and len(s) < 60:
                content_start = i + 1
                found_date_line = True
                break

        # If we didn't find UTC, try to find content after "Date:" line
        if not found_date_line:
            for i, line in enumerate(lines):
                if line.strip().startswith('Date:'):
                    # Skip the date value line too
                    content_start = i + 2
                    break

        # Find where content ends (before NI navigation)
        content_end = len(lines)
        for i in range(len(lines) - 1, max(content_start, 0), -1):
            s = lines[i].strip()
            if s in ('View original', 'View in thread', 'Previous', 'Next',
                     'Back to emails', 'Back to posts', ''):
                content_end = i
            else:
                break

        text = '\n'.join(lines[content_start:content_end])

    # === STEP 3: Remove "Quote from:" blocks ===
    # BitcoinTalk format: "Quote from: USER on DATE\n<quoted text>"
    # The quoted text continues until a blank line or another quote
    lines = text.split('\n')
    clean_lines = []
    in_quote = False

    for line in lines:
        s = line.strip()

        # Start of a quote block
        if s.startswith('Quote from:'):
            in_quote = True
            continue

        # End of quote block: blank line after being in a quote
        if in_quote and s == '':
            in_quote = False
            continue

        # Still in a quote block - skip
        if in_quote:
            continue

        clean_lines.append(line)

    text = '\n'.join(clean_lines)

    # === STEP 4: Remove email-style quoted lines ===
    lines = text.split('\n')
    clean_lines = []
    for line in lines:
        s = line.strip()
        if s.startswith('>'):
            continue
        if re.match(r'^On .{10,80} wrote:\s*$', s):
            continue
        if re.match(r'^.{1,30} wrote:\s*$', s):
            continue
        if s == 'Quote':
            continue
        clean_lines.append(line)
    text = '\n'.join(clean_lines)

    # === STEP 5: Remove signatures ===
    sig = re.search(r'\n-- ?\n', text)
    if sig:
        text = text[:sig.start()]

    # === STEP 6: Remove URLs ===
    text = re.sub(r'https?://\S+', '', text)

    # === STEP 7: Remove code blocks ===
    text = re.sub(r'Code:\s*\n.*?(?=\n\n|\Z)', '', text, flags=re.DOTALL)

    # === STEP 8: Remove remaining metadata lines ===
    lines = text.split('\n')
    clean_lines = []
    for line in lines:
        s = line.strip()
        # Skip standalone metadata
        if s in ('BitcoinTalk', 'P2P Foundation', 'Cryptography Mailing List',
                 'Bitcoin-list', 'P2P Research', 'View original', 'View in thread',
                 'Satoshi Nakamoto Institute'):
            continue
        if s.lower() in ('satoshi', 'satoshi nakamoto'):
            continue
        if re.match(r'^\d{1,4}$', s):  # standalone numbers
            continue
        if re.match(r'^From:\s*$', s) or re.match(r'^Subject:\s*$', s) or re.match(r'^Date:\s*$', s):
            continue
        # Date lines
        if re.match(r'^(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}', s):
            continue
        if 'UTC' in s and len(s) < 50 and re.search(r'\d{2}:\d{2}', s):
            continue
        # Email headers
        if re.match(r'^(From|To|Cc|Subject|Date|Message-ID|References|In-Reply-To|Content-|MIME-|X-|Return-Path|Received|Delivered):', s):
            continue
        # Bracketed annotations
        if re.match(r'^\[.{1,38}\]$', s):
            continue
        clean_lines.append(line)
    text = '\n'.join(clean_lines)

    # === STEP 9: Normalize whitespace and punctuation ===
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    # Strip each line
    lines = text.split('\n')
    lines = [line.strip() for line in lines]
    # Collapse consecutive blank lines to one
    clean = []
    prev_blank = False
    for line in lines:
        if not line:
            if not prev_blank:
                clean.append('')
                prev_blank = True
        else:
            clean.append(line)
            prev_blank = False
    text = '\n'.join(clean).strip()

    # Normalize punctuation
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2013', '-').replace('\u2014', '-')
    text = text.replace('\u2026', '...')

    return text


def main():
    # === PROCESS SATOSHI ===
    print("=== PROCESSING SATOSHI ===", flush=True)
    # Use V5 expanded data if available, otherwise V4
    sat_file = os.path.join(RAW_DIR, "v5_all_satoshi.json")
    if not os.path.exists(sat_file):
        sat_file = os.path.join(RAW_DIR, "v4_satoshi.json")
    print(f"  Loading from: {os.path.basename(sat_file)}", flush=True)
    with open(sat_file, encoding='utf-8') as f:
        sat_raw = json.load(f)

    satoshi_texts = []
    for item in sat_raw:
        # NI sources have the wrapper; other sources (malmi, wei_dai, etc.) don't
        is_ni = item['source'] in ('bitcointalk', 'p2pfoundation', 'email_cryptography', 'email_bitcoin-list', 'email_p2p-research')
        cleaned = extract_own_words(item['text'], is_nakamoto_institute=is_ni)
        word_count = len(cleaned.split())
        if word_count < 15:
            continue
        satoshi_texts.append({
            "text": cleaned,
            "word_count": word_count,
            "source": item['source'],
        })

    print(f"  Raw: {len(sat_raw)} -> Clean: {len(satoshi_texts)}", flush=True)

    # === PROCESS NON-SATOSHI ===
    print("\n=== PROCESSING NON-SATOSHI ===", flush=True)
    with open(os.path.join(RAW_DIR, "v4_non_satoshi.json"), encoding='utf-8') as f:
        nsat_raw = json.load(f)

    non_satoshi_texts = []
    for item in nsat_raw:
        cleaned = extract_own_words(item['text'], is_nakamoto_institute=False)
        word_count = len(cleaned.split())
        if word_count < 15:
            continue
        non_satoshi_texts.append({
            "text": cleaned,
            "word_count": word_count,
            "source": item['source'],
            "author": item.get('author', 'unknown'),
        })

    print(f"  Raw: {len(nsat_raw)} -> Clean: {len(non_satoshi_texts)}", flush=True)

    # === SAVE ===
    sat_path = os.path.join(OUT_DIR, "satoshi_clean.json")
    nsat_path = os.path.join(OUT_DIR, "non_satoshi_clean.json")
    with open(sat_path, 'w', encoding='utf-8') as f:
        json.dump(satoshi_texts, f, indent=2, ensure_ascii=False)
    with open(nsat_path, 'w', encoding='utf-8') as f:
        json.dump(non_satoshi_texts, f, indent=2, ensure_ascii=False)

    # === SHOW SAMPLES ===
    print("\n=== SATOSHI SAMPLES ===", flush=True)
    import random
    random.seed(42)
    for d in random.sample(satoshi_texts, min(5, len(satoshi_texts))):
        print(f"\n--- {d['word_count']} words | {d['source']} ---")
        print(d['text'][:300])

    print("\n\n=== NON-SATOSHI SAMPLES ===", flush=True)
    for d in random.sample(non_satoshi_texts, min(5, len(non_satoshi_texts))):
        print(f"\n--- {d['word_count']} words | {d['source']} | {d['author']} ---")
        print(d['text'][:300])

    # === QUALITY CHECKS ===
    print("\n\n=== QUALITY CHECKS ===", flush=True)

    # Check for NI chrome leakage
    for phrase in ['View original', 'View in thread', 'From:', 'Subject:', 'Date:', 'Satoshi Nakamoto Institute']:
        c = sum(1 for d in satoshi_texts if phrase in d['text'])
        if c > 0:
            print(f"  WARNING: {c} Satoshi texts still contain '{phrase}'")

    # Check for Quote from leakage
    c = sum(1 for d in non_satoshi_texts if 'Quote from:' in d['text'])
    if c > 0:
        print(f"  WARNING: {c} non-Satoshi texts still contain 'Quote from:'")

    # Check for other people's words in Satoshi
    c = sum(1 for d in satoshi_texts if d['text'].strip().startswith('Quote'))
    if c > 0:
        print(f"  WARNING: {c} Satoshi texts start with 'Quote'")

    # Word count distributions
    sat_wc = [d['word_count'] for d in satoshi_texts]
    nsat_wc = [d['word_count'] for d in non_satoshi_texts]
    print(f"\n  Satoshi: {len(satoshi_texts)} texts, median {sorted(sat_wc)[len(sat_wc)//2]} words, mean {sum(sat_wc)/len(sat_wc):.0f}")
    print(f"  Non-Sat: {len(non_satoshi_texts)} texts, median {sorted(nsat_wc)[len(nsat_wc)//2]} words, mean {sum(nsat_wc)/len(nsat_wc):.0f}")

    # Are any satoshi texts suspiciously starting with someone else's name?
    suspicious = []
    for d in satoshi_texts:
        first_line = d['text'].split('\n')[0].strip()
        if ',' in first_line[:30] and first_line[0].isupper() and len(first_line.split(',')[0].split()) <= 3:
            suspicious.append(first_line[:80])
    if suspicious:
        print(f"\n  Satoshi texts that might start with someone else addressing Satoshi:")
        for s in suspicious[:5]:
            print(f"    '{s}'")

    # === DEDUPLICATION ===
    print("\n=== DEDUPLICATION ===", flush=True)
    seen = set()
    deduped_sat = []
    for d in satoshi_texts:
        key = d['text'][:200]
        if key not in seen:
            seen.add(key)
            deduped_sat.append(d)
    print(f"  Satoshi: {len(satoshi_texts)} -> {len(deduped_sat)} (removed {len(satoshi_texts)-len(deduped_sat)} dupes)")
    satoshi_texts = deduped_sat

    seen = set()
    deduped_nsat = []
    for d in non_satoshi_texts:
        key = d['text'][:200]
        if key not in seen:
            seen.add(key)
            deduped_nsat.append(d)
    print(f"  Non-Sat: {len(non_satoshi_texts)} -> {len(deduped_nsat)} (removed {len(non_satoshi_texts)-len(deduped_nsat)} dupes)")
    non_satoshi_texts = deduped_nsat

    # === CROSS-CONTAMINATION CHECK ===
    sat_keys = set(d['text'][:200] for d in satoshi_texts)
    non_satoshi_texts = [d for d in non_satoshi_texts if d['text'][:200] not in sat_keys]
    print(f"  Removed cross-contaminated texts from non-Satoshi", flush=True)

    # === RE-SAVE ===
    with open(sat_path, 'w', encoding='utf-8') as f:
        json.dump(satoshi_texts, f, indent=2, ensure_ascii=False)
    with open(nsat_path, 'w', encoding='utf-8') as f:
        json.dump(non_satoshi_texts, f, indent=2, ensure_ascii=False)

    print(f"\n  FINAL: {len(satoshi_texts)} Satoshi, {len(non_satoshi_texts)} Non-Satoshi")
    print(f"  Saved to: {sat_path}")
    print(f"  Saved to: {nsat_path}")


if __name__ == "__main__":
    main()
