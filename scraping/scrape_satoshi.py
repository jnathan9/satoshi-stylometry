"""
Scrape all Satoshi Nakamoto writings from satoshi.nakamotoinstitute.org
- BitcoinTalk posts: /posts/bitcointalk/5/ through /posts/bitcointalk/543/
- P2P Foundation posts: /posts/p2pfoundation/1/ through /posts/p2pfoundation/4/
- Emails: /emails/cryptography/, /emails/bitcoin-list/, /emails/p2p-research/
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import os
import re
import sys

BASE = "https://satoshi.nakamotoinstitute.org"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
os.makedirs(OUTPUT_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (research project - satoshi stylometry)"
}

def clean_text(text):
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()

def scrape_page(url, source_type):
    """Scrape a single page from the Nakamoto Institute."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return None

        soup = BeautifulSoup(resp.text, 'html.parser')

        # Get content from main tag
        main = soup.select_one('main')
        if not main:
            return None

        content = main.get_text(separator='\n')

        # Get title
        title = None
        h1 = soup.find('h1')
        if h1:
            title = h1.get_text().strip()
            # Remove title from content to avoid duplication
            content = content.replace(title, '', 1)

        # Get date from time element
        date = None
        time_el = soup.find('time')
        if time_el:
            date = time_el.get('datetime', time_el.get_text())

        content = clean_text(content)

        if len(content) < 10:
            return None

        return {
            "source": source_type,
            "url": url,
            "title": title,
            "date": date,
            "text": content,
            "author": "satoshi"
        }
    except Exception as e:
        print(f"  ERROR {url}: {e}", file=sys.stderr)
        return None

def main():
    all_items = []

    # 1. BitcoinTalk posts (5-543)
    print("=== Scraping BitcoinTalk posts ===", flush=True)
    for i in range(5, 544):
        url = f"{BASE}/posts/bitcointalk/{i}/"
        item = scrape_page(url, "post_bitcointalk")
        if item:
            item['number'] = i
            all_items.append(item)
        if i % 50 == 0:
            print(f"  Progress: {i}/543 bitcointalk posts ({len(all_items)} collected)", flush=True)
        time.sleep(0.25)

    bt_count = len(all_items)
    print(f"  -> {bt_count} BitcoinTalk posts", flush=True)

    # 2. P2P Foundation posts (1-4, plus extras)
    print("=== Scraping P2P Foundation posts ===", flush=True)
    for i in range(1, 10):
        url = f"{BASE}/posts/p2pfoundation/{i}/"
        item = scrape_page(url, "post_p2pfoundation")
        if item:
            item['number'] = i
            all_items.append(item)
        time.sleep(0.25)

    p2p_count = len(all_items) - bt_count
    print(f"  -> {p2p_count} P2P Foundation posts", flush=True)

    # 3. Emails - Cryptography list (1-18)
    print("=== Scraping Cryptography emails ===", flush=True)
    pre_count = len(all_items)
    for i in range(1, 30):
        url = f"{BASE}/emails/cryptography/{i}/"
        item = scrape_page(url, "email_cryptography")
        if item:
            item['number'] = i
            all_items.append(item)
        time.sleep(0.25)

    print(f"  -> {len(all_items) - pre_count} cryptography emails", flush=True)

    # 4. Emails - Bitcoin-list
    print("=== Scraping Bitcoin-list emails ===", flush=True)
    pre_count = len(all_items)
    for i in range(1, 40):
        url = f"{BASE}/emails/bitcoin-list/{i}/"
        item = scrape_page(url, "email_bitcoin_list")
        if item:
            item['number'] = i
            all_items.append(item)
        time.sleep(0.25)

    print(f"  -> {len(all_items) - pre_count} bitcoin-list emails", flush=True)

    # 5. Emails - P2P Research
    print("=== Scraping P2P Research emails ===", flush=True)
    pre_count = len(all_items)
    for i in range(1, 15):
        url = f"{BASE}/emails/p2p-research/{i}/"
        item = scrape_page(url, "email_p2p_research")
        if item:
            item['number'] = i
            all_items.append(item)
        time.sleep(0.25)

    print(f"  -> {len(all_items) - pre_count} P2P Research emails", flush=True)

    # Save
    outpath = os.path.join(OUTPUT_DIR, "satoshi_raw.json")
    with open(outpath, 'w', encoding='utf-8') as f:
        json.dump(all_items, f, indent=2, ensure_ascii=False)

    print(f"\n=== DONE ===", flush=True)
    print(f"Total Satoshi items scraped: {len(all_items)}", flush=True)
    print(f"Saved to: {outpath}", flush=True)

    # Print breakdown
    sources = {}
    for item in all_items:
        src = item['source']
        sources[src] = sources.get(src, 0) + 1
    for src, count in sorted(sources.items()):
        print(f"  {src}: {count}", flush=True)

    # Print text length stats
    lengths = [len(item['text']) for item in all_items]
    print(f"\nText length stats:")
    print(f"  Min: {min(lengths)} chars")
    print(f"  Max: {max(lengths)} chars")
    print(f"  Mean: {sum(lengths)/len(lengths):.0f} chars")
    print(f"  Median: {sorted(lengths)[len(lengths)//2]} chars")

if __name__ == "__main__":
    main()
