"""
V4 Satoshi scraper: Re-scrape all writings WITH thread IDs.
For each BitcoinTalk post, extract the original thread ID so we can
scrape non-Satoshi replies from the same threads.
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import os
import re
import sys

BASE = "https://satoshi.nakamotoinstitute.org"
OUTPUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw", "v4_satoshi.json")
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (research project - satoshi stylometry v4)"}


def scrape_page(url):
    """Scrape a single Nakamoto Institute page."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return None
        return BeautifulSoup(resp.text, 'html.parser')
    except Exception as e:
        print(f"  ERROR {url}: {e}", file=sys.stderr)
        return None


def extract_content(soup):
    """Extract text content from a Nakamoto Institute page."""
    main = soup.select_one('main')
    if not main:
        return None, None, None

    # Title
    h1 = soup.find('h1')
    title = h1.get_text().strip() if h1 else None

    # Date
    time_el = soup.find('time')
    date = time_el.get('datetime', time_el.get_text()) if time_el else None

    # Content from main tag
    content = main.get_text(separator='\n')
    return content, title, date


def extract_thread_id(soup):
    """Extract the original BitcoinTalk thread ID from a Nakamoto Institute post page."""
    # Look for links to bitcointalk.org
    for a in soup.find_all('a', href=True):
        href = a['href']
        if 'bitcointalk.org' in href and 'topic=' in href:
            # Extract topic ID: ?topic=5.msg28#msg28 -> 5
            match = re.search(r'topic=(\d+)', href)
            if match:
                return int(match.group(1))
    return None


def main():
    all_items = []
    thread_ids = set()

    # 1. BitcoinTalk posts (5-543) - with thread ID extraction
    print("=== Scraping BitcoinTalk posts with thread IDs ===", flush=True)
    for i in range(5, 544):
        url = f"{BASE}/posts/bitcointalk/{i}/"
        soup = scrape_page(url)
        if not soup:
            continue

        content, title, date = extract_content(soup)
        if not content:
            continue

        thread_id = extract_thread_id(soup)
        if thread_id:
            thread_ids.add(thread_id)

        all_items.append({
            "source": "bitcointalk",
            "number": i,
            "url": url,
            "title": title,
            "date": date,
            "text": content.strip(),
            "thread_id": thread_id,
            "author": "satoshi"
        })

        if i % 50 == 0:
            print(f"  {i}/543 posts ({len(thread_ids)} unique threads)", flush=True)
        time.sleep(0.25)

    print(f"  -> {sum(1 for x in all_items if x['source']=='bitcointalk')} posts, {len(thread_ids)} unique threads", flush=True)

    # 2. P2P Foundation posts
    print("=== Scraping P2P Foundation posts ===", flush=True)
    for i in range(1, 10):
        url = f"{BASE}/posts/p2pfoundation/{i}/"
        soup = scrape_page(url)
        if not soup:
            continue
        content, title, date = extract_content(soup)
        if content:
            all_items.append({
                "source": "p2pfoundation",
                "number": i, "url": url, "title": title,
                "date": date, "text": content.strip(),
                "thread_id": None, "author": "satoshi"
            })
        time.sleep(0.25)

    # 3. Emails - Cryptography list
    print("=== Scraping Cryptography emails ===", flush=True)
    for i in range(1, 30):
        for list_name in ["cryptography", "bitcoin-list", "p2p-research"]:
            url = f"{BASE}/emails/{list_name}/{i}/"
            soup = scrape_page(url)
            if not soup:
                continue
            content, title, date = extract_content(soup)
            if content:
                all_items.append({
                    "source": f"email_{list_name}",
                    "number": i, "url": url, "title": title,
                    "date": date, "text": content.strip(),
                    "thread_id": None, "author": "satoshi"
                })
            time.sleep(0.2)

    # Save
    with open(OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(all_items, f, indent=2, ensure_ascii=False)

    # Also save thread IDs separately for the non-Satoshi scraper
    thread_ids_path = os.path.join(os.path.dirname(OUTPUT), "v4_satoshi_thread_ids.json")
    with open(thread_ids_path, 'w') as f:
        json.dump(sorted(thread_ids), f)

    print(f"\n=== DONE ===", flush=True)
    print(f"Total items: {len(all_items)}", flush=True)
    sources = {}
    for item in all_items:
        src = item['source']
        sources[src] = sources.get(src, 0) + 1
    for src, count in sorted(sources.items()):
        print(f"  {src}: {count}", flush=True)
    print(f"Unique BitcoinTalk thread IDs: {len(thread_ids)}", flush=True)
    print(f"Thread IDs: {sorted(thread_ids)[:20]}... ", flush=True)


if __name__ == "__main__":
    main()
