"""
V4 Non-Satoshi scraper: Collect from the SAME threads/conversations Satoshi posted in.
This eliminates topic as a shortcut — both classes discuss the exact same topics.

Sources:
1. BitcoinTalk: same threads as Satoshi (primary, ~80-90% of data)
2. BitcoinTalk: supplemental early threads
3. Crypto mailing list: Bitcoin discussion threads only
4. Bitcoin-list mailing list
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
OUTPUT = os.path.join(DATA_DIR, "v4_non_satoshi.json")
os.makedirs(DATA_DIR, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (research project)"}


def scrape_bitcointalk_thread(topic_id, max_pages=10):
    """Scrape ALL non-Satoshi posts from a BitcoinTalk thread."""
    posts = []
    for page in range(max_pages):
        start = page * 20
        url = f"https://bitcointalk.org/index.php?topic={topic_id}.{start}"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code != 200:
                break

            soup = BeautifulSoup(resp.text, 'html.parser')
            post_divs = soup.find_all('div', class_='post')
            if not post_divs:
                break

            found = 0
            for pd in post_divs:
                text = pd.get_text(separator='\n').strip()
                if len(text) < 30:
                    continue

                # Find author
                author = "unknown"
                poster_td = pd.find_previous('td', class_='poster_info')
                if poster_td:
                    a = poster_td.find('a')
                    if a:
                        author = a.get_text().strip()

                # Skip Satoshi
                if author.lower() == 'satoshi':
                    continue

                posts.append({
                    "source": "bitcointalk",
                    "author": author,
                    "thread_id": topic_id,
                    "text": text,
                })
                found += 1

            if found == 0:
                break
            time.sleep(1.0)

        except Exception as e:
            time.sleep(2)
            continue

    return posts


def scrape_mail_archive_bitcoin_threads():
    """Scrape non-Satoshi posts from the Bitcoin discussion threads on cryptography@metzdowd.com.
    Only the Bitcoin-related threads (topic-matched to Satoshi's emails)."""

    SESSION = requests.Session()
    SESSION.headers.update(HEADERS)
    posts = []

    # These are the Bitcoin-related messages from the cryptography mailing list
    # Identified from our earlier research: msgs around 09959-10210
    # We only want the Bitcoin threads (subject contains "Bitcoin")
    bitcoin_msgs = list(range(9959, 10220))

    print(f"  Checking {len(bitcoin_msgs)} mailing list messages...", flush=True)

    for msg_num in bitcoin_msgs:
        url = f"https://www.mail-archive.com/cryptography@metzdowd.com/msg{msg_num:05d}.html"
        try:
            resp = SESSION.get(url, timeout=8)
            if resp.status_code != 200:
                continue

            soup = BeautifulSoup(resp.text, 'html.parser')

            # Check subject contains "Bitcoin" or "bitcoin"
            h1 = soup.find('h1')
            if not h1:
                continue
            subject = h1.get_text().strip()
            if 'bitcoin' not in subject.lower() and 'btc' not in subject.lower():
                continue

            # Author
            author = "unknown"
            msg_head = soup.find('div', class_='msgHead')
            if msg_head:
                for link in msg_head.find_all('a'):
                    href = link.get('href', '')
                    if 'info/' in href:
                        name = link.get_text().strip()
                        if name and '@' not in name and len(name) > 1:
                            author = name
                            break

            # Skip Satoshi
            if 'satoshi' in author.lower():
                continue

            # Content
            msg_body = soup.find('div', class_='msgBody')
            if not msg_body:
                continue

            text = msg_body.get_text(separator='\n').strip()
            if len(text) < 50:
                continue

            posts.append({
                "source": "email_cryptography",
                "author": author,
                "title": subject,
                "text": text,
                "msg_num": msg_num,
            })

        except:
            continue
        time.sleep(0.2)

    return posts


def scrape_bitcoin_list():
    """Scrape non-Satoshi posts from the bitcoin-list mailing list."""
    SESSION = requests.Session()
    SESSION.headers.update(HEADERS)
    posts = []

    list_name = "bitcoin-list@lists.sourceforge.net"
    # Try a range of messages
    for msg_num in range(1, 500):
        url = f"https://www.mail-archive.com/{list_name}/msg{msg_num:05d}.html"
        try:
            resp = SESSION.get(url, timeout=8)
            if resp.status_code != 200:
                continue

            soup = BeautifulSoup(resp.text, 'html.parser')

            # Author
            author = "unknown"
            msg_head = soup.find('div', class_='msgHead')
            if msg_head:
                for link in msg_head.find_all('a'):
                    href = link.get('href', '')
                    if 'info/' in href:
                        name = link.get_text().strip()
                        if name and '@' not in name and len(name) > 1:
                            author = name
                            break

            if 'satoshi' in author.lower():
                continue

            # Content
            msg_body = soup.find('div', class_='msgBody')
            if not msg_body:
                continue
            text = msg_body.get_text(separator='\n').strip()
            if len(text) < 50:
                continue

            posts.append({
                "source": "email_bitcoin-list",
                "author": author,
                "text": text,
                "msg_num": msg_num,
            })

        except:
            continue
        time.sleep(0.2)

    return posts


def main():
    all_posts = []

    # 1. Load Satoshi thread IDs
    thread_ids_path = os.path.join(DATA_DIR, "v4_satoshi_thread_ids.json")
    if os.path.exists(thread_ids_path):
        with open(thread_ids_path) as f:
            satoshi_thread_ids = json.load(f)
        print(f"Loaded {len(satoshi_thread_ids)} Satoshi thread IDs", flush=True)
    else:
        print("WARNING: No thread IDs file found. Run scrape_v4_satoshi.py first!", flush=True)
        satoshi_thread_ids = []

    # 2. Scrape same-thread BitcoinTalk posts
    print(f"\n=== BitcoinTalk same-threads ({len(satoshi_thread_ids)} threads) ===", flush=True)
    same_thread_count = 0
    for i, tid in enumerate(satoshi_thread_ids):
        posts = scrape_bitcointalk_thread(tid, max_pages=5)
        for p in posts:
            p['same_thread_as_satoshi'] = True
        all_posts.extend(posts)
        same_thread_count += len(posts)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(satoshi_thread_ids)} threads, {same_thread_count} posts", flush=True)
        time.sleep(0.5)

    print(f"  -> {same_thread_count} same-thread posts", flush=True)

    # 3. Supplemental BitcoinTalk threads (early era)
    print("\n=== BitcoinTalk supplemental threads ===", flush=True)
    supplemental_topics = [t for t in range(1, 2001, 25) if t not in satoshi_thread_ids][:40]
    supp_count = 0
    for tid in supplemental_topics:
        posts = scrape_bitcointalk_thread(tid, max_pages=3)
        for p in posts:
            p['same_thread_as_satoshi'] = False
        all_posts.extend(posts)
        supp_count += len(posts)
        time.sleep(0.5)

    print(f"  -> {supp_count} supplemental posts", flush=True)

    # 4. Crypto mailing list Bitcoin threads
    print("\n=== Crypto mailing list (Bitcoin threads only) ===", flush=True)
    ml_posts = scrape_mail_archive_bitcoin_threads()
    all_posts.extend(ml_posts)
    print(f"  -> {len(ml_posts)} mailing list posts", flush=True)

    # 5. Bitcoin-list mailing list
    print("\n=== Bitcoin-list mailing list ===", flush=True)
    bl_posts = scrape_bitcoin_list()
    all_posts.extend(bl_posts)
    print(f"  -> {len(bl_posts)} bitcoin-list posts", flush=True)

    # Save
    with open(OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(all_posts, f, indent=2, ensure_ascii=False)

    print(f"\n=== DONE ===", flush=True)
    print(f"Total: {len(all_posts)}", flush=True)
    sources = {}
    for p in all_posts:
        sources[p['source']] = sources.get(p['source'], 0) + 1
    for s, c in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"  {s}: {c}", flush=True)

    same = sum(1 for p in all_posts if p.get('same_thread_as_satoshi'))
    print(f"  Same-thread as Satoshi: {same}", flush=True)


if __name__ == "__main__":
    main()
