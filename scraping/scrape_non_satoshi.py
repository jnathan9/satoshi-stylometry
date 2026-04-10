"""
Scrape non-Satoshi writings - streamlined version.
Focus on the cryptography mailing list around the Bitcoin discussion era.
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import os
import re
import sys

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
os.makedirs(OUTPUT_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (research project)"
}

def clean_text(text):
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()

def remove_quoted_text(text):
    lines = text.split('\n')
    original = []
    for line in lines:
        s = line.strip()
        if s.startswith('>'):
            continue
        if re.match(r'^On .+ wrote:$', s):
            continue
        original.append(line)
    return '\n'.join(original).strip()


def scrape_mail_archive_range(start, end, list_name="cryptography@metzdowd.com"):
    """Scrape a range of messages from mail-archive.com."""
    posts = []
    for msg_num in range(start, end + 1):
        url = f"https://www.mail-archive.com/{list_name}/msg{msg_num:05d}.html"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=8)
            if resp.status_code != 200:
                continue

            soup = BeautifulSoup(resp.text, 'html.parser')

            # Author - try multiple methods
            author = "unknown"
            # Method 1: Look for the sender info
            msg_head = soup.find('div', class_='msgHead')
            if msg_head:
                # Find text that looks like an author name
                links = msg_head.find_all('a')
                for link in links:
                    href = link.get('href', '')
                    if 'mailto:' in href or 'info/' in href:
                        name = link.get_text().strip()
                        if name and '@' not in name and len(name) > 1:
                            author = name
                            break

            # Skip Satoshi
            if 'satoshi' in author.lower():
                continue

            # Subject
            title = None
            h1 = soup.find('h1')
            if h1:
                title = h1.get_text().strip()

            # Content
            msg_body = soup.find('div', class_='msgBody')
            if not msg_body:
                continue

            text = msg_body.get_text(separator='\n')
            text = remove_quoted_text(text)
            text = clean_text(text)

            if len(text) < 50:
                continue

            posts.append({
                "source": f"mailing_list_{list_name.split('@')[0]}",
                "author": author,
                "title": title,
                "text": text,
                "url": url,
                "msg_num": msg_num
            })

        except Exception as e:
            continue

        time.sleep(0.2)

    return posts


def scrape_bitcointalk_thread(topic_id, max_pages=3):
    """Scrape non-Satoshi posts from a bitcointalk thread."""
    posts = []
    for page in range(max_pages):
        start = page * 20
        url = f"https://bitcointalk.org/index.php?topic={topic_id}.{start}"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code != 200:
                break

            soup = BeautifulSoup(resp.text, 'html.parser')

            # Find all post containers - SMF uses td.windowbg or td.windowbg2
            # Each post has a div.post containing the message text
            post_divs = soup.find_all('div', class_='post')

            for post_div in post_divs:
                # Get the post text
                text = post_div.get_text(separator='\n')
                text = clean_text(text)
                if len(text) < 30:
                    continue

                # Find author - look for the poster info above this post
                # In SMF, the poster name is in a link near the post
                poster_td = post_div.find_previous('td', class_='poster_info')
                author = "unknown_btctalk"
                if poster_td:
                    author_link = poster_td.find('a')
                    if author_link:
                        author = author_link.get_text().strip()

                # Skip Satoshi
                if author.lower() == 'satoshi':
                    continue

                # Find date
                date = None
                date_div = post_div.find_previous('div', class_='smalltext')
                if date_div:
                    date_text = date_div.get_text().strip()
                    if len(date_text) < 50:
                        date = date_text

                posts.append({
                    "source": "bitcointalk",
                    "author": author,
                    "title": f"topic_{topic_id}",
                    "date": date,
                    "text": text,
                    "url": url,
                    "topic_id": topic_id
                })

            time.sleep(1.5)
        except Exception as e:
            print(f"  Error on topic {topic_id} page {page}: {e}", flush=True)
            time.sleep(2)
            continue

    return posts


def main():
    all_posts = []

    # 1. Cryptography mailing list - focused ranges around Bitcoin era
    print("=== Cryptography mailing list ===", flush=True)

    # Range around the Bitcoin discussion (2008-2009): msgs 9900-10300
    print("  Scraping msgs 9900-10300...", flush=True)
    posts1 = scrape_mail_archive_range(9900, 10300)
    all_posts.extend(posts1)
    print(f"  -> {len(posts1)} posts from 9900-10300", flush=True)

    # Slightly earlier (2008): msgs 9500-9900
    print("  Scraping msgs 9500-9900...", flush=True)
    posts2 = scrape_mail_archive_range(9500, 9900)
    all_posts.extend(posts2)
    print(f"  -> {len(posts2)} posts from 9500-9900", flush=True)

    # Later period (2009-2010): msgs 10300-10700
    print("  Scraping msgs 10300-10700...", flush=True)
    posts3 = scrape_mail_archive_range(10300, 10700)
    all_posts.extend(posts3)
    print(f"  -> {len(posts3)} posts from 10300-10700", flush=True)

    print(f"  Total mailing list: {len(all_posts)}", flush=True)

    # 2. BitcoinTalk threads
    print("\n=== BitcoinTalk threads ===", flush=True)
    topic_ids = [12, 15, 16, 20, 47, 68, 108, 116, 120, 170, 196, 223, 234, 267, 286,
                 382, 568, 823, 1314, 1735, 2162, 2500, 3000, 4000, 5000]
    btc_posts = []
    for tid in topic_ids:
        posts = scrape_bitcointalk_thread(tid, max_pages=3)
        btc_posts.extend(posts)
        print(f"  topic {tid}: {len(posts)} posts (total: {len(btc_posts)})", flush=True)
        time.sleep(1)

    all_posts.extend(btc_posts)
    print(f"  Total bitcointalk: {len(btc_posts)}", flush=True)

    # Save
    outpath = os.path.join(OUTPUT_DIR, "non_satoshi_raw.json")
    with open(outpath, 'w', encoding='utf-8') as f:
        json.dump(all_posts, f, indent=2, ensure_ascii=False)

    print(f"\n=== DONE ===", flush=True)
    print(f"Total non-Satoshi items: {len(all_posts)}", flush=True)
    print(f"Saved to: {outpath}", flush=True)

    # Breakdown
    sources = {}
    for p in all_posts:
        s = p['source']
        sources[s] = sources.get(s, 0) + 1
    for s, c in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"  {s}: {c}", flush=True)

    authors = {}
    for p in all_posts:
        a = p['author']
        authors[a] = authors.get(a, 0) + 1
    print("\nTop 15 authors:", flush=True)
    for a, c in sorted(authors.items(), key=lambda x: -x[1])[:15]:
        print(f"  {a}: {c}", flush=True)

    # Text length stats
    if all_posts:
        lengths = [len(p['text']) for p in all_posts]
        print(f"\nText length: min={min(lengths)}, max={max(lengths)}, mean={sum(lengths)/len(lengths):.0f}", flush=True)


if __name__ == "__main__":
    main()
