"""
Fast non-Satoshi scraper using concurrent requests.
"""

import requests
from bs4 import BeautifulSoup
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
os.makedirs(OUTPUT_DIR, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (research project)"}
SESSION = requests.Session()
SESSION.headers.update(HEADERS)


def remove_quoted_text(text):
    lines = text.split('\n')
    out = []
    for line in lines:
        s = line.strip()
        if s.startswith('>'):
            continue
        if re.match(r'^On .+ wrote:$', s):
            continue
        out.append(line)
    return '\n'.join(out).strip()


def fetch_mail_archive(msg_num):
    """Fetch a single message from mail-archive.com."""
    url = f"https://www.mail-archive.com/cryptography@metzdowd.com/msg{msg_num:05d}.html"
    try:
        resp = SESSION.get(url, timeout=5)
        if resp.status_code != 200:
            return None

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
            return None

        # Title
        h1 = soup.find('h1')
        title = h1.get_text().strip() if h1 else None

        # Content
        msg_body = soup.find('div', class_='msgBody')
        if not msg_body:
            return None

        text = msg_body.get_text(separator='\n')
        text = remove_quoted_text(text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text).strip()

        if len(text) < 50:
            return None

        return {
            "source": "mailing_list_cryptography",
            "author": author,
            "title": title,
            "text": text,
            "msg_num": msg_num
        }
    except:
        return None


def fetch_bitcointalk_thread(topic_id, page=0):
    """Fetch posts from a bitcointalk thread page."""
    url = f"https://bitcointalk.org/index.php?topic={topic_id}.{page * 20}"
    posts = []
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return posts

        soup = BeautifulSoup(resp.text, 'html.parser')
        post_divs = soup.find_all('div', class_='post')

        for post_div in post_divs:
            text = post_div.get_text(separator='\n')
            text = re.sub(r'\n{3,}', '\n\n', text).strip()
            if len(text) < 30:
                continue

            author = "unknown_btctalk"
            poster_td = post_div.find_previous('td', class_='poster_info')
            if poster_td:
                a = poster_td.find('a')
                if a:
                    author = a.get_text().strip()

            if author.lower() == 'satoshi':
                continue

            posts.append({
                "source": "bitcointalk",
                "author": author,
                "title": f"topic_{topic_id}",
                "text": text,
                "topic_id": topic_id
            })
    except:
        pass
    return posts


def main():
    all_posts = []

    # 1. Mailing list - concurrent fetching
    print("=== Mailing list (concurrent) ===", flush=True)
    msg_nums = list(range(9500, 10700))
    print(f"  Fetching {len(msg_nums)} messages with 10 threads...", flush=True)

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_mail_archive, n): n for n in msg_nums}
        done = 0
        for future in as_completed(futures):
            result = future.result()
            if result:
                all_posts.append(result)
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(msg_nums)} done, {len(all_posts)} collected", flush=True)

    print(f"  -> {len(all_posts)} mailing list posts", flush=True)

    # 2. BitcoinTalk - sequential (they rate limit)
    print("\n=== BitcoinTalk threads ===", flush=True)
    topics = [12, 15, 16, 20, 47, 68, 108, 116, 120, 170, 196, 223, 234, 267, 286,
              382, 568, 823, 1314, 1735, 2162, 2500, 3000, 4000, 5000,
              6000, 7000, 8000, 9000, 10000]

    btc_count = 0
    for tid in topics:
        for page in range(3):
            posts = fetch_bitcointalk_thread(tid, page)
            all_posts.extend(posts)
            btc_count += len(posts)
            import time; time.sleep(1.0)
        if btc_count > 0 and tid % 5 == 0:
            print(f"  Through topic {tid}: {btc_count} btctalk posts", flush=True)

    print(f"  -> {btc_count} BitcoinTalk posts", flush=True)

    # Save
    outpath = os.path.join(OUTPUT_DIR, "non_satoshi_raw.json")
    with open(outpath, 'w', encoding='utf-8') as f:
        json.dump(all_posts, f, indent=2, ensure_ascii=False)

    print(f"\n=== DONE ===", flush=True)
    print(f"Total: {len(all_posts)}", flush=True)

    # Stats
    sources = {}
    for p in all_posts:
        s = p['source']
        sources[s] = sources.get(s, 0) + 1
    for s, c in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"  {s}: {c}", flush=True)

    authors = {}
    for p in all_posts:
        authors[p['author']] = authors.get(p['author'], 0) + 1
    print(f"\nTop 15 authors:", flush=True)
    for a, c in sorted(authors.items(), key=lambda x: -x[1])[:15]:
        print(f"  {a}: {c}", flush=True)

    lengths = [len(p['text']) for p in all_posts]
    if lengths:
        print(f"\nText length: min={min(lengths)}, max={max(lengths)}, mean={sum(lengths)/len(lengths):.0f}", flush=True)


if __name__ == "__main__":
    main()
