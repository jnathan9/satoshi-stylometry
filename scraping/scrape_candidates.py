"""
Scrape real writings from Satoshi candidates for comparison scoring.
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import os
import re
import sys

OUTPUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "candidates")
os.makedirs(OUTPUT, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (research project - satoshi stylometry)"}


def scrape_bitcointalk_user(username, user_id, max_pages=10):
    """Scrape posts from a BitcoinTalk user."""
    posts = []
    for page in range(max_pages):
        start = page * 20
        url = f"https://bitcointalk.org/index.php?action=profile;u={user_id};sa=showPosts;start={start}"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code != 200:
                break
            soup = BeautifulSoup(resp.text, 'html.parser')
            post_divs = soup.find_all('div', class_='post')
            if not post_divs:
                break
            for pd in post_divs:
                text = pd.get_text(separator='\n').strip()
                # Remove quote blocks
                lines = text.split('\n')
                clean = [l for l in lines if not l.strip().startswith('Quote from:')]
                text = '\n'.join(clean).strip()
                if len(text.split()) >= 20:
                    posts.append(text)
            time.sleep(1.5)
        except Exception as e:
            print(f"  Error page {page}: {e}", flush=True)
            time.sleep(3)
    return posts


def scrape_szabo_blog():
    """Scrape Nick Szabo's Unenumerated blog."""
    posts = []
    # Get the blog archive page
    base = "http://unenumerated.blogspot.com/"
    try:
        # Try to get recent posts from the main page and archive pages
        for page_url in [
            base,
            base + "search?updated-max=2014-01-01T00:00:00-08:00&max-results=25",
            base + "search?updated-max=2012-01-01T00:00:00-08:00&max-results=25",
            base + "search?updated-max=2010-01-01T00:00:00-08:00&max-results=25",
            base + "search?updated-max=2008-01-01T00:00:00-08:00&max-results=25",
        ]:
            resp = requests.get(page_url, headers=HEADERS, timeout=15)
            if resp.status_code != 200:
                continue
            soup = BeautifulSoup(resp.text, 'html.parser')
            # Blog posts are in div.post-body
            for post_div in soup.find_all('div', class_='post-body'):
                text = post_div.get_text(separator='\n').strip()
                if len(text.split()) >= 30:
                    posts.append(text)
            time.sleep(1)
    except Exception as e:
        print(f"  Error: {e}", flush=True)
    return posts


def scrape_wright_medium():
    """Scrape Craig Wright's Medium articles."""
    # Medium is hard to scrape directly. Try the public profile page.
    posts = []
    try:
        url = "https://medium.com/@craig_10243"
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, 'html.parser')
            # Get article links
            links = soup.find_all('a', href=re.compile(r'medium.com/@craig_10243/'))
            article_urls = list(set(a['href'] for a in links if a.get('href')))[:20]
            for aurl in article_urls:
                try:
                    aresp = requests.get(aurl, headers=HEADERS, timeout=15)
                    if aresp.status_code != 200:
                        continue
                    asoup = BeautifulSoup(aresp.text, 'html.parser')
                    # Get article body
                    article = asoup.find('article')
                    if article:
                        text = article.get_text(separator='\n').strip()
                        if len(text.split()) >= 50:
                            posts.append(text[:3000])  # Cap at 3000 chars
                    time.sleep(1)
                except:
                    continue
    except Exception as e:
        print(f"  Error: {e}", flush=True)
    return posts


def scrape_lesswrong_user(username):
    """Scrape posts from a LessWrong user via their API."""
    posts = []
    try:
        # LessWrong has a GraphQL API
        url = f"https://www.lesswrong.com/users/{username}"
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, 'html.parser')
            # Get post content from the page
            for post_div in soup.find_all('div', class_=re.compile(r'post|comment')):
                text = post_div.get_text(separator='\n').strip()
                if len(text.split()) >= 30:
                    posts.append(text[:2000])
    except Exception as e:
        print(f"  Error: {e}", flush=True)
    return posts


def main():
    candidates = {}

    # 1. Hal Finney - BitcoinTalk user 2436
    print("=== Hal Finney (BitcoinTalk #2436) ===", flush=True)
    hal_posts = scrape_bitcointalk_user("hal_finney", 2436, max_pages=10)
    candidates['hal_finney'] = hal_posts
    print(f"  -> {len(hal_posts)} posts", flush=True)

    # 2. Gavin Andresen - BitcoinTalk user 224
    print("=== Gavin Andresen (BitcoinTalk #224) ===", flush=True)
    gavin_posts = scrape_bitcointalk_user("gavin_andresen", 224, max_pages=10)
    candidates['gavin_andresen'] = gavin_posts
    print(f"  -> {len(gavin_posts)} posts", flush=True)

    # 3. Wei Dai - BitcoinTalk user 1954
    print("=== Wei Dai (BitcoinTalk #1954) ===", flush=True)
    wei_posts = scrape_bitcointalk_user("wei_dai", 1954, max_pages=5)
    candidates['wei_dai'] = wei_posts
    print(f"  -> {len(wei_posts)} posts", flush=True)

    # 4. Nick Szabo - Unenumerated blog
    print("=== Nick Szabo (Unenumerated blog) ===", flush=True)
    szabo_posts = scrape_szabo_blog()
    candidates['nick_szabo'] = szabo_posts
    print(f"  -> {len(szabo_posts)} posts", flush=True)

    # 5. Craig Wright - Medium (may be limited)
    print("=== Craig Wright (Medium) ===", flush=True)
    wright_posts = scrape_wright_medium()
    candidates['craig_wright'] = wright_posts
    print(f"  -> {len(wright_posts)} posts", flush=True)

    # 6. Adam Back - try BitcoinTalk search
    print("=== Adam Back (BitcoinTalk) ===", flush=True)
    # Adam Back's BTCtalk user ID is reportedly 2442 or similar
    adam_posts = scrape_bitcointalk_user("adam_back", 2442, max_pages=5)
    if not adam_posts:
        # Try alternate ID
        adam_posts = scrape_bitcointalk_user("adam_back", 94, max_pages=5)
    candidates['adam_back'] = adam_posts
    print(f"  -> {len(adam_posts)} posts", flush=True)

    # Save all
    for name, posts in candidates.items():
        outpath = os.path.join(OUTPUT, f"{name}.json")
        with open(outpath, 'w', encoding='utf-8') as f:
            json.dump(posts, f, indent=2, ensure_ascii=False)

    print(f"\n=== SUMMARY ===", flush=True)
    for name, posts in candidates.items():
        total_words = sum(len(p.split()) for p in posts)
        print(f"  {name}: {len(posts)} texts, {total_words:,} words", flush=True)


if __name__ == "__main__":
    main()
