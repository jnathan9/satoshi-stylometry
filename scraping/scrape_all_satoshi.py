"""
Scrape ALL known Satoshi Nakamoto writings from every public source.
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import os
import re
import sys

OUTPUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw", "v5_all_satoshi.json")
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (research project - satoshi stylometry)"}


def scrape_malmi_emails():
    """Scrape Satoshi's 144 emails to Martti Malmi from mmalmi.github.io/satoshi/"""
    print("=== Scraping Martti Malmi emails ===", flush=True)

    url = "https://mmalmi.github.io/satoshi/"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    soup = BeautifulSoup(resp.text, 'html.parser')

    messages = soup.find_all('div', class_='message')
    emails = []

    for msg in messages:
        classes = msg.get('class', [])
        if 'satoshi' not in classes:
            continue

        # Get header info
        header = msg.find('div', class_='header')
        date = subject = None
        if header:
            ht = header.get_text(separator='\n')
            dm = re.search(r'Date:\s*(.*)', ht)
            if dm: date = dm.group(1).strip()
            sm = re.search(r'Subject:\s*(.*)', ht)
            if sm: subject = sm.group(1).strip()

        # Get body text
        body = msg.find('div', class_='body')
        if body:
            text = body.get_text(separator='\n')
        else:
            pres = msg.find_all('pre')
            text = '\n'.join(p.get_text() for p in pres) if pres else msg.get_text(separator='\n')

        # Remove quoted lines
        lines = text.split('\n')
        clean = [l for l in lines if not l.strip().startswith('>') and not re.match(r'^On .+ wrote:', l.strip())]
        text = '\n'.join(clean).strip()

        if len(text.split()) < 5:
            continue

        emails.append({
            "source": "email_malmi",
            "date": date,
            "title": subject,
            "text": text,
            "author": "satoshi"
        })

    print(f"  -> {len(emails)} emails, {sum(len(e['text'].split()) for e in emails):,} words", flush=True)
    return emails


def scrape_whitepaper():
    """Scrape the Bitcoin whitepaper prose."""
    print("=== Scraping Bitcoin whitepaper ===", flush=True)

    url = "https://nakamotoinstitute.org/library/bitcoin/"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    soup = BeautifulSoup(resp.text, 'html.parser')

    for tag in soup.find_all(['nav', 'header', 'footer', 'script', 'style']):
        tag.decompose()

    main = soup.find('main') or soup.find('article') or soup
    text = main.get_text(separator='\n')

    # Clean: remove math-heavy lines, keep prose
    lines = text.split('\n')
    clean = []
    for line in lines:
        s = line.strip()
        if re.match(r'^[pqz\s=\-\+\*\(\)\{\}\<\>\.]+$', s):
            continue
        if s.startswith('q =') or s.startswith('p =') or s.startswith('P ='):
            continue
        clean.append(line)
    text = '\n'.join(clean).strip()

    words = len(text.split())
    print(f"  -> {words} words", flush=True)

    return [{
        "source": "whitepaper",
        "date": "2008-10-31",
        "title": "Bitcoin: A Peer-to-Peer Electronic Cash System",
        "text": text,
        "author": "satoshi"
    }]


def scrape_gwern_wei_dai():
    """Scrape Satoshi's emails to Wei Dai."""
    print("=== Scraping Wei Dai emails ===", flush=True)

    url = "https://gwern.net/doc/bitcoin/2008-nakamoto"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        soup = BeautifulSoup(resp.text, 'html.parser')
        text = soup.get_text(separator='\n')

        emails = []
        # Look for Satoshi's text blocks
        blocks = re.split(r'(?=From:\s*(?:satoshi|Satoshi))', text)
        for block in blocks:
            if 'satoshi' not in block[:200].lower():
                continue
            lines = block.split('\n')
            body_start = 0
            for i, line in enumerate(lines):
                if any(line.strip().startswith(h) for h in ['From:', 'To:', 'Subject:', 'Date:']):
                    body_start = i + 1
            body = '\n'.join(lines[body_start:])
            body = '\n'.join(l for l in body.split('\n') if not l.strip().startswith('>'))
            body = body.strip()
            if len(body.split()) >= 10:
                emails.append({
                    "source": "email_wei_dai",
                    "text": body,
                    "author": "satoshi"
                })

        print(f"  -> {len(emails)} emails", flush=True)
        return emails
    except Exception as e:
        print(f"  ERROR: {e}", flush=True)
        return []


def scrape_trammell_emails():
    """Scrape from bitcoin wiki."""
    print("=== Scraping Trammell emails ===", flush=True)
    try:
        url = "https://en.bitcoin.it/wiki/Source:Trammell/Nakamoto_emails"
        resp = requests.get(url, headers=HEADERS, timeout=30)
        soup = BeautifulSoup(resp.text, 'html.parser')
        text = soup.get_text(separator='\n')

        emails = []
        blocks = re.split(r'(?=From:\s*satoshi)', text, flags=re.IGNORECASE)
        for block in blocks:
            if 'satoshi' not in block[:200].lower():
                continue
            lines = block.split('\n')
            body_start = 0
            for i, line in enumerate(lines):
                if any(line.strip().startswith(h) for h in ['From:', 'To:', 'Subject:', 'Date:']):
                    body_start = i + 1
            body = '\n'.join(lines[body_start:])
            body = '\n'.join(l for l in body.split('\n') if not l.strip().startswith('>'))
            body = body.strip()
            if len(body.split()) >= 10:
                emails.append({
                    "source": "email_trammell",
                    "text": body,
                    "author": "satoshi"
                })

        print(f"  -> {len(emails)} emails", flush=True)
        return emails
    except Exception as e:
        print(f"  ERROR: {e}", flush=True)
        return []


def scrape_bitcoin_com_hal_finney():
    """Scrape Satoshi's emails to Hal Finney."""
    print("=== Scraping Hal Finney emails ===", flush=True)
    try:
        url = "https://www.bitcoin.com/satoshi-archive/emails/hal-finney/"
        resp = requests.get(url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(resp.text, 'html.parser')
        text = soup.get_text(separator='\n')

        emails = []
        blocks = re.split(r'(?=From:\s*(?:satoshi|Satoshi))', text)
        for block in blocks:
            if 'satoshi' not in block[:200].lower():
                continue
            lines = block.split('\n')
            body_start = 0
            for i, line in enumerate(lines):
                if any(line.strip().startswith(h) for h in ['From:', 'To:', 'Subject:', 'Date:']):
                    body_start = i + 1
            body = '\n'.join(lines[body_start:])
            body = '\n'.join(l for l in body.split('\n') if not l.strip().startswith('>'))
            body = body.strip()
            if len(body.split()) >= 10:
                emails.append({
                    "source": "email_hal_finney",
                    "text": body,
                    "author": "satoshi"
                })

        print(f"  -> {len(emails)} emails", flush=True)
        return emails
    except Exception as e:
        print(f"  ERROR: {e}", flush=True)
        return []


def main():
    all_items = []

    # 1. Existing Nakamoto Institute data (forum posts + public mailing list emails)
    existing_path = os.path.join(os.path.dirname(OUTPUT), "v4_satoshi.json")
    with open(existing_path, encoding='utf-8') as f:
        existing = json.load(f)
    print(f"Existing NI data: {len(existing)} items", flush=True)
    all_items.extend(existing)

    # 2. Whitepaper
    all_items.extend(scrape_whitepaper())

    # 3. Martti Malmi emails (THE BIG ONE)
    all_items.extend(scrape_malmi_emails())

    # 4. Wei Dai emails
    all_items.extend(scrape_gwern_wei_dai())

    # 5. Dustin Trammell emails
    all_items.extend(scrape_trammell_emails())

    # 6. Hal Finney emails
    all_items.extend(scrape_bitcoin_com_hal_finney())

    # Save
    with open(OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(all_items, f, indent=2, ensure_ascii=False)

    print(f"\n=== DONE ===", flush=True)
    from collections import Counter
    sources = Counter(d['source'] for d in all_items)
    total_words = 0
    for s, c in sorted(sources.items(), key=lambda x: -x[1]):
        words = sum(len(d['text'].split()) for d in all_items if d['source'] == s)
        total_words += words
        print(f"  {s}: {c} items, {words:,} words", flush=True)
    print(f"  TOTAL: {len(all_items)} items, {total_words:,} words", flush=True)


if __name__ == "__main__":
    main()
