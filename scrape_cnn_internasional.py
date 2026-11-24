#!/usr/bin/env python3
"""
Scraper berita kategori Internasional CNN Indonesia.

Instalasi:
    pip install requests beautifulsoup4

Contoh pemakaian:
    python scrape_cnn_internasional.py --pages 1            # tes kecil (~10 artikel)
    python scrape_cnn_internasional.py --pages 50           # 50 halaman indeks (~500 artikel)
    python scrape_cnn_internasional.py --all                # SEMUA halaman indeks (maks 1000)
    python scrape_cnn_internasional.py --all --no-content   # semua, tanpa isi berita (jauh lebih cepat)
    python scrape_cnn_internasional.py --all --format json

Fitur:
    - Data disimpan bertahap (checkpoint), aman kalau dihentikan dengan Ctrl+C.
    - Resume otomatis: kalau file output sudah ada, artikel yang sudah ada dilewati.
"""

import argparse
import csv
import json
import os
import re
import time
from urllib import robotparser

import requests
from bs4 import BeautifulSoup

BASE = "https://www.cnnindonesia.com"
HOME_URL = BASE + "/internasional"
INDEX_URL = BASE + "/internasional/indeks/6?page={page}"
MAX_PAGES = 1000  # batas halaman indeks yang ditampilkan situs
ARTICLE_RE = re.compile(r"^https://www\.cnnindonesia\.com/internasional/\d{14}-\d+-\d+/")
FIELDS = ["title", "date", "author", "description", "image", "url", "content"]
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "id-ID,id;q=0.9,en;q=0.8",
}

session = requests.Session()
session.headers.update(HEADERS)


def get_soup(url, retries=3, timeout=20):
    for attempt in range(1, retries + 1):
        try:
            r = session.get(url, timeout=timeout)
            if r.status_code == 200:
                return BeautifulSoup(r.text, "html.parser")
            print(f"  [!] HTTP {r.status_code} untuk {url}")
            if r.status_code in (404, 410):
                return None
        except requests.RequestException as e:
            print(f"  [!] Error ({attempt}/{retries}) {url}: {e}")
        time.sleep(2 * attempt)
    return None


def normalize(href):
    if href.startswith("/"):
        href = BASE + href
    return href.split("?")[0].split("#")[0]


def collect_links(soup):
    links = []
    for a in soup.find_all("a", href=True):
        href = normalize(a["href"])
        if ARTICLE_RE.match(href) and href not in links:
            links.append(href)
    return links


def parse_jsonld(soup):
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "")
        except (json.JSONDecodeError, TypeError):
            continue
        items = data if isinstance(data, list) else [data]
        for item in items:
            if isinstance(item, dict) and item.get("@type") in (
                "NewsArticle", "Article", "ReportageNewsArticle"
            ):
                return item
    return {}


def meta(soup, **attrs):
    tag = soup.find("meta", attrs=attrs)
    return tag["content"].strip() if tag and tag.get("content") else ""


def parse_article(url, with_content=True):
    soup = get_soup(url)
    if not soup:
        return None

    ld = parse_jsonld(soup)

    h1 = soup.find("h1")
    title = (h1.get_text(strip=True) if h1 else "") or ld.get("headline") or meta(soup, property="og:title")

    date = ld.get("datePublished") or meta(soup, property="article:published_time")
    if not date:
        d = soup.select_one("div.date, .date")
        date = d.get_text(" ", strip=True) if d else ""

    author = ld.get("author")
    if isinstance(author, dict):
        author = author.get("name", "")
    elif isinstance(author, list):
        author = ", ".join(a.get("name", "") for a in author if isinstance(a, dict))
    author = author or meta(soup, name="author")

    description = meta(soup, property="og:description") or meta(soup, name="description")
    image = meta(soup, property="og:image")

    content = ""
    if with_content:
        body = soup.select_one("div.detail-text") or soup.select_one("div.detail_text") \
            or soup.select_one("article")
        if body:
            for junk in body.select("script, style, table, .para_caption, .box_baca, "
                                    ".linksisip, .sisip, iframe, noscript, .widget"):
                junk.decompose()
            paragraphs = [p.get_text(" ", strip=True) for p in body.find_all("p")]
            if not paragraphs:
                paragraphs = [body.get_text(" ", strip=True)]
            content = "\n".join(p for p in paragraphs if p)

    return {
        "title": title,
        "date": date,
        "author": author or "",
        "description": description,
        "image": image,
        "url": url,
        "content": content,
    }


def save(rows, path, fmt, quiet=False):
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    if fmt == "json":
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
    else:
        with open(tmp, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
    os.replace(tmp, path)
    if not quiet:
        print(f"  >> checkpoint: {len(rows)} artikel tersimpan di {path}")


def load_existing(path, fmt):
    if not os.path.exists(path):
        return []
    try:
        if fmt == "json":
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        with open(path, newline="", encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))
    except Exception as e:
        print(f"[!] File lama tidak bisa dibaca ({e}), mulai dari awal.")
        return []


def check_robots():
    robots_url = BASE + "/robots.txt"
    try:
        r = session.get(robots_url, timeout=15)
    except requests.RequestException as e:
        print(f"[!] robots.txt tidak bisa diambil ({e}). Lanjut dengan hati-hati.")
        return True

    if r.status_code != 200:
        print(f"[!] robots.txt membalas HTTP {r.status_code}, jadi isinya tidak bisa diperiksa. "
              f"Buka {robots_url} di browser untuk cek manual. Lanjut dengan hati-hati.")
        return True

    rp = robotparser.RobotFileParser()
    rp.parse(r.text.splitlines())
    test_urls = [HOME_URL, BASE + "/internasional/indeks/6"]
    blocked = [u for u in test_urls if not rp.can_fetch("*", u)]
    if blocked:
        print("[!] robots.txt CNN Indonesia melarang akses ke:")
        for u in blocked:
            print("    -", u)
        print(f"    Cek sendiri isinya di {robots_url}. Script berhenti demi menghormati aturan situs.")
        return False
    print("robots.txt: path ini diizinkan.")
    return True


def main():
    ap = argparse.ArgumentParser(description="Scraper berita Internasional CNN Indonesia")
    ap.add_argument("--pages", type=int, default=5, help="jumlah halaman indeks (default 5, ~10 artikel/halaman)")
    ap.add_argument("--all", action="store_true", help=f"scrape semua halaman indeks (maks {MAX_PAGES})")
    ap.add_argument("--delay", type=float, default=1.5, help="jeda antar request, detik (default 1.5)")
    ap.add_argument("--no-content", action="store_true", help="jangan ambil isi artikel (lebih cepat)")
    ap.add_argument("--format", choices=["csv", "json"], default="csv")
    ap.add_argument("-o", "--output", default=None, help="nama file output (default cnn_internasional.<format>)")
    ap.add_argument("--outdir", default=os.path.join("data", "cnn"),
                    help="folder output (default data/cnn, dibuat otomatis)")
    ap.add_argument("--checkpoint", type=int, default=25, help="simpan setiap N artikel baru (default 25)")
    args = ap.parse_args()

    filename = args.output or f"cnn_internasional.{args.format}"
    output = os.path.join(args.outdir, filename)
    max_pages = MAX_PAGES if args.all else args.pages

    if not check_robots():
        return

    rows = load_existing(output, args.format)
    done = {r["url"] for r in rows if r.get("url")}
    if done:
        print(f"Resume: {len(done)} artikel sudah ada di {output}, akan dilewati.")

    new_count = 0
    empty_streak = 0
    try:
        for page in range(1, max_pages + 1):
            soup = get_soup(INDEX_URL.format(page=page))
            links = collect_links(soup) if soup else []
            if not links:
                print(f"Halaman indeks {page}: tidak ada artikel, berhenti.")
                break

            todo = [u for u in links if u not in done]
            print(f"Halaman indeks {page}/{max_pages}: {len(links)} link, {len(todo)} baru")

            # kalau 5 halaman berturut-turut tidak ada yang baru, anggap sudah mentok / sudah pernah diambil
            empty_streak = empty_streak + 1 if not todo else 0
            if empty_streak >= 5 and not args.all:
                print("5 halaman berturut-turut tanpa artikel baru, berhenti.")
                break

            for url in todo:
                data = parse_article(url, with_content=not args.no_content)
                time.sleep(args.delay)
                if not data:
                    continue
                rows.append(data)
                done.add(url)
                new_count += 1
                print(f"  [{len(rows)}] {data['title'][:80]}")
                if new_count % args.checkpoint == 0:
                    save(rows, output, args.format)
            time.sleep(args.delay)
    except KeyboardInterrupt:
        print("\nDihentikan manual, menyimpan data yang sudah terkumpul...")

    save(rows, output, args.format, quiet=True)
    print(f"\nSelesai: {len(rows)} artikel total ({new_count} baru) di {output}")


if __name__ == "__main__":
    main()
