import time
import csv
import re
from datetime import datetime, timedelta

import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from webdriver_manager.chrome import ChromeDriverManager

KEYWORDS = [
    # 1. Istilah Inti Geopolitik
    "geopolitics",
    "geopolitical risk",
    "geopolitical tensions",
    "geopolitical fragmentation",
    # 2. Konflik & Keamanan Militer
    "armed conflict",
    "global conflict",
    "international conflict2qwQQqqqwwwwwss",
    "military tensions",
    "national security",
    # 3. Geopolitik Ekonomi
    "sanctions",
    "trade war",
    "tariffs",
    "embargo",
    "export controls",
    "protectionism",
    "supply chain disruption",
    # 4. Diplomasi & Kebijakan Luar Negeri
    "diplomacy",
    "foreign policy",
    # 5. Organisasi & Blok Internasional
    "NATO",
    "BRICS",
    "OPEC",
    "G7",
]

EXCLUDED_SECTIONS = {
    "CNBC en Español",
}

BOILERPLATE_PATTERNS = [
    r"SUBSCRIBE HERE.*?(?=[A-Z][a-z])",
    r"Subscribe here to receive.*?inbox\.",
    r"Sign up (for|here).*?(inbox|newsletter)\.?",
    r"This is CNBC'?s .*? newsletter\.",
    r"Get .*? in your inbox first\.",
]
BOILERPLATE_REGEX = re.compile("|".join(BOILERPLATE_PATTERNS), re.IGNORECASE)

MAX_SCROLL_NO_NEW = 3
MAX_SCROLL_TOTAL = 60
SCROLL_DELAY = 1.5
KEYWORD_DELAY = 3
ARTICLE_FETCH_DELAY = 1.0

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; AcademicResearchBot/1.0; "
        "NLP-Geopolitik-USD-UGM-Project; contact: research-project)"
    )
}

OUTPUT_FILE = "2-cnbc-geoplotical-5years.csv"

CUTOFF_DATE = datetime.now() - timedelta(days=5 * 365.25)


def clean_boilerplate(text):
    if not text:
        return text
    cleaned = BOILERPLATE_REGEX.sub("", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

def looks_non_english(title, summary):
    text = f"{title} {summary}"
    accented = re.findall(r"[áéíóúñü¿¡]", text, flags=re.IGNORECASE)
    return len(accented) >= 2

def parse_cnbc_date(date_raw):
    """Menerjemahkan teks tanggal CNBC menjadi objek datetime."""
    if not date_raw:
        return None
    
    date_raw = date_raw.strip()
    
    if "ago" in date_raw.lower():
        return datetime.now()
        
    clean_date = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', date_raw)
    
    formats = [
        "%A, %b %d %Y",  # "Tuesday, Sep 8 2026"
        "%a, %b %d %Y",  # "Tue, Sep 8 2026"
        "%b %d %Y",      # "Sep 8 2026"
        "%B %d %Y",      # "September 8 2026"
        "%m/%d/%Y",      # "09/08/2026"
        "%Y-%m-%dT%H:%M:%S%z"
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(clean_date, fmt)
        except ValueError:
            continue
            
    return None

def setup_driver():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument(
        "user-agent=Mozilla/5.0 (compatible; AcademicResearchBot/1.0; "
        "NLP-Geopolitik-USD-UGM-Project)"
    )
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    return driver


def scrape_keyword(driver, keyword):
    url = f"https://www.cnbc.com/search/?query={keyword.replace(' ', '%20')}&qsearchterm={keyword.replace(' ', '%20')}"
    print(f"\n[+] Scraping keyword: '{keyword}'")

    driver.get(url)

    try:
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.ID, "searchcontainer"))
        )
    except Exception:
        print("    [!] Timeout menunggu search container, skip keyword ini.")
        return []

    time.sleep(2)

    seen_count = 0
    no_new_count = 0
    scroll_num = 0

    while no_new_count < MAX_SCROLL_NO_NEW and scroll_num < MAX_SCROLL_TOTAL:
        soup = BeautifulSoup(driver.page_source, "html.parser")
        results = soup.select(".SearchResult-searchResult")
        current_count = len(results)

        if current_count > seen_count:
            no_new_count = 0
            seen_count = current_count
        else:
            no_new_count += 1

        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(SCROLL_DELAY)
        scroll_num += 1

    print(f"    Total hasil mentah di halaman: {seen_count}")

    soup = BeautifulSoup(driver.page_source, "html.parser")
    result_blocks = soup.select(".SearchResult-searchResult")

    articles = []
    skipped_lang = 0
    skipped_section = 0
    skipped_old = 0

    for block in result_blocks:
        title_tag = block.select_one(".SearchResult-searchResultTitle a")
        preview_tag = block.select_one(".SearchResult-searchResultPreview")
        date_tag = block.select_one(".SearchResult-publishedDate")
        section_tag = block.select_one(".SearchResult-searchResultEyebrow")

        title = title_tag.get_text(strip=True) if title_tag else None
        article_url = title_tag["href"] if title_tag and title_tag.has_attr("href") else None
        preview = preview_tag.get_text(strip=True) if preview_tag else None
        date_raw = date_tag.get_text(strip=True) if date_tag else None
        section = section_tag.get_text(strip=True) if section_tag else None

        if not title or not article_url:
            continue

        if section in EXCLUDED_SECTIONS:
            skipped_section += 1
            continue

        if looks_non_english(title, preview or ""):
            skipped_lang += 1
            continue
            
        parsed_date = parse_cnbc_date(date_raw)
        if parsed_date:
            if parsed_date < CUTOFF_DATE:
                skipped_old += 1
                continue
        else:
            pass

        title_clean = clean_boilerplate(title)
        preview_clean = clean_boilerplate(preview)

        articles.append({
            "keyword_matched": keyword,
            "title": title_clean,
            "url": article_url,
            "preview_summary": preview_clean,
            "date_raw": date_raw,
            "section": section,
        })

    print(f"    Valid: {len(articles)} | Skip(>5 Thn): {skipped_old} | Skip(non-EN): {skipped_section + skipped_lang}")
    return articles

def fetch_full_article(url):
    try:
        resp = requests.get(url, headers=REQUEST_HEADERS, timeout=10)
        if resp.status_code != 200:
            return None
        soup = BeautifulSoup(resp.text, "html.parser")

        body_div = soup.select_one('[class*="ArticleBody-articleBody"]')
        if not body_div:
            return None

        paragraphs = body_div.find_all("p")
        full_text = " ".join(p.get_text(strip=True) for p in paragraphs)
        full_text = clean_boilerplate(full_text)
        return full_text if full_text else None

    except requests.RequestException:
        return None

def main():
    driver = setup_driver()
    all_articles = []

    try:
        for i, keyword in enumerate(KEYWORDS, 1):
            print(f"\n===== [{i}/{len(KEYWORDS)}] =====")
            try:
                articles = scrape_keyword(driver, keyword)
                all_articles.extend(articles)
            except Exception as e:
                print(f"    [!] Error scraping '{keyword}': {e}")

            time.sleep(KEYWORD_DELAY)
    finally:
        driver.quit()

    print(f"\n[SUMMARY] Total artikel (setelah filter bahasa & 5 thn): {len(all_articles)}")

    seen_urls = set()
    deduped = []
    for art in all_articles:
        if art["url"] not in seen_urls:
            seen_urls.add(art["url"])
            deduped.append(art)
        else:
            for d in deduped:
                if d["url"] == art["url"]:
                    if art["keyword_matched"] not in d["keyword_matched"]:
                        d["keyword_matched"] += f"; {art['keyword_matched']}"
                    break

    print(f"[SUMMARY] Total artikel unik setelah dedup: {len(deduped)}")

    print("\n[+] Mulai scrape isi artikel penuh (pakai requests)...")
    for idx, art in enumerate(deduped, 1):
        if idx % 50 == 0:
            print(f"    Progress: {idx}/{len(deduped)}")
        full_text = fetch_full_article(art["url"])
        art["full_text"] = full_text
        time.sleep(ARTICLE_FETCH_DELAY)

    n_full_text_ok = sum(1 for a in deduped if a.get("full_text"))
    print(f"[SUMMARY] Berhasil ambil full text: {n_full_text_ok}/{len(deduped)}")

    if deduped:
        fieldnames = [
            "keyword_matched", "title", "url", "preview_summary",
            "full_text", "date_raw", "section",
        ]
        with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(deduped)
        print(f"[SUMMARY] Data disimpan ke: {OUTPUT_FILE}")
    else:
        print("[SUMMARY] Tidak ada data untuk disimpan.")

if __name__ == "__main__":
    main()