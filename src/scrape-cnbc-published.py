"""
scrape_cnbc_published.py

Scraping ulang tanggal PUBLISH (tanggal, jam, timezone) dari setiap URL artikel
CNBC yang ada di kolom `url` pada file CSV, lalu menyimpannya sebagai kolom baru:

    - published_raw       -> teks mentah yang ditemukan di halaman,
                              contoh: "Published Tue, Sep 8 20263:28 AM EDT"
    - published_date      -> tanggal, format YYYY-MM-DD (contoh: 2026-09-08)
    - published_time      -> jam, format HH:MM AM/PM (contoh: 03:28 AM)
    - published_timezone  -> timezone singkatan (contoh: EDT, EST, GMT, dst)
    - published_iso        -> datetime ISO lengkap dengan offset UTC jika tersedia
                              dari JSON-LD (contoh: 2026-09-08T03:28:00-0400)
    - scrape_status       -> "ok" / "not_found" / "error: <pesan>"

CARA PAKAI
----------
1. Install dependency (sekali saja):
       pip install requests beautifulsoup4 pandas tqdm lxml

2. Jalankan:
       python scrape-cnbc-published.py --input data/raw/2-cnbc-geopolitics-5years.csv --output cnbc-with-published.csv

3. Script ini BISA DILANJUTKAN (resume) kalau berhenti di tengah jalan:
   jalankan ulang command yang sama dengan --output yang sama, baris yang sudah
   berhasil di-scrape (status "ok") tidak akan di-scrape ulang.

CATATAN PENTING
----------------
- File kamu punya ~16.700 baris. Scraping semuanya akan makan waktu LAMA
  (bisa 1-3+ jam tergantung koneksi & rate limit dari CNBC) dan berisiko
  di-block sementara oleh CNBC kalau terlalu cepat. Karena itu script ini:
    * pakai multithreading terbatas (default 8 worker, bisa diubah --workers)
    * kasih jeda kecil + retry otomatis kalau kena error / rate-limit (HTTP 429/503)
    * nulis progress ke file output SECARA BERKALA (checkpoint), bukan nunggu
      sampai semua selesai baru disimpan -> aman kalau script terhenti/di-Ctrl+C
- Kalau mau coba dulu dengan jumlah baris kecil, pakai --limit, contoh:
       python scrape_cnbc_published.py --input in.csv --output out.csv --limit 50
"""

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# ----------------------------------------------------------------------------
# Konfigurasi
# ----------------------------------------------------------------------------

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

REQUEST_TIMEOUT = 15
MAX_RETRIES = 3
RETRY_BACKOFF_SEC = 3

# Pola teks "Published Tue, Sep 8 20263:28 AM EDT" (kadang tanpa spasi antara
# tahun & jam karena elemen HTML digabung waktu di-scrape / di-render).
PUBLISHED_TEXT_RE = re.compile(
    r"Published\s+"
    r"(?P<dow>[A-Za-z]{3}),\s*"
    r"(?P<month>[A-Za-z]{3})\s+"
    r"(?P<day>\d{1,2})\s+"
    r"(?P<year>\d{4})\s*"
    r"(?P<time>\d{1,2}:\d{2}\s*[AP]M)\s*"
    r"(?P<tz>[A-Za-z]{2,5})",
    re.IGNORECASE,
)

MONTH_MAP = {
    m.lower(): i
    for i, m in enumerate(
        [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ],
        start=1,
    )
}


# ----------------------------------------------------------------------------
# Ekstraksi tanggal publish dari satu halaman artikel
# ----------------------------------------------------------------------------

def _from_jsonld(soup: BeautifulSoup):
    """
    Cara paling akurat: banyak halaman CNBC menaruh datePublished di dalam
    JSON-LD (<script type="application/ld+json">) dengan format ISO8601
    lengkap dengan offset timezone, misal: 2026-09-08T03:28:00-0400
    """
    for script in soup.find_all("script", {"type": "application/ld+json"}):
        raw = script.string or script.get_text()
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue

        candidates = data if isinstance(data, list) else [data]
        for item in candidates:
            if not isinstance(item, dict):
                continue
            date_val = item.get("datePublished")
            if date_val:
                return date_val
            # kadang @graph berisi list object bersarang
            graph = item.get("@graph")
            if isinstance(graph, list):
                for g in graph:
                    if isinstance(g, dict) and g.get("datePublished"):
                        return g["datePublished"]
    return None


def _from_visible_text(soup: BeautifulSoup, full_page_text: str):
    """
    Fallback: cari teks "Published ..." yang tampil di halaman, biasanya ada
    di elemen dengan class mengandung 'ArticleHeader' / 'Timestamp'.
    """
    # coba elemen spesifik dulu (lebih bersih)
    candidates = soup.select(
        "[class*='ArticleHeader-time'], [class*='ArticleHeader-timestamp'], "
        "[class*='Timestamp'], time"
    )
    for el in candidates:
        text = el.get_text(" ", strip=True)
        m = PUBLISHED_TEXT_RE.search(text)
        if m:
            return m.group(0)

    # fallback terakhir: cari di seluruh teks halaman
    m = PUBLISHED_TEXT_RE.search(full_page_text)
    if m:
        return m.group(0)
    return None


def parse_published_text(text: str):
    """
    Parse teks mentah "Published Tue, Sep 8 20263:28 AM EDT" (dengan atau
    tanpa spasi sebelum jam) menjadi (date, time, timezone).
    """
    m = PUBLISHED_TEXT_RE.search(text)
    if not m:
        return None, None, None

    month = MONTH_MAP.get(m.group("month").lower()[:3])
    day = int(m.group("day"))
    year = int(m.group("year"))
    time_str = re.sub(r"\s+", " ", m.group("time")).upper().strip()
    tz = m.group("tz").upper()

    date_str = f"{year:04d}-{month:02d}-{day:02d}" if month else None
    return date_str, time_str, tz


def parse_iso_datetime(iso_str: str):
    """
    Parse datetime ISO dari JSON-LD, contoh: 2026-09-08T03:28:00-0400
    Mengembalikan (date, time, timezone_offset).
    """
    try:
        # normalisasi "Z" -> "+00:00" agar fromisoformat bisa parse
        cleaned = iso_str.replace("Z", "+00:00")
        dt = datetime.fromisoformat(cleaned)
    except ValueError:
        return None, None, None

    date_str = dt.strftime("%Y-%m-%d")
    time_str = dt.strftime("%I:%M %p")
    tz_str = dt.strftime("%z")  # contoh: -0400
    return date_str, time_str, tz_str


def scrape_one_url(url: str):
    """
    Ambil satu halaman artikel dan ekstrak info publish.
    Return dict siap dimasukkan ke DataFrame.
    """
    result = {
        "published_raw": "",
        "published_date": "",
        "published_time": "",
        "published_timezone": "",
        "published_iso": "",
        "scrape_status": "",
    }

    if not isinstance(url, str) or not url.strip():
        result["scrape_status"] = "error: empty url"
        return result

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            if resp.status_code in (429, 503):
                # kena rate limit, tunggu lalu coba lagi
                time.sleep(RETRY_BACKOFF_SEC * attempt)
                continue
            resp.raise_for_status()

            soup = BeautifulSoup(resp.text, "lxml")

            # 1) coba JSON-LD dulu (paling akurat & selalu ada offset timezone)
            iso_val = _from_jsonld(soup)
            if iso_val:
                date_str, time_str, tz_str = parse_iso_datetime(iso_val)
                if date_str:
                    result.update(
                        published_raw=iso_val,
                        published_date=date_str,
                        published_time=time_str,
                        published_timezone=tz_str,
                        published_iso=iso_val,
                        scrape_status="ok",
                    )
                    return result

            # 2) fallback ke teks "Published ..." yang tampil di halaman
            page_text = soup.get_text(" ", strip=True)
            raw_text = _from_visible_text(soup, page_text)
            if raw_text:
                date_str, time_str, tz_str = parse_published_text(raw_text)
                result.update(
                    published_raw=raw_text,
                    published_date=date_str or "",
                    published_time=time_str or "",
                    published_timezone=tz_str or "",
                    published_iso="",
                    scrape_status="ok" if date_str else "not_found",
                )
                return result

            result["scrape_status"] = "not_found"
            return result

        except requests.RequestException as e:
            last_err = e
            time.sleep(RETRY_BACKOFF_SEC * attempt)
            continue

    result["scrape_status"] = f"error: {last_err}"
    return result


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path CSV input")
    parser.add_argument("--output", required=True, help="Path CSV output (juga dipakai untuk resume)")
    parser.add_argument("--url-column", default="url", help="Nama kolom URL (default: url)")
    parser.add_argument("--workers", type=int, default=8, help="Jumlah thread paralel (default: 8)")
    parser.add_argument("--limit", type=int, default=None, help="Batasi jumlah baris untuk uji coba")
    parser.add_argument("--checkpoint-every", type=int, default=100, help="Simpan progress setiap N baris")
    args = parser.parse_args()

    print(f"Membaca input: {args.input}")
    df_in = pd.read_csv(args.input)

    if args.limit:
        df_in = df_in.head(args.limit).copy()

    result_cols = [
        "published_raw", "published_date", "published_time",
        "published_timezone", "published_iso", "scrape_status",
    ]

    # Resume: kalau file output sudah ada, pakai itu sebagai basis dan
    # lanjutkan baris yang belum "ok".
    try:
        df_out = pd.read_csv(args.output)
        if len(df_out) == len(df_in) and "scrape_status" in df_out.columns:
            print(f"Ditemukan file output sebelumnya: {args.output} -> melanjutkan (resume)")
            df = df_out
        else:
            df = df_in.copy()
            for c in result_cols:
                df[c] = ""
    except FileNotFoundError:
        df = df_in.copy()
        for c in result_cols:
            df[c] = ""

    todo_idx = df.index[df["scrape_status"] != "ok"].tolist()
    print(f"Total baris: {len(df)} | Belum berhasil di-scrape: {len(todo_idx)}")

    if not todo_idx:
        print("Semua baris sudah berhasil di-scrape sebelumnya. Tidak ada yang perlu dilakukan.")
        return

    def save_progress():
        df.to_csv(args.output, index=False)

    processed_since_checkpoint = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_idx = {
            executor.submit(scrape_one_url, df.loc[i, args.url_column]): i
            for i in todo_idx
        }

        with tqdm(total=len(future_to_idx), desc="Scraping") as pbar:
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    res = future.result()
                except Exception as e:  # noqa: BLE001 - simpan error apapun, jangan crash
                    res = {c: "" for c in result_cols}
                    res["scrape_status"] = f"error: {e}"

                for c in result_cols:
                    df.at[idx, c] = res.get(c, "")

                processed_since_checkpoint += 1
                pbar.update(1)

                if processed_since_checkpoint >= args.checkpoint_every:
                    save_progress()
                    processed_since_checkpoint = 0

    save_progress()

    ok = (df["scrape_status"] == "ok").sum()
    print(f"Selesai. Berhasil: {ok}/{len(df)}. Hasil disimpan ke: {args.output}")
    fail_mask = df["scrape_status"] != "ok"
    if fail_mask.any():
        print(f"Baris gagal/tidak ditemukan: {fail_mask.sum()} "
              f"(jalankan ulang command yang sama untuk retry otomatis)")


if __name__ == "__main__":
    main()
