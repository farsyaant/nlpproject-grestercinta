# Data Dictionary: bundle export `data/export/`

Dibuat oleh notebook `pipeline` (bagian 5b). Semua tanggal dan jam memakai **WIB (Asia/Jakarta)**.
Format tabel: parquet (kalau `pyarrow` terpasang) atau CSV sebagai cadangan. Baca dengan `pd.read_parquet` / `pd.read_csv`.

## Daftar file

| File | Isi | Kunci |
|---|---|---|
| `articles.parquet` | Artikel bersih, tanpa fitur sentimen | `article_id` |
| `prices_daily.parquet` | Kurs harian, target, fitur harga, dan kolom `split` | `date` |
| `splits.json` | Batas tanggal train/val/test (terkunci) | |
| `sentiment_articles.parquet` | Skor VADER/LM per artikel (baseline Tugas 2) | `article_id` |
| `sentiment_daily.parquet` | Skor sentimen teragregasi per hari trading (baseline Tugas 2) | `date` |
| `embeddings_<tag>.npy` + `.json` | Cache embedding per artikel (dibuat `src/make_embeddings.py`, opsional) | baris ke-i = `article_id` i |

## `articles`

| Kolom | Tipe | Arti |
|---|---|---|
| `article_id` | int | ID urut 0..n-1, diurutkan menurut `published_wib` lalu `url`. Baris ke-i di file embedding = artikel ini. |
| `url` | str | URL artikel (unik). |
| `section` | str | Section CNBC. Bisa kosong. |
| `keyword_matched` | str | Keyword pencarian yang menemukan artikel. Bisa kosong. |
| `title` | str | Judul asli. |
| `preview_summary` | str | Ringkasan dari halaman daftar. |
| `full_text` | str | Isi artikel (minimal 350 kata). |
| `text_clean` | str | Judul (2x) + ringkasan + isi, dibersihkan (tanpa HTML/URL/emoji/tanda baca, huruf kecil). Dipakai untuk skor LM dan bag-of-words. Untuk model transformer sebaiknya pakai `title`/`full_text` asli. |
| `published_wib` | datetime | Waktu terbit dalam WIB. |
| `date` | datetime | Tanggal terbit (WIB), tanpa jam. |
| `jam` | str | Jam terbit WIB, format 24 jam `HH:MM`. |
| `aligned_trading_date` | datetime | Hari trading tempat artikel dipasangkan dengan kurs. Aturan: terbit **setelah 15:00 WIB** dihitung ke hari berikutnya; Sabtu/Minggu digeser ke Senin; lalu digeser ke hari trading terdekat yang ada di data kurs. Selalu ada di `prices_daily`/`sentiment_daily` kalau hari itu bukan hari yang dibuang (lihat catatan). |

## `prices_daily`

Satu baris per hari trading. Hanya baris yang masuk salah satu split.

| Kolom | Arti |
|---|---|
| `date` | Hari trading t. |
| `split` | `train`, `val`, atau `test` (terkunci, jangan dihitung ulang). |
| `kurs_jual`, `kurs_beli` | Kurs transaksi BI (IDR per USD). |
| `kurs_mid` | (`kurs_jual` + `kurs_beli`) / 2. |
| `ret` | Log return hari ini: `ln(kurs_mid[t]) - ln(kurs_mid[t-1])`. **Bukan fitur**, hanya untuk baseline naif "return kemarin". |
| `target` | Log return hari trading berikutnya: `ret[t+1]`. Ini yang diprediksi. |
| `kurs_lag1` ... `kurs_lag5` | `kurs_mid` pada t-1 ... t-5 (level kurs, bukan return). |
| `kurs_roll_mean5` | Rata-rata `kurs_mid` t-1..t-5 (rolling 5 hari, di-`shift(1)` sehingga kurs hari t tidak ikut). |
| `kurs_roll_std5` | Std rolling 5 hari dengan `shift(1)` yang sama. |

Fitur harga = 7 kolom (`kurs_lag1`..`kurs_lag5`, `kurs_roll_mean5`, `kurs_roll_std5`).

## `splits.json`

Split kronologis 70/15/15 dari dataset yang sudah dibersihkan. Baris terakhir train dan val **dibuang**, karena target-nya (t+1) adalah hari pertama split berikutnya. Jadi tidak ada target yang bocor lintas split, dan `split` + tanggal di file ini adalah satu-satunya sumber batas.

## `sentiment_articles` / `sentiment_daily` (baseline Tugas 2)

Disimpan terpisah supaya eksperimen NLP baru tidak otomatis bercampur dengan fitur Tugas 2.

- `vader_title`: skor compound VADER dari judul asli (-1..1).
- `lm_title_*` / `lm_text_*`: proporsi kata Loughran-McDonald (`negative`, `positive`, `uncertainty`, `litigious`) dibanding total kata, dari judul / isi artikel. `*_net` = `positive` - `negative`.
- `sentiment_daily` meng-agregasi per `aligned_trading_date`: `news_count`, `vader_title_{mean,min,max,std,pos_ratio,neg_ratio}`, dan rata-rata tiap skor LM. **Hari tanpa berita: `news_count` = 0 dan semua skor diisi 0.0** (dianggap netral), jadi 0 di hari itu bukan berarti skor benar-benar nol.

## Evaluasi

Pakai `from eval_utils import evaluate` (file `src/eval_utils.py`) untuk semua metrik: MAE, MSE, RMSE, DirAcc, PredStd. `naive_baselines()` memberi pembanding "Prediksi 0" dan "Return kemarin". DirAcc melewati hari dengan target tepat 0.

## Catatan

- Fitur harga adalah **level** kurs, bukan return. Kurs naik dari sekitar 14 ribu (2021) ke sekitar 17,8 ribu (2026), jadi level di val/test kemungkinan keluar dari rentang train; model berbasis pohon tidak bisa ekstrapolasi dan scaler di-fit pada rentang yang berbeda. Kalau mau, tambahkan fitur berbasis return di eksperimen masing-masing, tapi pakai split yang sama.
- Sebagian judul punya kata yang menempel (misalnya `risinggeopoliticalrisks`) akibat scraping. Ini sudah ada di `title` mentah dan ikut ke `text_clean`.
- Embedding hanya valid untuk versi `articles` yang sidik jarinya sama. Cek dengan membandingkan `articles_fingerprint` di file `embeddings_<tag>.json` dengan yang dicetak notebook saat export.
