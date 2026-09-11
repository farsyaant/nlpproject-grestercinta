# Pipeline Berita Geopolitik dan Kurs BI

Pipeline ini membersihkan metadata berita, mengubah waktu publikasi GDELT ke
WIB, lalu meroll tanggal berita ke hari perdagangan BI berikutnya.

## Struktur data

- `data/raw/news_cnbc_gdelt.csv`: berita dengan kolom URL, tema, tone, dan waktu.
- `data/raw/bi_usd_rate.csv`: kurs BI dengan kolom `date` dan nilai kurs.
- `data/processed/`: keluaran pipeline.

## Menjalankan

```bash
pip install -r requirements.txt
python main.py
```

Nama kolom berita yang didukung meliputi `url`/`URL`, `themes`/`V2Themes`,
`tone_scores`/`V2Tone`, dan `publish_time`/`DATE`. Kalender perdagangan diambil
dari tanggal yang tersedia pada file kurs BI, termasuk hari libur yang tidak ada.