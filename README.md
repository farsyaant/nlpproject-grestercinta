# Analisis Berita Geopolitik dan Kurs USD BI

Pipeline NLP untuk mengumpulkan berita geopolitik CNBC, membersihkan teks,
mengekstrak fitur sentimen dan risiko geopolitik, lalu menyelaraskannya dengan
data kurs USD Bank Indonesia.

## Tujuan

Proyek ini menyiapkan data berita dan kurs untuk analisis hubungan antara
pemberitaan geopolitik dan perubahan nilai tukar USD/IDR. Tahapan utamanya:

1. Mengumpulkan artikel CNBC yang relevan dengan geopolitik.
2. Membersihkan, memfilter, dan menormalisasi teks berita.
3. Menyesuaikan waktu publikasi ke zona waktu WIB dan tanggal perdagangan BI.
4. Menghitung fitur VADER, Loughran-McDonald, keyword density, serta TF-IDF +
	 SVD.
5. Mengagregasikan fitur berita ke level harian dan menggabungkannya dengan
	 data kurs BI.

## Struktur proyek

```text
.
|-- data/
|   |-- raw/                         # Data mentah berita dan kurs
|   `-- processed/                   # Data hasil preprocessing
|-- src/
|   |-- cnbc-geopolitics-scraper.py  # Scraper arsip berita CNBC
|   |-- scrape-cnbc-published.py     # Scraper tanggal publikasi artikel
|   |-- preprocessing_pipeline.ipynb
|   |-- preprocessing_pipeline_published.ipynb
|   |-- final.ipynb                  # Pipeline final dan ekstraksi fitur NLP
|   `-- testing.ipynb                # Eksperimen dan pengujian
|-- requirements.txt
`-- README.md
```

## Data utama

- `data/raw/2-cnbc-geoplotical-5years.csv`: hasil pengumpulan berita geopolitik.
- `data/raw/cnbc_with_published.csv`: berita CNBC dengan tanggal publikasi.
- `data/raw/bi-usd-rate.csv`: data kurs USD BI.
- `data/processed/`: lokasi data keluaran setelah preprocessing.

## Instalasi

Gunakan Python 3.10 atau yang lebih baru, kemudian jalankan:

```bash
pip install -r requirements.txt
```

Notebook `final.ipynb` juga menggunakan NLTK, Sastrawi, langdetect,
vaderSentiment, pysentiment2, scikit-learn, holidays, dan Jupyter. Jika modul
tersebut belum tersedia di environment aktif, instal dengan:

```bash
pip install nltk Sastrawi langdetect vaderSentiment pysentiment2 scikit-learn holidays jupyter
```

Resource NLTK yang dibutuhkan akan diunduh oleh notebook saat pertama kali
dijalankan. Pastikan koneksi internet tersedia pada eksekusi pertama.

## Menjalankan pipeline

1. Buka folder proyek di VS Code.
2. Pilih Python interpreter atau kernel Jupyter yang sesuai.
3. Jalankan notebook sesuai kebutuhan:
	 - `preprocessing_pipeline.ipynb` untuk preprocessing awal.
	 - `preprocessing_pipeline_published.ipynb` untuk preprocessing dengan tanggal
		 publikasi.
	 - `final.ipynb` untuk pipeline final, ekstraksi fitur NLP, agregasi harian,
		 dan sanity check.

Script scraper dapat dijalankan dari root proyek dengan:

```bash
python src/cnbc-geopolitics-scraper.py
python src/scrape-cnbc-published.py
```

Scraper pertama membutuhkan browser Chrome dan ChromeDriver. Scraper kedua
menggunakan argumen command line; lihat bantuan lengkapnya dengan:

```bash
python src/scrape-cnbc-published.py --help
```

## Catatan

- Path data pada notebook menggunakan lokasi relatif terhadap folder `src`.
	Jalankan notebook dengan workspace proyek sebagai root agar path tetap sesuai.
- Waktu berita dikonversi ke WIB (`Asia/Jakarta`).
- Berita yang melewati cutoff perdagangan akan dialihkan ke tanggal perdagangan
	berikutnya, termasuk penyesuaian akhir pekan.
- Jangan menyimpan kredensial atau data sensitif ke repository.