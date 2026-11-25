# Tugas 3: Model A, B, dan C

Folder ini berisi studi ablasi untuk target yang sama: log return kurs tengah USD/IDR
hari berikutnya (`target = ret.shift(-1)`). Ketiga notebook memakai baris dan split
yang identik, yaitu 836 hari latih, 178 hari validasi, dan 180 hari uji.

| Notebook | Isi | Waktu jalan (CPU) |
|---|---|---|
| `Model_A.ipynb` | 7 fitur kurs + LSTM PyTorch | sekitar 1 menit |
| `Model_B.ipynb` | kurs + VADER + Loughran-McDonald | sekitar 1 menit |
| `Model_C.ipynb` | kurs + embedding FinBERT dan DistilRoBERTa | sekitar 6 menit |

## Sumber data

Kurs diambil dari `data/raw/bi-usd-rate.csv` (Model A) dan
`data/export/prices_daily.parquet` (Model B dan C). Kolom `split` di parquet itu
sudah berisi pembagian kronologis yang dipakai bersama, sehingga tidak ada
perhitungan ulang split di B dan C.

Sentimen Model B berasal dari `data/export/sentiment_daily.parquet`. Model C
menghitung embedding sendiri dari `data/export/articles.parquet`, memakai
`aligned_trading_date` (hari bursa versi WIB) sebagai kunci agregasi harian.
Tidak ada fingerprint atau hash antar-notebook; kesamaan baris diverifikasi
dengan mencetak jumlah baris, tanggal, dan selisih target terhadap
`results/dataset_model_A.parquet`.

## Urutan menjalankan

1. `Model_A.ipynb`
2. `Model_B.ipynb`
3. `Model_C.ipynb`

Model B dan C membaca `results/dataset_model_A.parquet`, jadi A harus dijalankan
lebih dulu. Model C menyimpan embedding harian ke `results/embeddings_daily_*.parquet`
dan memakai ulang cache itu bila model serta jumlah artikelnya masih sama.

Kebutuhan paket: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `torch`,
`torch`, `transformers`, dan `matplotlib`. Tidak ada paket tambahan di luar itu.

## Keluaran

Semua keluaran ada di `results/`:

```
metrics_model_A.csv, metrics_model_B.csv, metrics_model_C.csv
predictions_val_model_*.csv, predictions_test_model_*.csv   (format long)
tuning_model_B.csv, tuning_model_C.csv
importance_model_B.csv, importance_model_C.csv
naive_baselines_model_A.csv
dataset_model_A.parquet
metadata_model_*.json
embeddings_daily_finbert.parquet, embeddings_daily_roberta.parquet
tsne_finbert.png
error_analysis_model_C.csv
```

`metrics_*` memuat MAE, MSE, RMSE, DirAcc, PredStd, dan kolom `Degenerate`.
Model dengan PredStd di bawah `1e-9` ditandai degenerate dan tidak dipakai
memilih model terbaik. Pada data ini Lasso, ElasticNet, dan SVR selalu masuk
kategori itu.

## Angka utama (Test)

| Varian | Model | RMSE | DirAcc |
|---|---|---|---|
| A | KNN | 0,0029806 | 0,5284 |
| B_lex | LinearRegression | 0,0029213 | 0,5568 |
| C_combined | RandomForest | 0,0028349 | 0,5909 |

Baseline naif `prediksi 0` menghasilkan RMSE 0,0029078, jadi keunggulan C_combined
atas baseline tanpa model hanya sekitar 2,5%. Angka lengkap dan pembahasannya ada
di laporan.

## Laporan

Sumber LaTeX dan PDF ada di `report/`:

```
report/laporan_tugas3.tex
report/laporan_tugas3.pdf
```

Kompilasi dari folder `report/`:

```
pdflatex laporan_tugas3.tex
pdflatex laporan_tugas3.tex
```

Dua kali jalan diperlukan agar referensi tabel dan gambar terisi. Laporan memakai
gambar `../results/tsne_finbert.png`, jadi folder `results/` harus ada saat kompilasi.
