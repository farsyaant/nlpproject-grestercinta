# Tugas 3: urutan pengerjaan

**Mesin GPU** (sekali jalan, upload hasilnya ke `data/export/`):
```
python src/make_embeddings.py --model BAAI/bge-small-en-v1.5 --tag bge_small --max-seq-length 512
python src/make_embeddings.py --model BAAI/bge-base-en-v1.5  --tag bge_base  --max-seq-length 512
python src/make_finbert.py --max-seq-length 512
```
Hasil: `embeddings_{bge_small,bge_base}.{npy,json}`, `embeddings_*_anchors.npy`, `finbert_probs.{npy,json}`.

**Siapa pun** (CPU): jalankan notebook berurutan `01_eda` → `02_features` → `03_tuning` → `04_evaluation`.
Pengaturan (fold, gap, trial, PCA k) ada di `config.json`. `03_tuning` memakai cache `outputs/tuned.joblib` (aman diulang);
`04_evaluation` satu-satunya yang membuka Test.
