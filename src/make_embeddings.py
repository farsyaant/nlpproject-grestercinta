"""Buat cache embedding per artikel -> data/export/embeddings_<tag>.npy (+ .json metadata).

Jalankan sekali, lalu semua orang tinggal np.load. File ini berdiri sendiri (tidak butuh modul lain).

    pip install sentence-transformers
    python make_embeddings.py --model sentence-transformers/all-MiniLM-L6-v2 --tag minilm

Model di atas hanya default contoh. Ganti --model dengan model yang dipakai tim.
Baris ke-i di file .npy = artikel dengan article_id == i.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import hashlib

import pandas as pd


def read_articles(export_dir):
    stem = Path(export_dir) / "articles"
    df = pd.read_parquet(stem.with_suffix(".parquet")) if stem.with_suffix(".parquet").exists() else pd.read_csv(stem.with_suffix(".csv"))
    return df.sort_values("article_id").reset_index(drop=True)


def articles_fingerprint(articles):  # harus sama dengan fungsi di notebook
    return hashlib.sha1("\n".join(articles["url"].astype(str)).encode("utf-8")).hexdigest()

RECIPES = {
    "title_text": lambda a: (a["title"].str.strip() + ". " + a["full_text"].str.strip()),
    "title": lambda a: a["title"].str.strip(),
    "full_text": lambda a: a["full_text"].str.strip(),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-dir", default="data/export")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--tag", default="minilm", help="nama pendek untuk nama file")
    ap.add_argument("--recipe", default="title_text", choices=list(RECIPES))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-seq-length", type=int, default=None, help="potong teks (token); default mengikuti model")
    ap.add_argument("--no-normalize", action="store_true")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer  # import di sini supaya modul lain tidak wajib install

    export_dir = Path(args.export_dir)
    articles = read_articles(export_dir)
    texts = RECIPES[args.recipe](articles).tolist()

    model = SentenceTransformer(args.model)
    if args.max_seq_length:
        model.max_seq_length = args.max_seq_length

    X = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=not args.no_normalize,
        convert_to_numpy=True,
    ).astype(np.float32)

    np.save(export_dir / f"embeddings_{args.tag}.npy", X)
    meta = {
        "model": args.model,
        "recipe": args.recipe,
        "max_seq_length": int(model.max_seq_length),
        "normalized": not args.no_normalize,
        "n": int(X.shape[0]),
        "dim": int(X.shape[1]),
        "articles_fingerprint": articles_fingerprint(articles),
    }
    (export_dir / f"embeddings_{args.tag}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Tersimpan: embeddings_{args.tag}.npy {X.shape}")


if __name__ == "__main__":
    main()
