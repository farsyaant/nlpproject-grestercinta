"""Buat cache embedding per artikel -> data/export/embeddings_<tag>.npy (+ .json metadata).

    pip install sentence-transformers
    python src/make_embeddings.py --model BAAI/bge-small-en-v1.5 --tag bge_small --max-seq-length 512
    python src/make_embeddings.py --model BAAI/bge-base-en-v1.5  --tag bge_base  --max-seq-length 512

Baris ke-i di .npy = artikel dengan article_id == i (dicek: article_id harus 0..n-1).
Kalimat jangkar di src/anchors.txt ikut di-embed -> embeddings_<tag>_anchors.npy (untuk fitur F7).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
REQUIRED = ["article_id", "url", "title", "full_text"]
RECIPES = {  # tokenizer memotong dari ujung, jadi title_text @512 token = judul + lead
    "title_text": lambda a: a["title"].str.strip() + ". " + a["full_text"].str.strip(),
    "title": lambda a: a["title"].str.strip(),
    "full_text": lambda a: a["full_text"].str.strip(),
}


def read_articles(export_dir):
    stem = Path(export_dir) / "articles"
    pq = stem.with_suffix(".parquet")
    df = pd.read_parquet(pq) if pq.exists() else pd.read_csv(stem.with_suffix(".csv"))
    missing = [c for c in REQUIRED if c not in df.columns]
    assert not missing, f"kolom wajib hilang: {missing}"
    df = df.sort_values("article_id").reset_index(drop=True)
    assert (df["article_id"].to_numpy() == np.arange(len(df))).all(), "article_id harus 0..n-1 tanpa celah"
    assert df[["title", "full_text"]].notna().all().all(), "title/full_text ada yang kosong"
    return df


def articles_fingerprint(articles):  # harus sama dengan fungsi di notebook
    return hashlib.sha1("\n".join(articles["url"].astype(str)).encode("utf-8")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-dir", default="data/export")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True, help="nama pendek untuk nama file")
    ap.add_argument("--recipe", default="title_text", choices=list(RECIPES))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-seq-length", type=int, required=True, help="harus sama untuk semua model (512)")
    ap.add_argument("--anchors", default="src/anchors.txt")
    ap.add_argument("--no-normalize", action="store_true")
    args = ap.parse_args()

    import sentence_transformers
    import torch
    from sentence_transformers import SentenceTransformer

    torch.manual_seed(SEED)
    export_dir = Path(args.export_dir)
    articles = read_articles(export_dir)
    texts = RECIPES[args.recipe](articles).tolist()

    model = SentenceTransformer(args.model)
    model.max_seq_length = args.max_seq_length
    model.eval()
    kw = dict(batch_size=args.batch_size, normalize_embeddings=not args.no_normalize, convert_to_numpy=True)

    X = model.encode(texts, show_progress_bar=True, **kw).astype(np.float32)
    np.save(export_dir / f"embeddings_{args.tag}.npy", X)

    n_anchors = 0
    if Path(args.anchors).exists():
        anchors = [t.strip() for t in Path(args.anchors).read_text(encoding="utf-8").splitlines() if t.strip()]
        np.save(export_dir / f"embeddings_{args.tag}_anchors.npy", model.encode(anchors, **kw).astype(np.float32))
        n_anchors = len(anchors)

    try:
        pooling = model[1].get_pooling_mode_str()
    except Exception:
        pooling = None
    meta = {
        "model": args.model,
        "revision": getattr(model[0].auto_model.config, "_commit_hash", None),
        "pooling": pooling,
        "recipe": args.recipe,
        "max_seq_length": int(model.max_seq_length),
        "normalized": not args.no_normalize,
        "n": int(X.shape[0]),
        "dim": int(X.shape[1]),
        "n_anchors": n_anchors,
        "articles_fingerprint": articles_fingerprint(articles),
        "created": datetime.now().isoformat(timespec="seconds"),
        "seed": SEED,
        "versions": {"sentence_transformers": sentence_transformers.__version__, "torch": torch.__version__},
    }
    (export_dir / f"embeddings_{args.tag}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Tersimpan: embeddings_{args.tag}.npy {X.shape}")


if __name__ == "__main__":
    main()
