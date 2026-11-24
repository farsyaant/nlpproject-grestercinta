"""Probabilitas sentimen FinBERT per artikel -> data/export/finbert_probs.npy (+ .json).

    pip install transformers torch
    python src/make_finbert.py --max-seq-length 512

Kolom .npy mengikuti urutan `labels` di .json (ProsusAI/finbert: positive, negative, neutral).
Input teks sama dengan make_embeddings.py (judul + isi, dipotong di max-seq-length).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from make_embeddings import RECIPES, SEED, articles_fingerprint, read_articles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-dir", default="data/export")
    ap.add_argument("--model", default="ProsusAI/finbert")
    ap.add_argument("--recipe", default="title_text", choices=list(RECIPES))
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-seq-length", type=int, required=True)
    args = ap.parse_args()

    import torch
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    export_dir = Path(args.export_dir)
    articles = read_articles(export_dir)
    texts = RECIPES[args.recipe](articles).tolist()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device).eval()
    labels = [model.config.id2label[i].lower() for i in range(model.config.num_labels)]

    out = []
    with torch.no_grad():
        for i in range(0, len(texts), args.batch_size):
            enc = tok(texts[i:i + args.batch_size], truncation=True, max_length=args.max_seq_length,
                      padding=True, return_tensors="pt").to(device)
            out.append(torch.softmax(model(**enc).logits, dim=-1).cpu().numpy())
            if (i // args.batch_size) % 20 == 0:
                print(f"{i}/{len(texts)}", flush=True)
    P = np.concatenate(out).astype(np.float32)

    np.save(export_dir / "finbert_probs.npy", P)
    meta = {
        "model": args.model,
        "revision": getattr(model.config, "_commit_hash", None),
        "labels": labels,
        "recipe": args.recipe,
        "max_seq_length": args.max_seq_length,
        "n": int(P.shape[0]),
        "articles_fingerprint": articles_fingerprint(articles),
        "created": datetime.now().isoformat(timespec="seconds"),
        "seed": SEED,
        "versions": {"transformers": transformers.__version__, "torch": torch.__version__},
    }
    (export_dir / "finbert_probs.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Tersimpan: finbert_probs.npy {P.shape} labels={labels}")


if __name__ == "__main__":
    main()
