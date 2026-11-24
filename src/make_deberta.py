"""DeBERTa untuk Tugas 3 -> data/export/.  Dua mode:

  embed : rata-rata token (mean pooling, L2-normalized) -> embeddings_<tag>.npy/.json (+ _anchors.npy)
  probs : probabilitas kelas sentimen dari checkpoint fine-tuned -> <tag>_probs.npy/.json

    pip install transformers sentencepiece protobuf
    python src/make_deberta.py --mode embed --model microsoft/deberta-v3-base --tag deberta_base --max-seq-length 512
    python src/make_deberta.py --mode probs --model nickmuchi/deberta-v3-base-finetuned-finance-text-classification --tag deberta_ft --max-seq-length 512
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
    ap.add_argument("--mode", required=True, choices=["embed", "probs"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--export-dir", default="data/export")
    ap.add_argument("--recipe", default="title_text", choices=list(RECIPES))
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-seq-length", type=int, required=True)
    ap.add_argument("--anchors", default="src/anchors.txt")
    args = ap.parse_args()

    import torch
    import transformers
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    export_dir = Path(args.export_dir)
    articles = read_articles(export_dir)
    tok = AutoTokenizer.from_pretrained(args.model)
    cls = AutoModelForSequenceClassification if args.mode == "probs" else AutoModel
    model = cls.from_pretrained(args.model).to(dev).eval()   # fp32 (DeBERTa-v3 sering bermasalah di fp16)

    @torch.no_grad()
    def run(texts):
        out = []
        for i in range(0, len(texts), args.batch_size):
            enc = tok(texts[i:i + args.batch_size], truncation=True, max_length=args.max_seq_length,
                      padding=True, return_tensors="pt").to(dev)
            if args.mode == "probs":
                v = torch.softmax(model(**enc).logits, dim=-1)
            else:
                h, m = model(**enc).last_hidden_state, enc["attention_mask"].unsqueeze(-1)
                v = torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=-1)
            out.append(v.cpu().numpy())
            if (i // args.batch_size) % 20 == 0:
                print(f"{i}/{len(texts)}", flush=True)
        return np.concatenate(out).astype(np.float32)

    X = run(RECIPES[args.recipe](articles).tolist())
    meta = {
        "model": args.model, "mode": args.mode, "recipe": args.recipe,
        "revision": getattr(model.config, "_commit_hash", None),
        "max_seq_length": args.max_seq_length, "n": int(X.shape[0]), "dim": int(X.shape[1]),
        "articles_fingerprint": articles_fingerprint(articles),
        "created": datetime.now().isoformat(timespec="seconds"), "seed": SEED,
        "versions": {"transformers": transformers.__version__, "torch": torch.__version__},
    }
    if args.mode == "probs":
        meta["labels"] = [model.config.id2label[i].lower() for i in range(model.config.num_labels)]
        stem = f"{args.tag}_probs"
    else:
        meta["pooling"] = "mean (attention mask), L2-normalized"
        stem = f"embeddings_{args.tag}"
        if Path(args.anchors).exists():
            anchors = [t.strip() for t in Path(args.anchors).read_text(encoding="utf-8").splitlines() if t.strip()]
            np.save(export_dir / f"{stem}_anchors.npy", run(anchors))
            meta["n_anchors"] = len(anchors)
    np.save(export_dir / f"{stem}.npy", X)
    (export_dir / f"{stem}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Tersimpan: {stem}.npy {X.shape}", meta.get("labels", ""))


if __name__ == "__main__":
    main()
