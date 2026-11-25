import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[5]
RUN = ROOT / "trial-and-error" / "Ken" / "runs" / "model_c_v1"
DEVICE = "cuda"
MAX_LENGTH = 64

MODELS = {
    "bge_base": ("BAAI/bge-base-en-v1.5", "cls"),
    "harrier270m": ("microsoft/harrier-oss-v1-270m", "last"),
    "qwen3_06b": ("Qwen/Qwen3-Embedding-0.6B", "last"),
    "granite_r2": ("ibm-granite/granite-embedding-english-r2", "cls"),
    "jina_v5_nano": ("jinaai/jina-embeddings-v5-text-nano-classification", "last"),
    "finmodernbert": ("BatuhanECB/FinModernBERT-embed-large", "mean"),
}


def pool_hidden(h, mask, mode):
    if mode == "cls":
        return h[:, 0]
    if mode == "mean":
        m = mask.unsqueeze(-1).to(h.dtype)
        return (h * m).sum(1) / m.sum(1).clamp(min=1)
    if mode == "last":
        idx = (mask.sum(1) - 1).clamp(min=0)
        return h[torch.arange(h.size(0), device=h.device), idx]
    raise ValueError(mode)


def run_one(tag, repo, pooling, texts):
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(repo)
    tok.padding_side = "right"
    model = AutoModel.from_pretrained(repo, dtype=torch.float16).to(DEVICE).eval()
    t_load = time.time() - t0
    enc = tok(texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc).last_hidden_state
    vec = pool_hidden(out.float(), enc["attention_mask"], pooling).cpu().numpy()
    finite = bool(np.isfinite(vec).all())
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "tag": tag,
        "repo": repo,
        "pooling": pooling,
        "load_s": round(t_load, 1),
        "encode_s": round(time.time() - t0 - t_load, 2),
        "shape": list(vec.shape),
        "finite": finite,
        "vram_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }


def main():
    articles = pd.read_parquet(ROOT / "data" / "export" / "articles.parquet")
    texts = articles["title"].str.strip().head(16).tolist()
    print("device", DEVICE, torch.cuda.get_device_name(0), flush=True)
    rows = []
    for tag, (repo, pooling) in MODELS.items():
        try:
            r = run_one(tag, repo, pooling, texts)
            r["status"] = "ok"
        except Exception as e:
            r = {"tag": tag, "repo": repo, "pooling": pooling,
                 "status": "fail", "error": f"{type(e).__name__}: {str(e)[:300]}"}
            gc.collect()
            torch.cuda.empty_cache()
        rows.append(r)
        print(json.dumps(r, ensure_ascii=False), flush=True)
    out = RUN / "logs" / "cp3_smoke_encoders.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("written", out, flush=True)


if __name__ == "__main__":
    main()
