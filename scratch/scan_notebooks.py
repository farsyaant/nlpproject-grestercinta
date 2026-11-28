import json, re
from pathlib import Path

for folder in ["percobaan task 3"]:
    for p in sorted(Path(folder).glob("*.ipynb")):
        nb = json.load(open(p, encoding="utf-8"))
        src = "\n".join("".join(c["source"]) for c in nb["cells"])
        cache_hits = [x.strip() for x in re.findall(r".*(?:cache|npz|npy|pkl|CACHE|\.exists\().*", src, re.I)]
        freeze_hits = [x.strip() for x in re.findall(r".*(?:freeze|requires_grad|unfreeze|finetune|fine_tune|AutoModel|Trainer|Longformer|RoBERTa|Roberta|Deberta|FinBERT|SentenceTransformer|top_layers).*".replace(".*", ".*"), src, re.I)]
        tune_hits = [x.strip() for x in re.findall(r".*(?:GridSearch|RandomizedSearch|optuna|tune|tuning|param_grid|best_params|TimeSeriesSplit).*".replace(".*", ".*"), src, re.I)]
        print(f"\n=== {p} ({len(nb['cells'])} cells) ===")
        print("  CACHE:", len(cache_hits))
        for h in cache_hits[:10]: print("   ", h[:120])
        print("  MODEL/FREEZE:", len(freeze_hits))
        for h in freeze_hits[:12]: print("   ", h[:120])
        print("  TUNE:", len(tune_hits))
        for h in tune_hits[:10]: print("   ", h[:120])
