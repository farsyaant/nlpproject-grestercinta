import os
os.environ["HF_HUB_OFFLINE"] = "1"
import ast
import json
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

nb_path = r"d:/uni - 5th sem/lecture/NLP/tugas 1/nlpproject-grestercinta/percobaan task 3/trial5_roberta_spsm_mcm_lda.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Verified JSON: {nb_path} ({len(nb['cells'])} cells)")
for idx, c in enumerate(nb["cells"]):
    if c["cell_type"] == "code" and idx != 2:
        ast.parse("".join(c["source"]))
print("All code cells passed AST syntax check!")

ns = {"display": lambda df: print(df.to_string())}
for ci in [4, 6, 8, 10]:
    exec("".join(nb["cells"][ci]["source"]), ns)

# Check LeanSPSM parameter count & 1-batch forward pass from Cell 12 definitions
cell12_src = "".join(nb["cells"][12]["source"])
# Extract class definitions only (before `if cache_spsm_path.exists()`)
defs_src = cell12_src.split("if cache_spsm_path.exists()")[0]
exec(defs_src, ns)

tok_r = ns["AutoTokenizer"].from_pretrained("roberta-base", local_files_only=True)
spsm_chk = ns["LeanSPSM"]("roberta-base")
tot_p = sum(p.numel() for p in spsm_chk.parameters())
act_p = sum(p.numel() for p in spsm_chk.parameters() if p.requires_grad)
print(f"Verified LeanSPSM params: Total={tot_p:,} | Trainable (2-Layer+Head)={act_p:,} ({act_p/tot_p*100:.1f}%)")
ds_chk = ns["ArticleSentDataset"](ns["df_art"]["article_text"].iloc[:2], ns["df_art"]["llm_sentiment"].iloc[:2], tok_r, max_len=64)
b = next(iter(ns["DataLoader"](ds_chk, batch_size=2)))
out_s = spsm_chk(b["input_ids"], b["attention_mask"])
assert out_s.shape == (2,) and float(out_s.min()) >= -1.0 and float(out_s.max()) <= 1.0
print("Verified LeanSPSM 1-batch forward pass cleanly!")

# Supply daily SPSM proxy & Frozen RoBERTa 768-d proxy in memory (without writing proxy .npy to disk!)
ns["spsm_pred_tr"] = ns["df_fx"].iloc[ns["idx_tr"]]["llm_sentiment"].values
ns["spsm_pred_va"] = ns["df_fx"].iloc[ns["idx_va"]]["llm_sentiment"].values
ns["spsm_pred_te"] = ns["df_fx"].iloc[ns["idx_te"]]["llm_sentiment"].values

unm = pd.read_csv("data/processed/news_processed_unmerged.csv", usecols=["aligned_trading_date"])
unm_dates = pd.to_datetime(unm["aligned_trading_date"]).dt.normalize()
rob_art = 0.5 * np.load("data/cache_embeddings/roberta_title.npy") + 0.5 * np.load("data/cache_embeddings/roberta_fulltext.npy")
date_to_idx = {d: i for i, d in enumerate(ns["df_fx"]["Tanggal_dt"])}
X_rob_proxy = np.zeros((len(ns["df_fx"]), 768), dtype=np.float32)
for dt_val, grp in unm.groupby(unm_dates):
    if dt_val in date_to_idx:
        X_rob_proxy[date_to_idx[dt_val]] = rob_art[grp.index].mean(axis=0)
ns["X_rob_daily"] = X_rob_proxy

# Run Cell 14 downstream part (after `X_rob_daily` extraction), plus Cells 16 & 18!
cell14_src = "".join(nb["cells"][14]["source"])
cell14_head_src = "# Standarisasi fitur Frozen RoBERTa" + cell14_src.split("# Standarisasi fitur Frozen RoBERTa")[1]
exec(cell14_head_src, ns)
exec("".join(nb["cells"][16]["source"]), ns)
exec("".join(nb["cells"][18]["source"]), ns)

print("ALL RIGHT-SIZED TRIAL 5 CHECKS PASSED!")
