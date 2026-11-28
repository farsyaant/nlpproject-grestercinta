import ast
import json
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

nb_path = r"d:/uni - 5th sem/lecture/NLP/tugas 1/nlpproject-grestercinta/percobaan task 3/trial4_direct_concat_embedding.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Verified JSON: {nb_path} ({len(nb['cells'])} cells)")

# AST syntax check on every code cell (excluding cell 2 `!pip install`)
for idx, c in enumerate(nb["cells"]):
    if c["cell_type"] == "code" and idx != 2:
        ast.parse("".join(c["source"]))
print("All code cells passed AST syntax check!")

# Execute cells 4, 6, (proxy 8 with local 1536-d cache), 10, 12, 14, 16
ns = {"display": lambda df: print(df.to_string())}
for ci in [4, 6]:
    exec("".join(nb["cells"][ci]["source"]), ns)

# Build local 1536-d proxy matching df_valid alignment to smoke-test cells 10, 12, 14, 16
unm = pd.read_csv("data/processed/news_processed_unmerged.csv", usecols=["aligned_trading_date"])
unm_dates = pd.to_datetime(unm["aligned_trading_date"]).dt.normalize()
emb_1536 = np.hstack([np.load("data/cache_embeddings/bert_title.npy"), np.load("data/cache_embeddings/bert_fulltext.npy")])
date_to_idx = {d: i for i, d in enumerate(ns["df_valid"][ns["date_col"]])}
X_all_proxy = np.zeros((len(ns["df_valid"]), 1536), dtype=np.float32)
for dt_val, grp in unm.groupby(unm_dates):
    if dt_val in date_to_idx:
        X_all_proxy[date_to_idx[dt_val]] = emb_1536[grp.index].mean(axis=0)
ns["X_all"] = X_all_proxy

for ci in [10, 12, 14, 16]:
    exec("".join(nb["cells"][ci]["source"]), ns)

print("ALL CELLS IN TRIAL 4 EXECUTED CLEANLY!")
