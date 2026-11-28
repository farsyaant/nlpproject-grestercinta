import json
import re
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch

print("CUDA available:", torch.cuda.is_available())
ns_common = {"display": lambda x: print(x.head(5) if hasattr(x, "head") else x)}

# 1. Verify Trial 1 tuning + eval + plot cells
print("\n=== TEST 1: trial1.ipynb ===")
nb1 = json.load(open("percobaan task 3/trial1.ipynb", encoding="utf-8"))
ns1 = dict(ns_common)
for ci in [1, 3, 5]:
    exec("".join(nb1["cells"][ci]["source"]), ns1)

emb_smoke = ns1["extract_embeddings"](["Oil prices rise", "Dollar falls"], "bert-base-uncased", desc_tag="smoke", batch_size=2, max_length=16)
print("Frozen BERT smoke extract_embeddings OK:", emb_smoke.shape)

n_tot = len(ns1["bi_clean"])
exec("".join(nb1["cells"][8]["source"]).split("for key, emb in")[0], ns1)
ns1["daily_datasets"] = {
    "finbert_title": np.random.randn(n_tot, 768).astype(np.float32),
    "bert_title": np.random.randn(n_tot, 768).astype(np.float32),
}
ns1["n_total"] = n_tot
ns1["n_train"] = int(0.70 * n_tot)
ns1["n_val"] = int(0.15 * n_tot)
ns1["idx_train"] = np.arange(0, ns1["n_train"])
ns1["idx_val"] = np.arange(ns1["n_train"], ns1["n_train"] + ns1["n_val"])
ns1["idx_test"] = np.arange(ns1["n_train"] + ns1["n_val"], n_tot)
ns1["y_all"] = ns1["bi_clean"]["target"].values.astype(np.float32)
ns1["y_train"] = ns1["y_all"][ns1["idx_train"]]
ns1["y_val"] = ns1["y_all"][ns1["idx_val"]]
ns1["y_test"] = ns1["y_all"][ns1["idx_test"]]

for ci in [10, 12, 13, 15, 17]:
    exec("".join(nb1["cells"][ci]["source"]), ns1)
print("Trial 1 tuning & plot PASSED!")

# 2. Verify Trial 3 Stage 1 & Stage 2 tuning + ablation + plot cells
print("\n=== TEST 2: trial3_pipeline.ipynb ===")
nb3 = json.load(open("percobaan task 3/trial3_pipeline.ipynb", encoding="utf-8"))
ns3 = dict(ns_common)
for ci in [4, 6]:
    exec("".join(nb3["cells"][ci]["source"]), ns3)
for k in range(5):
    ns3["df_raw"][f"topic_prob_{k}"] = np.random.rand(len(ns3["df_raw"])).astype(np.float32)
ns3["fin_map"] = {}
for ci in [12]:
    exec("".join(nb3["cells"][ci]["source"]), ns3)
# Provide pca_features matching df_raw dates
dates_u = pd.to_datetime(ns3["df_raw"][ns3["date_col"]]).drop_duplicates().sort_values()
ns3["pca_features"] = pd.DataFrame(np.random.randn(len(dates_u), 6), index=dates_u, columns=[f"pca_{i}" for i in range(6)])
for ci in [16, 18, 20, 22, 24, 26]:
    exec("".join(nb3["cells"][ci]["source"]), ns3)
print("Trial 3 tuning, ablation, & plot PASSED!")

# 3. Verify Trial 4 PCA + MLP/ML tuning + plot cells
print("\n=== TEST 3: trial4_direct_concat_embedding.ipynb ===")
nb4 = json.load(open("percobaan task 3/trial4_direct_concat_embedding.ipynb", encoding="utf-8"))
ns4 = dict(ns_common)
for ci in [4, 6]:
    exec("".join(nb4["cells"][ci]["source"]), ns4)
ns4["X_all"] = np.random.randn(len(ns4["df_valid"]), 1536).astype(np.float32)
for ci in [10, 12, 14]:
    exec("".join(nb4["cells"][ci]["source"]), ns4)
print("Trial 4 tuning & plot PASSED!")

# 4. Verify Trial 5 LDA + Frozen SPSM Head + Frozen MCM Head + Hybrid + Ridge tuning + plot
print("\n=== TEST 4: trial5_roberta_spsm_mcm_lda.ipynb ===")
nb5 = json.load(open("percobaan task 3/trial5_roberta_spsm_mcm_lda.ipynb", encoding="utf-8"))
ns5 = dict(ns_common)
for ci in [4, 6]:
    exec("".join(nb5["cells"][ci]["source"]), ns5)

df_b = ns5["df_base"]
dc = ns5["date_col"]
tc, bc = ns5["title_col"], ns5["body_col"]
FX_PAT = re.compile(r"\b(dollar|usd|rupiah|idr|fed|inflation|oil|crude|tariff|tariffs|sanction|economy|trade|rate)\b", re.I)
d_recs, a_recs = [], []
for _, row in df_b.iterrows():
    ts = [t.strip() for t in str(row[tc]).split(" | ") if t.strip() and t.strip().lower() != "nan"]
    if not ts:
        continue
    kept = [t for t in ts if FX_PAT.search(t)] or ts[:1]
    for k in kept:
        a_recs.append({dc: row[dc], "article_text": k, "llm_sentiment": float(np.random.uniform(-0.5, 0.5)), "llm_explanation": "test"})
    d_recs.append({
        dc: row[dc], "rate_close": row["rate_close"], "log_ret_next": row["log_ret_next"],
        "label_dir_next": row["label_dir_next"], "filtered_fx_text": " ".join(kept),
        "llm_sentiment": float(np.random.uniform(-0.5, 0.5)), "llm_explanation": "test", "kept_articles": len(kept)
    })
ns5["df_fx"] = pd.DataFrame(d_recs)
ns5["df_art"] = pd.DataFrame(a_recs)
n_fx = len(ns5["df_fx"])
n_tr, n_va = int(n_fx * 0.70), int(n_fx * 0.15)
ns5["idx_tr"] = np.arange(0, n_tr)
ns5["idx_va"] = np.arange(n_tr, n_tr + n_va)
ns5["idx_te"] = np.arange(n_tr + n_va, n_fx)
ns5["train_end_date"] = ns5["df_fx"][dc].iloc[ns5["idx_tr"][-1]]
ns5["val_end_date"] = ns5["df_fx"][dc].iloc[ns5["idx_va"][-1]]

exec("".join(nb5["cells"][10]["source"]), ns5)

c12_src = "".join(nb5["cells"][12]["source"])
ns5["E_art"] = np.random.randn(len(ns5["df_art"]), 768).astype(np.float32)
exec("tr_a = " + c12_src.split("tr_a = ")[1], ns5)

c14_src = "".join(nb5["cells"][14]["source"])
ns5["X_rob_daily"] = np.random.randn(n_fx, 768).astype(np.float32)
exec("sc_rob = " + c14_src.split("sc_rob = ")[1], ns5)

exec("".join(nb5["cells"][16]["source"]), ns5)
exec("".join(nb5["cells"][18]["source"]), ns5)
print("Trial 5 tuning & plot PASSED!")
