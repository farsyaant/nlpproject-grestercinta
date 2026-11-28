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
for ci in [4, 6, 8, 10, 12]:
    exec("".join(nb["cells"][ci]["source"]), ns)

# Verify SPSM & MCM forward pass + loss computation using locally cached roberta-base (1 batch)
tok_r = ns["AutoTokenizer"].from_pretrained("roberta-base", local_files_only=True)
spsm = ns["SPSMRegressor"]("roberta-base")
mcm = ns["MCMClassifier"]("roberta-base")

# Avoid downloading roberta-base if AutoModel needs local_files_only; note: above already succeeded if cached!
sample_ds = ns["DailyFXDataset"](ns["df_fx"].iloc[:2]["filtered_fx_text"].values,
                                 ns["df_fx"].iloc[:2]["llm_sentiment"].values,
                                 ns["df_fx"].iloc[:2]["label_dir_next"].values,
                                 tok_r, max_len=64)
b = next(iter(ns["DataLoader"](sample_ds, batch_size=2)))
out_s = spsm(b["input_ids"], b["attention_mask"])
out_m = mcm(b["input_ids"], b["attention_mask"])
assert out_s.shape == (2,) and float(out_s.min()) >= -1.0 and float(out_s.max()) <= 1.0
assert out_m.shape == (2, 2)
loss_s = nn.MSELoss()(out_s, b["sent_label"])
loss_m = nn.CrossEntropyLoss()(out_m, b["dir_label"])
print(f"Verified SPSM forward+MSE ({loss_s.item():.4f}) & MCM forward+CE ({loss_m.item():.4f}) cleanly!")

# Verify downstream cells 18 & 20 with proxy predictions from LDA/Sentiment just to confirm plotting & CSV export work cleanly
ns["spsm_pred_tr"] = ns["df_fx"].iloc[ns["idx_tr"]]["llm_sentiment"].values
ns["spsm_pred_va"] = ns["df_fx"].iloc[ns["idx_va"]]["llm_sentiment"].values
ns["spsm_pred_te"] = ns["df_fx"].iloc[ns["idx_te"]]["llm_sentiment"].values
ns["mcm_prob_tr"]  = ns["clf_lda"].predict_proba(ns["X_lda_tr"])[:, 1]
ns["mcm_prob_va"]  = ns["clf_lda"].predict_proba(ns["X_lda_va"])[:, 1]
ns["mcm_prob_te"]  = ns["prob_lda_te"]
ns["mcm_tr_hist"], ns["mcm_va_hist"] = [0.69, 0.68], [0.69, 0.685]
ns["spsm_tr_hist"], ns["spsm_va_hist"] = [0.25, 0.22], [0.26, 0.23]

for ci in [18, 20]:
    exec("".join(nb["cells"][ci]["source"]), ns)

print("ALL VERIFICATION CHECKS FOR TRIAL 5 PASSED!")
