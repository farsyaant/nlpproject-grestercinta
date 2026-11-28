import json
import pandas as pd
import numpy as np

paths = [
    r"d:/uni - 5th sem/lecture/NLP/tugas 1/nlpproject-grestercinta/percobaan task 3/trial3_pipeline.ipynb",
    r"d:/Downloads/trial3_pipeline.ipynb",
    r"d:/Downloads/trial-5.ipynb"
]

for p in paths:
    with open(p, "r", encoding="utf-8") as f:
        nb = json.load(f)
    print(f"Verified {p}: {len(nb['cells'])} cells")
    all_src = " ".join("".join(c.get("source", [])) for c in nb["cells"])
    for key in ["finbert_unique_titles.csv", "US_CASE", "pol_margin", "entity_dir_margin",
                "gpr_weak_margin", "consec_up_days", "dist_ma20", "GradientBoostingClassifier",
                "DirAwareMLP", "log_ret"]:
        assert key in all_src, f"Missing {key} in {p}"

with open(paths[0], "r", encoding="utf-8") as f:
    nb = json.load(f)

ns = {"display": lambda df: print(df.to_string())}
for ci in [4, 6, 10]:
    exec("".join(nb["cells"][ci]["source"]), ns)

# Stub topic_prob_0..4 (cell 8) & pca_features (cell 14) just to avoid CPU transformer/BERTopic wait in smoke test
for k in range(5):
    ns["df_raw"][f"topic_prob_{k}"] = 0.0

exec("".join(nb["cells"][12]["source"]), ns)
ns["pca_features"] = pd.DataFrame(0.0, index=ns["text_daily_base"].index, columns=[f"pca_{i}" for i in range(6)])

for ci in [16, 18, 20, 22, 24]:
    exec("".join(nb["cells"][ci]["source"]), ns)

print("\nSMOKE TEST PASSED SUCCESSFULLY!")
