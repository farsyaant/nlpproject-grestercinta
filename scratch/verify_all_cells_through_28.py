import json
import matplotlib
matplotlib.use("Agg")
import pandas as pd

with open(r"d:/uni - 5th sem/lecture/NLP/tugas 1/nlpproject-grestercinta/percobaan task 3/trial3_pipeline.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

ns = {"display": lambda df: print(df.to_string())}
for ci in [4, 6, 10]:
    exec("".join(nb["cells"][ci]["source"]), ns)
for k in range(5):
    ns["df_raw"][f"topic_prob_{k}"] = 0.0
exec("".join(nb["cells"][12]["source"]), ns)
ns["pca_features"] = pd.DataFrame(0.0, index=ns["text_daily_base"].index, columns=[f"pca_{i}" for i in range(6)])
for ci in [16, 18, 20, 22, 24, 26, 28]:
    exec("".join(nb["cells"][ci]["source"]), ns)

print("ALL CELLS THROUGH 28 EXECUTED CLEANLY!")
