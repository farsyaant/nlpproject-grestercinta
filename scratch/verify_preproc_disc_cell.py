import ast
import json
import matplotlib
matplotlib.use("Agg")

nb_path = "percobaan task 3/preprocessing.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for ci in [-3, -1]:
    ast.parse("".join(nb["cells"][ci]["source"]))
print(f"Verified JSON & AST syntax! Total cells={len(nb['cells'])}")

ns = {"display": lambda df: print(df.to_string(index=False))}
exec("".join(nb["cells"][-3]["source"]), ns)
exec("".join(nb["cells"][-1]["source"]), ns)
print("DISCRIMINATIVE WORDS + WORDCLOUD CELLS EXECUTED CLEANLY!")
