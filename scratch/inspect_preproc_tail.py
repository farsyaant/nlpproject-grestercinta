import json

with open("percobaan task 3/preprocessing.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print("Total cells:", len(nb["cells"]))
for i in range(29, len(nb["cells"])):
    c = nb["cells"][i]
    print(f"=== Cell {i} ({c['cell_type']}) ===")
    print("".join(c.get("source", []))[:900])
