import json
from pathlib import Path

def summarize_nb(p):
    nb = json.load(open(p, encoding="utf-8"))
    print(f"\n==================== {p} ({len(nb['cells'])} cells) ====================")
    for i, c in enumerate(nb["cells"]):
        src = "".join(c["source"]).strip()
        first_line = src.splitlines()[0] if src else "<EMPTY>"
        print(f"  [{i:02d}] {c['cell_type']:8s} | {first_line[:110]}")

for p in sorted(Path("percobaan task 3").glob("*.ipynb")):
    if "preprocessing" in p.name:
        continue
    summarize_nb(p)

print("\n--- OTHER NOTEBOOKS ---")
for p in [Path("notebookbaru.ipynb")] + sorted(Path("notebook").glob("*.ipynb")):
    summarize_nb(p)
