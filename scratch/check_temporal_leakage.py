import json

with open('percobaan task 3/trial3_pipeline.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    src = "".join(cell.get('source', []))
    if 'vol_5d' in src:
        print(f"Cell {i}:")
        for line in src.splitlines():
            if any(k in line for k in ['ret', 'vol', 'mom']):
                print("  ", line)
