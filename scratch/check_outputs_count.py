import json

with open('d:/Downloads/trial-5.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if 'bertopic' in src.lower() and cell.get('cell_type') == 'code':
        print(f"Cell {i} outputs count: {len(cell.get('outputs', []))}")
