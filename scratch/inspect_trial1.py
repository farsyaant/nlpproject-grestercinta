import json

with open('percobaan task 3/trial1.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    source = ''.join(cell.get('source', []))
    if any(k in source.lower() for k in ['benchmark', 'rmse', 'mlp', 'evaluate', 'test_rmse']):
        print(f"Cell {i} ({cell['cell_type']}):")
        lines = source.splitlines()
        for line in lines[:25]:
            print("  ", line)
        if len(lines) > 25:
            print("   ...")
        print('='*50)
