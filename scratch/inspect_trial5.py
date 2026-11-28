import json

path = r'd:\Downloads\trial-5.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")

for i, cell in enumerate(nb['cells']):
    ctype = cell.get('cell_type')
    source = "".join(cell.get('source', []))
    lines = source.splitlines()
    first_line = lines[0] if lines else ""
    
    if ctype == 'markdown':
        print(f"Cell {i:02d} [MD]: {first_line[:80]}")
    else:
        out_types = [o.get('output_type') for o in cell.get('outputs', [])]
        print(f"Cell {i:02d} [CODE]: {first_line[:60]} | outputs: {len(out_types)} ({out_types[:3]})")
