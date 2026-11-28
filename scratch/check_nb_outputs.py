import json

def check_outputs(path):
    print(f"=== Checking {path} ===")
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    for i, cell in enumerate(nb['cells']):
        if cell.get('cell_type') == 'code' and cell.get('outputs'):
            print(f"--- Cell {i} (outputs: {len(cell['outputs'])}) ---")
            for out in cell['outputs']:
                if 'text' in out:
                    text = "".join(out['text']).strip()
                    print(text[:400])
                elif 'data' in out and 'text/plain' in out['data']:
                    text = "".join(out['data']['text/plain']).strip()
                    print(text[:400])
                elif 'ename' in out:
                    print(f"ERROR: {out.get('ename')}: {out.get('evalue')}")

check_outputs('d:/Downloads/trial3_pipeline.ipynb')
check_outputs('percobaan task 3/trial3_pipeline.ipynb')
