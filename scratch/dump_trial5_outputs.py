import json

path = r'd:\Downloads\trial-5.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i in [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 28]:
    cell = nb['cells'][i]
    print(f"\n{'='*30} CELL {i} {'='*30}")
    # print code snippet
    code_lines = cell.get('source', [])
    print("CODE SNIPPET:")
    print("".join(code_lines[:5]).strip())
    print("\nOUTPUTS:")
    for out in cell.get('outputs', []):
        if 'text' in out:
            print("".join(out['text']).strip())
        elif 'data' in out:
            if 'text/plain' in out['data']:
                print("".join(out['data']['text/plain']).strip())
            if 'text/html' in out['data']:
                print("[HTML Table Output]")
                print("".join(out['data']['text/html'])[:400] + "...")
        elif 'ename' in out:
            print(f"ERROR: {out.get('ename')}: {out.get('evalue')}")
