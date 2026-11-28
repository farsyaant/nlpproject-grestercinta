import json

path = r'd:\Downloads\trial-5.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("=== CELL 7 (Markdown) ===")
print("".join(nb['cells'][7]['source']))
print("=== CELL 8 (Code) ===")
print("".join(nb['cells'][8]['source']))
