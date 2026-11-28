import json

path = r'percobaan task 3/trial2_bertopic.ipynb'
try:
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    print("trial2_bertopic.ipynb cells:", len(nb['cells']))
    for i, c in enumerate(nb['cells']):
        src = "".join(c.get('source', []))
        if 'bertopic' in src.lower() or 'topic_model' in src.lower():
            print(f"Cell {i}:")
            print(src[:400])
            print("="*40)
except Exception as e:
    print(e)
