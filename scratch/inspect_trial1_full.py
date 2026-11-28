import json

with open("percobaan task 3/trial1.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for i in [0, 6, 10, 15, 17, 18]:
    print(f"=== CELL {i} ({nb['cells'][i]['cell_type']}) ===")
    print("".join(nb["cells"][i].get("source", []))[:1200])
    for o in nb["cells"][i].get("outputs", []):
        if "text" in o:
            print("OUT:", "".join(o["text"])[:600])
        elif "data" in o and "text/plain" in o["data"]:
            print("DATA:", "".join(o["data"]["text/plain"])[:600])
