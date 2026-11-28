with open('d:/Downloads/trial-5.ipynb', 'r', encoding='utf-8') as f:
    text = f.read()

target = 'topic_prob_{k}" for k in range(5)'
pos = 0
while True:
    idx = text.find(target, pos)
    if idx == -1:
        break
    print(f"Found at {idx}:")
    print(text[max(0, idx-50):min(len(text), idx+150)])
    print("="*40)
    pos = idx + len(target)
