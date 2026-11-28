with open('d:/Downloads/trial-5.ipynb', 'r', encoding='utf-8') as f:
    text = f.read()

idx = text.find('display(df_raw[[f"topic_prob_{k}" for k in range(5)]].head())\\n"')
if idx != -1:
    print("Found! snippet:")
    print(text[idx:idx+400])
    idx2 = text.find('{"cell_type":"markdown","source":"## 4.', idx)
    print("Next cell starts at:", idx2)
    print("Target block between them:")
    print(text[idx:idx2])
