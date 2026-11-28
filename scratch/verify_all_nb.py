import json

paths = [
    r"d:/uni - 5th sem/lecture/NLP/tugas 1/nlpproject-grestercinta/percobaan task 3/trial3_pipeline.ipynb",
    r"d:/Downloads/trial3_pipeline.ipynb",
    r"d:/Downloads/trial-5.ipynb"
]

for p in paths:
    with open(p, "r", encoding="utf-8") as f:
        nb = json.load(f)
    print(f"Path: {p}")
    print(f"  Cells: {len(nb['cells'])}")
    all_text = " ".join("".join(c.get("source", [])) for c in nb["cells"])
    checks = [
        ("shift(1) temporal", "shift(1)" in all_text),
        ("OOF Stacking (KFold)", "KFold" in all_text and "oof_pred" in all_text),
        ("BERTopic UMAP random_state", "random_state=SEED" in all_text and "BERTopic" in all_text),
        ("FinBERT Polarity", "sentiment_polarity" in all_text),
        ("GPR Lexicon", "GPR_TERMS" in all_text),
        ("Multi-Model Benchmark", "Ensemble Blend" in all_text),
        ("5-Panel Visualizations", "add_gridspec" in all_text)
    ]
    for k, v in checks:
        print(f"  - {k}: {v}")
