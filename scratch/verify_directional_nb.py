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
        ("flag_us & flag_idr", "flag_us" in all_text and "flag_idr" in all_text),
        ("Bipolar polarity", "polarity_sign" in all_text),
        ("Entity directional signal", "entity_directional_signal" in all_text),
        ("GPR weakening signal", "gpr_weakening_signal" in all_text),
        ("Time-series reversal (consec_up_days)", "consec_up_days" in all_text),
        ("Time-series MA20 dist", "dist_ma20" in all_text),
        ("Target log_return", "log_ret" in all_text),
        ("Stage 1 Direction Classifier OOF", "clf_fold" in all_text and "oof_pred_prob" in all_text),
        ("5-Panel Visualizations", "add_gridspec" in all_text)
    ]
    for k, v in checks:
        print(f"  - {k}: {v}")
