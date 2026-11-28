import ast
import json
import re
from pathlib import Path

target_nbs = [
    Path("percobaan task 3/trial1.ipynb"),
    Path("percobaan task 3/trial2_bertopic.ipynb"),
    Path("percobaan task 3/trial3_pipeline.ipynb"),
    Path("percobaan task 3/trial4_direct_concat_embedding.ipynb"),
    Path("percobaan task 3/trial5_roberta_spsm_mcm_lda.ipynb"),
    Path("notebook/topic_modeling_lda_vs_bertopic_umap.ipynb"),
]

print("=== 1. SYNTAX, CACHE, FREEZE WEIGHTS, & TUNING AUDIT ===")
all_ok = True
for p in target_nbs:
    nb = json.load(open(p, encoding="utf-8"))
    code_cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    full_code = "\n".join(code_cells)

    # AST check
    for idx, src in enumerate(code_cells):
        clean_lines = [l for l in src.splitlines() if not l.lstrip().startswith(("!", "%"))]
        ast.parse("\n".join(clean_lines))

    # Cache check: ensure no .npy/.npz or cache CSV loading/saving in code
    bad_cache = []
    for line in full_code.splitlines():
        if re.search(r"np\.load|np\.save|CACHE_DIR|CACHE_EMB_PATH|cache_topic_path|fin_cache_path|emb_concat_cache|cache_emb_path|cache_daily_path|cache_art_path|cache_spsm_path|cache_rob_path|fin_uniq_path", line):
            bad_cache.append(line.strip())

    # Freeze weights check
    has_freeze = "requires_grad = False" in full_code or "requires_grad_(False)" in full_code
    has_unfreeze = 'encoder.layer.10' in full_code or 'requires_grad = True' in full_code

    # Tuning check
    has_tuning = bool(re.search(r"(tune_|param_grid|TimeSeriesSplit|Tuning|best_.*cfg|best_alpha|best_bt_cfg)", full_code))

    status = "OK" if (not bad_cache and has_freeze and not has_unfreeze and has_tuning) else "ISSUE"
    if status != "OK":
        all_ok = False
    print(f"[{status}] {p} | cells={len(nb['cells'])} | bad_cache={len(bad_cache)} | freeze={has_freeze} | unfreeze={has_unfreeze} | tuning={has_tuning}")
    for bc in bad_cache:
        print("   BAD CACHE LINE:", bc)

print("\n=== 2. SCAN ALL WORKSPACE NOTEBOOKS FOR REMAINING CACHES ===")
for p in sorted(Path(".").rglob("*.ipynb")):
    if ".ipynb_checkpoints" in str(p) or "scratch" in str(p):
        continue
    nb = json.load(open(p, encoding="utf-8"))
    full_code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    hits = [l.strip() for l in full_code.splitlines() if re.search(r"np\.load|np\.save|cache_emb|bertopic_probabilities|finbert_unique", l, re.I)]
    if hits:
        print(f"  WARN {p}: {hits}")
print("Audit done. ALL_OK =", all_ok)
