import json
import sys
from pathlib import Path

RUN = Path(r"trial-and-error/Ken/runs/model_c_v1")
MANIFEST = RUN / "logs" / "hash_cp0.json"

entries = json.loads(MANIFEST.read_text(encoding="utf-8-sig"))
bad, missing, ok = [], [], 0
for e in entries:
    p = Path(e["path"])
    if not p.exists():
        missing.append(e["path"])
        continue
    import hashlib

    h = hashlib.sha256(p.read_bytes()).hexdigest().upper()
    if h != e["sha256"].upper():
        bad.append((e["path"], e["sha256"], h))
    else:
        ok += 1

print(f"files checked : {len(entries)}")
print(f"unchanged     : {ok}")
print(f"missing       : {len(missing)}")
print(f"modified      : {len(bad)}")
for m in missing:
    print("  MISSING", m)
for path, before, after in bad:
    print(f"  CHANGED {path}\n    cp0={before}\n    now={after}")

sys.exit(0 if not bad and not missing else 1)
