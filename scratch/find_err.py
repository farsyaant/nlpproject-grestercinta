import os
import transformers

tf_dir = os.path.dirname(transformers.__file__)
target = "Could not import module"

for root, dirs, files in os.walk(tf_dir):
    for f in files:
        if f.endswith('.py'):
            p = os.path.join(root, f)
            try:
                with open(p, 'r', encoding='utf-8', errors='ignore') as fp:
                    content = fp.read()
                    if target.lower() in content.lower():
                        print(f"Found in {p}")
                        for line in content.splitlines():
                            if target.lower() in line.lower() or "requirements defined correctly" in line.lower():
                                print("  ", line.strip())
            except Exception as e:
                pass
