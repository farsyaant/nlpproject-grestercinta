import ast
import base64
import contextlib
import io
import json
import os
import sys
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["NUMBA_CACHE_DIR"] = os.path.abspath("scratch/numba_cache")
os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)
try:
    import plotly.io as pio
    import plotly.basedatatypes as pbd
    pio.renderers.default = "json"
    pbd.BaseFigure.show = lambda self, *a, **k: print(f"[Plotly Figure rendered: {type(self).__name__}]")
except Exception:
    pass

def execute_notebook(nb_path, save_back=True, out_fig_dir=None):
    t_start = time.time()
    print(f"\n==================== EXECUTING: {nb_path} ====================")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Ensure matplotlib charts render inline into notebook outputs and optionally save PNGs
    fig_counter = [0]
    def make_env(outputs_list):
        def _display(obj):
            if isinstance(obj, pd.DataFrame):
                txt = obj.to_string()
                print(txt)
                outputs_list.append({
                    "output_type": "execute_result",
                    "execution_count": 1,
                    "metadata": {},
                    "data": {"text/plain": [txt], "text/html": [obj.to_html()]}
                })
            elif hasattr(obj, "show") and "plotly" in type(obj).__module__:
                print(f"[Plotly Figure: {type(obj).__name__}]")
            else:
                print(repr(obj))

        class TqdmStub:
            def __init__(self, iterable=None, *args, **kwargs):
                self.it = iterable
            def __iter__(self):
                return iter(self.it) if self.it is not None else iter([])
            def __getattr__(self, name):
                return lambda *a, **k: None

        return _display

    ns = {}
    exec_count = 0

    for ci, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        lines = [l for l in src.splitlines() if not l.lstrip().startswith(("!", "%"))]
        clean_src = "\n".join(lines)
        if not clean_src.strip():
            continue

        exec_count += 1
        outputs = []
        buf = io.StringIO()
        ns["display"] = make_env(outputs)

        # Hook plt.show so figures are captured in notebook outputs
        orig_show = plt.show
        def _hooked_show(*args, **kwargs):
            for fnum in plt.get_fignums():
                fig = plt.figure(fnum)
                png_buf = io.BytesIO()
                fig.savefig(png_buf, format="png", dpi=140, bbox_inches="tight")
                b64 = base64.b64encode(png_buf.getvalue()).decode("ascii")
                outputs.append({
                    "output_type": "display_data",
                    "metadata": {},
                    "data": {"image/png": b64, "text/plain": ["<Figure>"]}
                })
                if out_fig_dir:
                    fig_counter[0] += 1
                    os.makedirs(out_fig_dir, exist_ok=True)
                    base = os.path.splitext(os.path.basename(nb_path))[0]
                    fig.savefig(os.path.join(out_fig_dir, f"{base}_cell{ci}_fig{fig_counter[0]}.png"), dpi=140, bbox_inches="tight")
                plt.close(fig)
        plt.show = _hooked_show

        t0 = time.time()
        class TeeOut(io.StringIO):
            def __init__(self, real):
                super().__init__()
                self.real = real
            def write(self, s):
                self.real.write(s)
                self.real.flush()
                return super().write(s)
        tee_buf = TeeOut(sys.stdout)
        print(f"--- [cell {ci:02d}] starting ---")
        try:
            with contextlib.redirect_stdout(tee_buf):
                tree = ast.parse(clean_src, mode="exec")
                if tree.body and isinstance(tree.body[-1], ast.Expr):
                    last_expr = ast.Expression(tree.body.pop().value)
                    if tree.body:
                        exec(compile(tree, f"{nb_path}#cell_{ci}", "exec"), ns)
                    res_val = eval(compile(last_expr, f"{nb_path}#cell_{ci}_expr", "eval"), ns)
                    if res_val is not None:
                        ns["display"](res_val)
                else:
                    exec(compile(tree, f"{nb_path}#cell_{ci}", "exec"), ns)
        finally:
            plt.show = orig_show

        out_txt = tee_buf.getvalue()
        if out_txt:
            outputs.insert(0, {
                "output_type": "stream",
                "name": "stdout",
                "text": [line + "\n" for line in out_txt.splitlines()]
            })
        print(f"--- [cell {ci:02d} finished in {time.time()-t0:.1f}s] ---")

        cell["execution_count"] = exec_count
        cell["outputs"] = outputs

    if save_back:
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"FINISHED {nb_path} in {time.time()-t_start:.1f}s!")

if __name__ == "__main__":
    nbs = sys.argv[1:]
    for p in nbs:
        execute_notebook(p, save_back=True, out_fig_dir="percobaan task 3/reports/figures/executed")
