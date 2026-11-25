import sys
import time
import traceback
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

RUN = Path(sys.argv[1]).resolve()
NB_IN = RUN / sys.argv[2]
NB_OUT = NB_IN.with_name(NB_IN.stem + "_executed.ipynb")


class LoggingClient(NotebookClient):
    def on_cell_start(self, cell, cell_index):
        print(f"[start {cell_index:02d}] {time.strftime('%H:%M:%S')}", flush=True)

    def on_cell_complete(self, cell, cell_index):
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream":
                sys.stdout.write("".join(out.get("text", "")))
            elif out.get("output_type") == "error":
                sys.stdout.write("\n".join(out.get("traceback", [])) + "\n")
        print(f"[done  {cell_index:02d}] {time.strftime('%H:%M:%S')}", flush=True)
        try:
            nbformat.write(self.nb, NB_OUT)
        except Exception as e:
            print("gagal simpan sementara:", e, flush=True)


def main():
    nb = nbformat.read(NB_IN, as_version=4)
    client = LoggingClient(
        nb,
        timeout=None,
        kernel_name="model_c_v1",
        resources={"metadata": {"path": str(RUN)}},
        allow_errors=False,
        record_timing=True,
    )
    status = "ok"
    try:
        client.execute()
    except CellExecutionError as e:
        status = "cell_error"
        print("\nCELL ERROR:\n", str(e)[-4000:], flush=True)
    except Exception:
        status = "fatal"
        traceback.print_exc()
    finally:
        nbformat.write(nb, NB_OUT)
        print("status", status, "| executed notebook", NB_OUT, flush=True)


if __name__ == "__main__":
    main()
