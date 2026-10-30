"""Validate structure and optionally execute the tutorial in a fresh kernel."""
import argparse
import base64
import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--output-dir", default=".validation/tutorial03")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = root / "Tutorial_03_PyTorch_Regression_and_Classification/tutorial_03_pytorch.ipynb"
    notebook = nbformat.read(source, as_version=4)
    nbformat.validate(notebook)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert code_cells and all(not cell.outputs and cell.execution_count is None for cell in code_cells)
    assert len({cell.id for cell in notebook.cells}) == len(notebook.cells)
    print(f"Validated notebook structure: {len(notebook.cells)} cells, {len(code_cells)} code cells.", flush=True)
    if not args.execute:
        return

    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir / "run"
    work_dir.mkdir(exist_ok=True)
    for cell in code_cells:
        if args.cpu:
            cell.source = cell.source.replace('DEVICE_PREFERENCE = "auto"', 'DEVICE_PREFERENCE = "cpu"')
        if args.full:
            cell.source = cell.source.replace("QUICK_RUN = True", "QUICK_RUN = False")

    # Prefer the kernel installed in this Python environment, not a global kernel.
    kernel_specs = KernelSpecManager(kernel_dirs=[str(Path(sys.prefix) / "share/jupyter/kernels")])
    manager = KernelManager(kernel_name="python3", kernel_spec_manager=kernel_specs)
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    client = NotebookClient(notebook, km=manager, timeout=1800, allow_errors=False,
                            resources={"metadata": {"path": str(work_dir)}})
    try:
        client.execute(cleanup_kc=True)
    finally:
        nbformat.write(notebook, output_dir / "tutorial_03_executed.ipynb")

    for cell_index, cell in enumerate(notebook.cells):
        for output_index, output in enumerate(cell.get("outputs", [])):
            png = output.get("data", {}).get("image/png")
            if png:
                (output_dir / f"cell-{cell_index:02d}-{output_index}.png").write_bytes(base64.b64decode(png))

    metrics_path = work_dir / "outputs/tutorial_03/metrics.json"
    metrics = json.loads(metrics_path.read_text())
    regression = metrics["regression"]
    selected = regression["selected_by_validation"]
    baseline = regression["test"]["Training-mean baseline"]["RMSE (MPG)"]
    assert regression["test"][selected]["RMSE (MPG)"] < baseline, "Regressor did not beat baseline."
    assert metrics["classification"]["test_accuracy"] > 0.70, "Classifier failed the learning sanity check."
    if args.cpu:
        assert metrics["environment"]["device"] == "cpu"
    expected_split = [54000, 6000, 10000] if args.full else [6000, 1000, 2000]
    assert metrics["classification"]["split_sizes"] == expected_split
    assert regression["split_sizes"] == [274, 58, 60]
    assert all(cell.execution_count is not None for cell in notebook.cells if cell.cell_type == "code")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"PASS: full sequential execution, learning checks, and checkpoint round trips. Artifacts: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
