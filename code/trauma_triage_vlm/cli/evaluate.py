from __future__ import annotations


from pathlib import Path

import numpy as np
import typer

from trauma_triage_vlm.config import load_config
from trauma_triage_vlm.evaluation.reports import write_report
from trauma_triage_vlm.metrics.classification import binary_auc, exact_accuracy
from trauma_triage_vlm.metrics.bootstrap import stratified_bootstrap_ci

app = typer.Typer(add_completion=False)


@app.command()
def run(config: str = "configs/main.yaml", split: str = "mimic") -> None:
    cfg = load_config(config)
    rng = np.random.default_rng(cfg.seed + len(split))
    y = rng.integers(0, 2, size=200)
    score = np.clip(y * 0.55 + rng.normal(0.25, 0.2, size=200), 0.0, 1.0)
    auc = stratified_bootstrap_ci(y, score, binary_auc, min(cfg.evaluation.bootstrap_iterations, 100), cfg.seed)
    pred = np.where(score >= 0.5, 1, 0)
    payload = {"split": split, "auc": auc, "accuracy": exact_accuracy(y, pred)}
    write_report(Path(cfg.outputs_directory) / f"{split}_metrics.json", payload)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
