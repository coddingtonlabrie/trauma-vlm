from __future__ import annotations


from pathlib import Path

import torch
import typer

from trauma_triage_vlm.config import load_config
from trauma_triage_vlm.training.calibrate import fit_calibration
from trauma_triage_vlm.training.checkpoint import atomic_save
from trauma_triage_vlm.utils.seed import set_seed

app = typer.Typer(add_completion=False)


@app.command()
def run(config: str = "configs/main.yaml") -> None:
    cfg = load_config(config)
    set_seed(cfg.seed)
    features = torch.randn(128, 16)
    labels = torch.randint(0, 5, (128,))
    model, result = fit_calibration(features, labels, epochs=2, batch_size=32, lr=1e-4)
    atomic_save({"seed": cfg.seed, "state_dict": model.state_dict(), "losses": result.losses}, Path(cfg.outputs_directory) / "calibration.pt")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
