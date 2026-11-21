from __future__ import annotations


import typer

from trauma_triage_vlm.config import load_config
from trauma_triage_vlm.data.volumes import synthetic_ct
from trauma_triage_vlm.pipeline import run_visual_pipeline
from trauma_triage_vlm.types import ClinicalContext

app = typer.Typer(add_completion=False)


@app.command()
def run(config: str = "configs/main.yaml", seed: int = 7) -> None:
    cfg = load_config(config)
    volume = synthetic_ct(seed)
    context = ClinicalContext(chief_complaint="trauma abdominal pain", mechanism="crash", heart_rate=112, systolic_bp=104, gcs=14, pain=8, resources_expected=3)
    report = run_visual_pipeline(volume, context, cfg)
    typer.echo(f"ESI {report.decision.esi} ISS {report.iss} high_acuity {report.decision.high_acuity}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
