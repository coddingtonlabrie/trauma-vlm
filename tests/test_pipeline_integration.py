from trauma_triage_vlm.config import load_config
from trauma_triage_vlm.data.volumes import synthetic_ct
from trauma_triage_vlm.pipeline import run_visual_pipeline
from trauma_triage_vlm.types import ClinicalContext


def test_pipeline_report_shape() -> None:
    cfg = load_config("configs/main.yaml")
    volume = synthetic_ct(3, slices=64, height=48, width=48)
    context = ClinicalContext("trauma", "fall", heart_rate=105, systolic_bp=110, pain=6, resources_expected=2)
    report = run_visual_pipeline(volume, context, cfg)
    assert 1 <= report.decision.esi <= 5
    assert report.iss >= 0
    assert len(report.findings) == 5
