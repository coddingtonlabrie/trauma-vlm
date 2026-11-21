from __future__ import annotations


from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class BackboneConfig:
    name: str
    precision: str
    temperature: float
    max_react_iterations: int
    edge_hardware: str


@dataclass(frozen=True)
class ImagingConfig:
    target_slices_min: int
    target_slices_max: int
    min_component_area: int
    strategy: str = "landmark"


@dataclass(frozen=True)
class EvaluationConfig:
    seeds: tuple[int, ...]
    bootstrap_iterations: int
    high_acuity_sensitivity_floor: float


@dataclass(frozen=True)
class ProjectConfig:
    seed: int
    backbone: BackboneConfig
    imaging: ImagingConfig
    evaluation: EvaluationConfig
    outputs_directory: Path


def read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("configuration root must be a mapping")
    return data


def load_config(path: str | Path) -> ProjectConfig:
    raw = read_yaml(Path(path))
    backbone_raw = raw["backbone"]
    imaging_raw = raw["imaging"]
    evaluation_raw = raw["evaluation"]
    outputs_raw = raw.get("outputs", {})
    return ProjectConfig(
        seed=int(raw["seed"]),
        backbone=BackboneConfig(
            name=str(backbone_raw["name"]),
            precision=str(backbone_raw["precision"]),
            temperature=float(backbone_raw["temperature"]),
            max_react_iterations=int(backbone_raw["max_react_iterations"]),
            edge_hardware=str(backbone_raw["edge_hardware"]),
        ),
        imaging=ImagingConfig(
            target_slices_min=int(imaging_raw["target_slices_min"]),
            target_slices_max=int(imaging_raw["target_slices_max"]),
            min_component_area=int(imaging_raw["min_component_area"]),
            strategy=str(imaging_raw.get("strategy", "landmark")),
        ),
        evaluation=EvaluationConfig(
            seeds=tuple(int(x) for x in evaluation_raw["seeds"]),
            bootstrap_iterations=int(evaluation_raw["bootstrap_iterations"]),
            high_acuity_sensitivity_floor=float(evaluation_raw["high_acuity_sensitivity_floor"]),
        ),
        outputs_directory=Path(str(outputs_raw.get("directory", "outputs"))),
    )
