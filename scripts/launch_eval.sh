#!/usr/bin/env bash
set -euo pipefail
python -m trauma_triage_vlm.cli.evaluate --config configs/main.yaml --split ratic
python -m trauma_triage_vlm.cli.evaluate --config configs/main.yaml --split cq500
python -m trauma_triage_vlm.cli.evaluate --config configs/main.yaml --split mimic
