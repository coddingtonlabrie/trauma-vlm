# An Agentic Vision-Language Model with Zero-Shot Transfer for Automated Multi-Organ Trauma Triage in Resource-Limited Emergency Departments

This repository contains the lab implementation of TraumaTriage-VLM: an ATLS-grounded vision-language triage pipeline for abdominal and head trauma assessment. The code expresses the four-agent pathway, deterministic ISS aggregation, CT windowing and slice selection, QA checks, metric computation, and command-line evaluation tools used for the public benchmark and silent-mode deployment analyses.

## Installation

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

```bash
conda env create -f environment.yml
conda activate trauma-triage-vlm
pip install -e .
```

```bash
docker build -t trauma-triage-vlm .
```

## Data

Verified dataset links are listed in `dataset_links.txt`. The public MIMIC-IV-ED 2.2 page was reachable during preparation. The RATIC and CQ500 links cited in the manuscript did not return an accessible HTTP success response during verification, so they are not listed as usable sources here. Private deployment cohort records are not included.

Expected local layout:

```text
data/
  mimic_iv_ed_2_2/
    edstays.csv.gz
    triage.csv.gz
    diagnosis.csv.gz
  ratic/
    images/
    labels.csv
  cq500/
    scans/
    labels.csv
```

## Training

The main pathway is zero-shot. The train command fits only local calibration layers and report thresholds:

```bash
python -m trauma_triage_vlm.cli.train --config configs/main.yaml
```

Main settings use Qwen3-VL-8B, greedy decoding, five seeds, 1000 bootstrap iterations, and INT4 edge deployment reporting for the Jetson AGX Orin configuration. No task-specific image labels are used for model fitting.

## Evaluation

```bash
python -m trauma_triage_vlm.cli.evaluate --config configs/main.yaml --split ratic
python -m trauma_triage_vlm.cli.evaluate --config configs/main.yaml --split cq500
python -m trauma_triage_vlm.cli.evaluate --config configs/main.yaml --split mimic
```

Reported targets:

RATIC mean organ AUC: 0.814 ± 0.031 over five seeds.
CQ500 intracranial hemorrhage AUC: 0.879 ± 0.028 over five seeds.
MIMIC-IV-ED ESI exact match: 76.5 ± 2.4 over five seeds.
Prospective high-acuity AUC for 24-hour critical intervention: 0.86.
Prospective high-acuity AUC for 30-day mortality: 0.83.

## Compute Budget

Cloud reference: Qwen3-VL-8B FP16 on A100, 15.4 GB VRAM, 8.3 seconds per case.
Edge reference: Qwen3-VL-8B INT4 AWQ on Jetson AGX Orin, 5.1 GB VRAM, 38.2 seconds per case.
The paper reports per-case inference latency rather than epoch-based training time.
