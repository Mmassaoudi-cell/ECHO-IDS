# ECHO-IDS Reproducibility Package

This folder contains the executable research code for:

**ECHO-IDS: Evidence-Calibrated Hallucination and Intrusion Detection for
LLM-Assisted Cyber-Physical Security**

It packages the proposed method and its scientific evaluation pipeline without
the manuscript, LaTeX tables, or graph-generation code.

## Included

- Leakage-safe preprocessing for Edge-IIoTset, RT-IoT2022, and UNSW-NB15.
- Fixed split auditing and source-row overlap checks.
- Logistic regression, random forest, XGBoost, MLP, CNN, LSTM, CNN-LSTM,
  CNN-BiLSTM, TCN, Transformer, graph-attention, and ECHO-IDS implementations.
- The variance-gated ECHO intrusion encoder and validation-trained hybrid.
- Structured evidence extraction from IDS inputs and predictions.
- Deterministic faithful and controlled-corruption cyber report generation.
- EG-SINdex semantic clustering, evidence mismatch, unsupported-claim checks,
  calibration, and field-level explanations.
- Hallucination baselines, HaluEval evaluation, tuning, ablations, sensitivity,
  runtime, and joint-pipeline evaluation.
- Compact run-level and summary CSV files from the reported experiments.

Large or externally licensed artifacts are intentionally excluded: raw
datasets, generated report corpora, pretrained model caches, predictions, and
checkpoints. They are regenerated locally.

## Structure

```text
Data/                   Dataset placement instructions and compact metadata
preprocessing/          Dataset cleaning, encoding, scaling, and split audits
models/                 IDS architectures and EG-SINdex
training/               Training, calibration, inference, and joint evaluation
evidence_builder/       Structured IDS evidence construction
llm_reports/            Faithful/corrupted reports and local LLM audit
baselines/              NLI and HaluEval baselines
tuning/                 Validation-only Optuna studies
ablations/              IDS and hallucination ablations and sensitivity
results/                Compact validated run-level and aggregate results
tests/                  Reproducibility and scientific-integrity checks
```

No `figures/`, `tables/`, or manuscript-generation module is included.

## Installation

Python 3.11 or newer is recommended. CUDA is optional but strongly recommended
for the complete deep-learning and embedding experiments.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorch CUDA wheels can be installed using the command published for the local
CUDA version at <https://pytorch.org/get-started/locally/>.

## Datasets

Place the raw datasets under `Data/` exactly as documented in
[`Data/README.md`](Data/README.md). The preprocessing code does not download
restricted or externally licensed datasets.

HaluEval can be obtained with:

```powershell
git clone https://github.com/RUCAIBox/HaluEval.git Data/HaluEval
```

## Full Reproduction

From this folder:

```powershell
.\reproduce.ps1
```

The script performs preprocessing, split auditing, three-seed binary and
multiclass IDS evaluation, hybrid calibration, evidence extraction, controlled
report generation, EG-SINdex evaluation and tuning, ablations, sensitivity,
and joint runtime analysis. It does not produce tables or graphs.

Useful switches:

```powershell
.\reproduce.ps1 -SkipPreprocessing
.\reproduce.ps1 -SkipIDS
.\reproduce.ps1 -IncludeExternalBaselines
.\reproduce.ps1 -IncludeLocalLLM
```

External NLI/HaluEval and local instruction-model stages are opt-in because
they download model weights and substantially increase runtime.

## Individual Stages

```powershell
# Preprocessing and leakage audit
python preprocessing/preprocess_ids.py --output Data/processed
python preprocessing/audit_splits.py --root Data/processed

# IDS experiments, three fixed seeds
python training/train_ids.py --tasks binary --seeds 17 29 43
python training/evaluate_hybrid_echo.py --task binary --seeds 17 29 43
python training/train_ids.py --tasks multiclass --seeds 17 29 43

# Validation-only tuning and IDS ablation
python tuning/tune_echo_ids.py --dataset unsw_nb15 --trials 15 --epochs 20
python ablations/run_ids_ablation.py --seeds 17 29 43 --epochs 20

# Evidence and cyber hallucination corpus
python evidence_builder/build_evidence.py --limit 1000
python llm_reports/generate_cyber_reports.py --generations 6 --seed 2026

# EG-SINdex, tuning, ablations, and sensitivity
python training/evaluate_hallucination.py --generations 6 --seeds 17 29 43
python tuning/tune_eg_sindex.py
python ablations/evaluate_score_ablations.py
python ablations/run_sensitivity.py --limit 600

# Optional external baselines
python baselines/evaluate_halueval.py --limit-per-task 2000
python baselines/evaluate_cyber_nli.py

# Joint pipeline and runtime
python training/summarize_joint_runtime.py
```

## Scientific Controls

- Raw rows are split before any fitted transformation.
- Encoders, imputers, scalers, class weights, calibration temperatures,
  thresholds, stackers, and Optuna objectives do not use test data.
- UNSW-NB15 retains its official test partition.
- Edge-IIoTset and RT-IoT2022 use fixed stratified 70/15/15 partitions.
- Oversampling is not performed before splitting.
- Seeds 17, 29, and 43 are reported; no best-seed selection is used.
- Controlled report labels are kept distinct from HaluEval labels and the
  unlabeled local-model audit.
- Negative ablation and multiclass results remain in `results/`.

## Verification

The included tests operate on compact metadata and run-level results, so they
can run before downloading the datasets:

```powershell
python -m compileall -q preprocessing models training evidence_builder `
  llm_reports baselines tuning ablations tests
python -m unittest discover -s tests -v
```

After preprocessing, rerun `preprocessing/audit_splits.py` to verify the actual
generated arrays and export their split hashes.

## Outputs

Generated artifacts are written to:

- `Data/processed/`
- `checkpoints/ids/`
- `evidence_builder/generated/`
- `llm_reports/*.jsonl`
- `results/`
- `tuning/`
- `logs/`

These large outputs are ignored by Git, while compact metadata and reported
CSV summaries remain versionable.
