# PreventAD Foundation Model Benchmark

Benchmarking neuroimaging foundation models (BrainLM, BrainHarmonix) on the PREVENT-AD dataset for downstream prediction of clinical and biological targets.

The Prevent-AD dataset requires a data usage agreement; hence, no subject-level data is shared.
- For fMRI data processing code, please see [SIMEXP/prevent-ad_dr8.1internal](https://github.com/SIMEXP/prevent-ad_dr8.1internal/)
- For better plotting, please see [SIMEXP/prevent-ad-benchmark-plotting](https://github.com/SIMEXP/prevent-ad-benchmark-plotting)
- Experiment results for generating the paper figures: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22926042.svg)](https://doi.org/10.5281/zenodo.22926042)

## Overview

This project evaluates whether pretrained foundation models for fMRI produce useful representations for predicting clinically relevant outcomes. Two models are compared against classical baselines across 8 prediction targets: sex, age, split-half age, MCI progression, centiloid, centiloid (binarized), amyloid SUVR, and amyloid SUVR (binarized).

Each model is tested in two transfer modes:
- **Feature extraction**: extract embeddings from frozen pretrained weights and train downstream classifiers
- **Fine-tuned**: fine-tune embedding layers on the training set, then run prediction pipeline

## Analysis Pipeline
```
Data Preparation  ->  Evaluation  ->  Reports
   (prepare)        (baseline/brainlm/brainharmonix)   (reports)
```

### Stage 1: Data Preparation (`inv prepare.*`)

| Task | Description |
|------|-------------|
| `prepare.models` | Download BrainLM pretrained weights from HuggingFace |
| `prepare.atlas` | Prepare atlases (A424 for BrainLM, Schaefer 400 for BrainHarmonix) |
| `prepare.t1` | Skull-strip T1 images and convert to tensors |
| `prepare.fmri` | Denoise fMRI (simple+GSR), optional z-scoring |
| `prepare.timeseries` | Extract timeseries and save as Arrow datasets |
| `prepare.split` | Create 20 stratified train/test splits (sex + MCI progression) |

### Stage 2: Evaluation (`inv baseline.*` / `inv brainlm.*` / `inv brainharmonix.*`)

**Baseline** (`inv baseline.*`): classical features evaluated with 20-fold CV

| Task | Description |
|------|-------------|
| `baseline.run` | Evaluate raw timeseries (PCA->75) + functional connectivity with linear models, plus a dummy chance-level reference |

**BrainLM** (`inv brainlm.*`): 4 preprocessing variants (2 atlases x 2 z-score settings) x 2 model sizes (111M, 650M) x 20 splits

| Task | Description |
|------|-------------|
| `brainlm.evaluate` | Run pretrained BrainLM prediction pipeline (no fine-tuning) |
| `brainlm.finetune` | Fine-tune embedding layers and run prediction pipeline |
| `brainlm.submit-evaluate` | Submit SLURM job array for pretrained prediction |
| `brainlm.submit-finetune` | Submit SLURM job array for fine-tuning + prediction |

**BrainHarmonix** (`inv brainharmonix.*`): 2 preprocessing variants (zscore / nozscore) x 20 splits

| Task | Description |
|------|-------------|
| `brainharmonix.evaluate` | Run pretrained BrainHarmonix prediction pipeline |
| `brainharmonix.finetune` | Fine-tune harmonizer and run prediction pipeline |
| `brainharmonix.submit-evaluate` | Submit SLURM job array for pretrained prediction |
| `brainharmonix.submit-finetune` | Submit SLURM job array for fine-tuning + prediction |

### Evaluation Design

```
Arrow Dataset -> Feature Extraction -> Classifiers -> Scores -> Summary Tables
          (Timeseries / FC / embeddings)  (Linear)        (per split)  (mean + 95% CI)
```

**Prediction targets**: 8 targets spanning demographics, cognition, and amyloid pathology:

| Type | Targets |
|------|---------|
| Classification | sex, split-half age, MCI progression, centiloid > 20, amyloid SUVR > 1.26 |
| Regression | age, centiloid, amyloid SUVR |

Classification vs regression is auto-detected from label type (string -> classification, numeric -> regression).

**Baseline evaluation**: classical features extracted directly from the Arrow dataset:
- *Timeseries*: flattened ROI timeseries reduced to 75 PCA components
- *Functional connectivity*: correlation-based connectivity vectors (vectorized, diagonal discarded)
- Evaluated with 20-fold stratified shuffle cross-validation using Linear and Dummy classifiers

**Foundation model evaluation**: learned embeddings from pretrained or fine-tuned models:
- *BrainLM*: CLS Token, CLS Embedding, Mean Embedding, Max Embedding
- *BrainHarmonix*: fMRI (mean), T1 (mean), Harmonizer (CLS), Harmonizer (latent)
- Evaluated on a precomputed train/test split (one per fold) using a Linear classifier

**Classifiers:**
- **Linear**: `LogisticRegression` / `LinearRegression`, with `RobustScaler`
- **Dummy**: `most_frequent` / `mean` strategy as chance-level reference (baseline only)

**Metrics:**

| Classification | Regression |
|----------------|------------|
| Accuracy, AUC, F1, Precision | RMSE, MAE, R² |

Results are reported as mean with 95% confidence interval (2.5th–97.5th percentile) across splits, each as its own column (`METRIC`, `METRIC_CI_LOW`, `METRIC_CI_HIGH`).

For classification, accuracy and precision are additionally tested against the Schaefer400 functional-connectivity baseline and the dummy-classifier baseline with a one-sided Welch's t-test (`METRIC_T_VS_FC`/`METRIC_DF_VS_FC`/`METRIC_P_VS_FC`/`METRIC_SIG_VS_FC`, and the same four for `DUMMY`; `T` = Welch's t statistic, `DF` = Welch–Satterthwaite degrees of freedom, `SIG` = p < 0.05), testing whether a model's performance significantly exceeds each baseline. The test is unpaired: baseline results come from independent cross-validation folds, while foundation-model results come from the fixed train/test splits, so the two aren't matched samples.

### Stage 3: Reports (`inv reports.*`)

| Task | Description |
|------|-------------|
| `reports.generate-summary` | Aggregate results across splits into mean +/- 95% CI tables, with t-tests against baselines |
| `reports.plot-learning-curves` | Plot fine-tuning train/val loss curves (per split, per condition, and combined) |
| `reports.plot-classification` | Per-target bar charts of classification accuracy/precision, ranked against baselines |

## Project Structure

```
├── src/preventad_benchmark/     # Main Python package
│   ├── config.py                # Centralized paths, model configs, constants
│   ├── cli/                     # CLI entry points (extract, finetune, evaluate)
│   ├── dataset/                 # Data loading, phenotype, train/test splits
│   ├── evaluation/              # Downstream pipelines (linear, dummy), targets
│   ├── models/                  # BrainLM and BrainHarmonix model code
│   └── plotting/                # Visualization utilities
├── tasks/                       # Invoke task definitions
│   ├── prepare.py               # Data preparation tasks
│   ├── brainlm.py               # BrainLM extraction/finetuning tasks
│   ├── brainharmonix.py         # BrainHarmonix extraction/finetuning tasks
│   ├── baseline.py              # Downstream evaluation tasks
│   ├── reports.py               # Result aggregation tasks
│   └── slurm.py                 # SLURM job submission helpers
├── data/
│   ├── source/                  # Raw PREVENT-AD dataset (not tracked)
│   ├── interim/                 # Intermediate processing outputs
│   ├── processed/               # Arrow datasets, train/test splits
│   └── external/                # External resources
├── models/                      # Pretrained model weights
├── outputs/                     # Extraction and fine-tuning outputs
├── resource/                    # Atlases, coordinates
├── scripts/                     # Generated SLURM submission scripts
└── slurm_config.yaml            # SLURM resource defaults and overrides
```

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone git@github.com:SIMEXP/prevent-ad-benchmark.git
cd prevent-ad-benchmark
```

### Pulling submodules (optional)

The BrainLM and BrainHarmonix submodules are for record keeping. 
The installation (when needed) is managed by the general project setup.

To pull it:

```bash
git submodule update --init --recursive
```

### Create virtual environment

On Rorqual, load required modules first:

```bash
module add cudacore/.12.6.2
module load httpproxy
```

Install with uv:

```bash
uv venv
uv sync --extra build
```

`flash-attn` compiles from source against a real CUDA toolkit (`CUDA_HOME`) -- this fails on a CPU-only machine or a plain login node. Run `uv sync` on a GPU node (e.g. via `salloc` on Fir) if you hit a `CUDA_HOME environment variable is not set` error.

### Post-install fix for brainharmonix

The `brainharmonix` package includes an internal `datasets` module that conflicts with the HuggingFace `datasets` library. After running `uv sync`, remove it:

```bash
rm -rf .venv/lib/python3.12/site-packages/brainharmonix/datasets
```

This needs to be re-run after any `uv sync` that reinstalls brainharmonix.

## Usage

Run tasks with `uv run inv <namespace>.<task>`:

```bash
# 1. Prepare data
uv run inv prepare.models
uv run inv prepare.atlas
uv run inv prepare.t1
uv run inv prepare.fmri -z
uv run inv prepare.fmri --no-zscore
uv run inv prepare.timeseries -w gigaconnectome -a schaefer400 -f data/interim/dataset-preventad.fmri.zscored
uv run inv prepare.timeseries -w gigaconnectome -a schaefer400 -f data/interim/dataset-preventad.fmri.NoZscore
uv run inv prepare.timeseries -w gigaconnectome -a a424 -f data/interim/dataset-preventad.fmri.zscored
uv run inv prepare.timeseries -w brainlm -a a424 -f data/interim/dataset-preventad.fmri.zscored
uv run inv prepare.timeseries -w gigaconnectome -a a424 -f data/interim/dataset-preventad.fmri.NoZscore
uv run inv prepare.timeseries -w brainlm -a a424 -f data/interim/dataset-preventad.fmri.NoZscore
uv run inv prepare.split

# 2. Run evaluation (feature extraction from frozen pretrained weights)
uv run inv baseline.run  # this will create the chance level results and the FC baseline
uv run inv brainlm.evaluate
uv run inv brainharmonix.evaluate

# 2b. Or submit SLURM job arrays for all splits
uv run inv brainlm.submit-evaluate
uv run inv brainharmonix.submit-evaluate

# 2c. Fine-tune (single split, interactive -- for debugging before a full submission)
uv run inv brainlm.finetune --preprocessing=brainlm --model-params=650M --split-index=0
uv run inv brainharmonix.finetune --split-index=0

# 2d. Or submit SLURM job arrays for fine-tuning + downstream prediction across all splits
uv run inv brainlm.submit-finetune --model-size=650M --preprocessing=all --n-splits=20 --rerun-finetune=True
uv run inv brainharmonix.submit-finetune --n-splits=20 --rerun-finetune=True

# 3. Generate preliminary summary tables and figures
uv run inv reports.generate-summary --experiment all
uv run inv reports.plot-learning-curves
uv run inv reports.plot-classification
```

Fine-tuning uses early stopping by default (`--patience=5` epochs of no val-loss improvement, up to `--epochs=50`); BrainLM also takes `--lr` (default `1e-4`). `--rerun-finetune` defaults to `False` on the `submit-finetune` tasks, which only reruns the downstream-prediction step against an *existing* fine-tuned checkpoint -- pass `--rerun-finetune=True` explicitly to actually fine-tune.

Use `--dry-run` on submit tasks to preview SLURM scripts without submitting.

Check `uv run inv --list` for all available commands and their documentation.

## CLI Entry Points

| Command | Description |
|---------|-------------|
| `preventad-extract-brainlm` | Extract BrainLM embeddings from Arrow dataset |
| `preventad-extract-brainharmonix` | Extract BrainHarmonix embeddings from Arrow dataset |
| `preventad-finetune-brainlm` | Fine-tune BrainLM ViT-MAE on training split |
| `preventad-finetune-brainharmonix` | Fine-tune BrainHarmonix harmonizer |


## Configuration

Key settings in `src/preventad_benchmark/config.py`:

| Constant | Value | Description |
|----------|-------|-------------|
| `TIMESERIES_LENGTH` | 140 | fMRI window length (timepoints) |
| `EVALUATION_N_SPLITS` | 20 | Number of train/test splits |
| `EVALUATION_PCA_COMPONENTS` | 75 | PCA dimensions for baseline features |
| `EVALUATION_TARGETS` | 8 targets | sex, age, splifhalfage, progess2mci, centiloid, abSUVR, abSUVRbin, centiloidbin |
| `DENOISE_STRATEGY_NAME` | simple+gsr | fMRI denoising strategy |

## SLURM Submission

Currently configured for the [Fir](https://docs.alliancecan.ca/wiki/Fir) cluster. SLURM resource settings are in `slurm_config.yaml`:

```yaml
defaults:
  account: def-hwang1
  time: "1:00:00"  # Fir enforces a 1-hour minimum walltime
  mem: "24G"
  cpus_per_task: 4
  gpus_per_node: "nvidia_h100_80gb_hbm3_2g.20gb:1"  # a MIG slice; use "h100:1" for a full H100

overrides:
  finetune_brainharmonix:
    time: "4:00:00"
    mem: "16G"
  finetune_brainlm:
    time: "4:00:00"
    mem: "16G"
```

Note: Fir GPUs are requested with `--gpus-per-node=<type>:<count>`, not the older `--gres=gpu:<count>` syntax used on some other Alliance clusters. If migrating this config to a different cluster, check its docs for the current GPU-request syntax and walltime limits first.

Submit tasks generate SLURM scripts in `scripts/` and submit job arrays (one job per split).

## Disclaimer

This project uses AI-assisted tools (Claude Code) for software development only. No AI-generated content was used in data processing, analysis, or scientific results.
