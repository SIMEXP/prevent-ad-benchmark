# PreventAD Foundation Model Benchmark

Benchmarking neuroimaging foundation models (BrainLM, BrainHarmonix) on the PREVENT-AD dataset for downstream prediction of clinical and biological targets.

## Overview

This project evaluates whether pretrained foundation models for fMRI produce useful representations for predicting clinically relevant outcomes. Two models are compared against classical baselines across 8 prediction targets: sex, age, split-half age, MCI progression, centiloid, centiloid (binarized), amyloid SUVR, and amyloid SUVR (binarized).

Each model is tested in two transfer modes:
- **Feature extraction**: extract embeddings from frozen pretrained weights and train downstream classifiers
- **Fine-tuned**: fine-tune embedding layers on the training set, then run prediction pipeline

## Analysis Pipeline

```
Data Preparation  :arrow_right:  Evaluation  :arrow_right:  Reports
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
| `baseline.run` | Evaluate raw timeseries (PCA->75) + functional connectivity with SVM & Linear models |

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
Arrow Dataset :arrow_right: Feature Extraction :arrow_right: Classifiers :arrow_right: Scores :arrow_right: Summary Tables
          (Timeseries / FC / embeddings)  (SVM, Linear)   (per split)  (mean + 95% CI)
```

**Prediction targets**: 8 targets spanning demographics, cognition, and amyloid pathology:

| Type | Targets |
|------|---------|
| Classification | sex, split-half age, MCI progression, centiloid > 20, amyloid SUVR > 1.26 |
| Regression | age, centiloid, amyloid SUVR |

Classification vs regression is auto-detected from label type (string :arrow_right: classification, numeric :arrow_right: regression).

**Baseline evaluation**: classical features extracted directly from the Arrow dataset:
- *Timeseries*: flattened ROI timeseries reduced to 75 PCA components
- *Functional connectivity*: correlation-based connectivity vectors (vectorized, diagonal discarded)
- Evaluated with 20-fold stratified shuffle cross-validation using SVM, Linear, and Dummy classifiers

**Foundation model evaluation**: learned embeddings from pretrained or fine-tuned models:
- *BrainLM*: CLS Token, CLS Embedding, Mean Embedding, Max Embedding
- *BrainHarmonix*: fMRI (mean), T1 (mean), Harmonizer (CLS), Harmonizer (latent)
- Evaluated on a precomputed train/test split (one per fold) using SVM and Linear classifiers

**Classifiers:**
- **SVM**: `SVC` (balanced class weights) / `SVR`, with `RobustScaler`
- **Linear**: `LogisticRegression` / `LinearRegression`, with `RobustScaler`
- **Dummy**: `most_frequent` / `mean` strategy as chance-level reference (baseline only)

**Metrics:**

| Classification | Regression |
|----------------|------------|
| Accuracy, AUC, F1 | RMSE, MAE, R² |

Results are reported as mean with 95% confidence interval (2.5th–97.5th percentile) across splits.

### Stage 3: Reports (`inv reports.*`)

| Task | Description |
|------|-------------|
| `reports.generate-summary` | Aggregate results across splits into mean +/- 95% CI tables |

## Project Structure

```
├── src/preventad_benchmark/     # Main Python package
│   ├── config.py                # Centralized paths, model configs, constants
│   ├── cli/                     # CLI entry points (extract, finetune, evaluate)
│   ├── dataset/                 # Data loading, phenotype, train/test splits
│   ├── evaluation/              # Downstream pipelines (SVM, linear), targets
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

The BrainLM submodule is for record keeping. To pull it:

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
uv run inv prepare.fmri
uv run inv prepare.timeseries
uv run inv prepare.split

# 2. Run evaluation (interactive, single split)
uv run inv baseline.run
uv run inv brainlm.evaluate
uv run inv brainharmonix.evaluate

# 2b. Or submit SLURM job arrays for all splits
uv run inv brainlm.submit-evaluate
uv run inv brainharmonix.submit-evaluate

# 3. Generate summary tables
uv run inv reports.generate-summary
```

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

SLURM resource settings are in `slurm_config.yaml`:

```yaml
defaults:
  account: rrg-pbellec
  time: "0:10:00"
  mem: "8G"
  cpus_per_task: 4
  gres: "gpu:1"

overrides:
  finetune_brainharmonix:
    time: "4:00:00"
    mem: "32G"
  finetune_brainlm:
    time: "24:00:00"
    mem: "64G"
```

Submit tasks generate SLURM scripts in `scripts/` and submit job arrays (one job per split).
