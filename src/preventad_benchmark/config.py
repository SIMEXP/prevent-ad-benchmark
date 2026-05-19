"""Centralized configuration for PreventAD Benchmark.

This module consolidates hardcoded paths, model configurations, and constants
that were previously duplicated across multiple files.
"""

from pathlib import Path


# =============================================================================
# Directory structure
# =============================================================================

PATHS = {
    "source": Path(
        "data/source/dataset-preventad_version-8.1internal_pipeline-gigapreprocess2"
    ),
    "source_fmriprep": Path(
        "data/source/dataset-preventad_version-8.1internal_pipeline-gigapreprocess2"
        "/mri/wave1/fmriprep-20.2.8lts"
    ),
    "external": Path("data/external"),
    "interim": Path("data/interim"),
    "processed": Path("data/processed"),
    "models": Path("models"),
    "outputs": Path("outputs"),
    "resource": Path("resource"),
}

# Example subject for atlas resampling (MNI space reference)
EXAMPLE_SUBJECT_MNI = (
    PATHS["source_fmriprep"]
    / "sub-MTL0001/ses-BL00A/func"
    / "sub-MTL0001_ses-BL00A_task-enc_space-MNI152NLin2009cAsym_boldref.nii.gz"
)

# =============================================================================
# Phenotype files
# =============================================================================

PHENOTYPE_SEXAGE = (
    PATHS["source"] / "dataset-preventad81internal_desc-sexage_pheno.tsv"
)

PHENOTYPE_TARGETS = (
    PATHS["source"] / "dataset-preventad81internal_desc-subjlvltargets_pheno.tsv"
)

# =============================================================================
# Denoising strategy (shared by gigaconnectome and prepare pipelines)
# =============================================================================

DENOISE_STRATEGY_NAME = "simple+gsr"
DENOISE_STRATEGY = {
    "denoise_strategy": "simple",
    "motion": "basic",
    "global_signal": "basic",
}

TIMESERIES_LENGTH = 140

# =============================================================================
# QC filtering thresholds (shared by gigaconnectome, brainharmonix, brainlm)
# =============================================================================

QC_FILTERS = {
    "cerebellum_coverage_min": 0.75,
    "proportion_kept_min": 0.5,
    "session": "BL00",
    "run": 1,
}

# =============================================================================
# BrainLM atlas configurations
# =============================================================================

A424_NPARCELS = 424

BRAINLM_DATASET_CONFIGS = {
    "preventad": {
        "atlas_file": Path("resource/preventad/resample_A424+2mm.nii.gz"),
        "seg_name": "a424",
        "ts_min_length": TIMESERIES_LENGTH,
        "phenotype": {
            "filepath": str(PHENOTYPE_SEXAGE),
            "index_col": "identifier",
            "convert_data": ["Sex", "Candidate_Age"],
        },
    },
}

BRAINLM_COORD_DATASET = Path("resource/brainlm/atlases/brainregion_coordinates.arrow")
BRAINLM_COORD_FILE = Path("resource/brainlm/atlases/A424_Coordinates.dat")

# =============================================================================
# BrainLM model configuration
# =============================================================================

BRAINLM_TIMESERIES_LENGTH = TIMESERIES_LENGTH

BRAINLM_MODEL_ARGUMENTS = {
    "mask_ratio": 0.75,
    "timepoint_patching_size": 20,
    "num_timepoints_per_voxel": BRAINLM_TIMESERIES_LENGTH,
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "output_attentions": True,
}

BRAINLM_IMAGE_COLUMNS = {
    "gigaconnectome": "raw_timeseries",
    "brainlm": "Subtract_Mean_Divide_Global_STD_Normalized_Recording",
}

# =============================================================================
# BrainHarmonix model configuration
# =============================================================================

BRAINHARMONIX_CHECKPOINTS = {
    "harmonizer": Path("models/brain-harmonix/harmonizer/model.pth"),
    "fmri_encoder": Path("models/brain-harmonix/harmonix-f/model.pth"),
    "t1_encoder": Path("models/brain-harmonix/harmonix-s/model.pth"),
}

BRAINHARMONIX_POS_EMBED_PATHS = {
    "gradient": "BrainHarmony/brainharmony_pos_embed/gradient_mapping_400.csv",
    "geo_harm": "BrainHarmony/brainharmony_pos_embed/schaefer400_roi_eigenmodes.csv",
}

BRAINHARMONIX_SEG_NAME = "schaefer400"
BRAINHARMONIX_SCHAEFER_ROIS = 400
BRAINHARMONIX_T1_TARGET_SHAPE = (160, 192, 160)
BRAINHARMONIX_NUM_LATENT_TOKENS = 128
BRAINHARMONIX_TIMESERIES_LENGTH = TIMESERIES_LENGTH

# BrainHarmonix temporal resampling parameters
BRAINHARMONIX_PREVENTAD_TR = 2.26
BRAINHARMONIX_STANDARD_TIME = 48 * 0.735  # ~35.28s, from BrainHarmonix paper
BRAINHARMONIX_TARGET_NUM_PATCHES = 18

# =============================================================================
# Downstream evaluation configuration
# =============================================================================

EVALUATION_N_SPLITS = 20
EVALUATION_PCA_COMPONENTS = 75
EVALUATION_TARGETS = ["sex", "age", "splifhalfage", "progess2mci", "abSUVRbin", "centiloidbin", "centiloid", "abSUVR"]
#  "centiloid", "abSUVR", "sex",