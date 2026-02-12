"""Data preparation tasks for PreventAD benchmark.

This module contains invoke tasks for preprocessing fMRI and T1 data
using different pipelines (BrainLM, GigaConnectome, BrainHarmonix).
"""
import os
from pathlib import Path

import invoke
from nilearn.datasets import fetch_atlas_schaefer_2018

from preventad_benchmark.dataset.brainlm import convert_fMRIvols_to_A424, convert_to_brainlm_arrow_datasets
from preventad_benchmark.dataset.structural import preprocess_t1_images
from preventad_benchmark.dataset.functional import denoise_dataset, extract_timeseries_from_nifti
from preventad_benchmark.dataset.utils import convert_to_arrow_dataset, brain_region_coord_to_arrow, resample_atlas
from preventad_benchmark.config import TIMESERIES_LENGTH


# Default paths
DEFAULT_SOURCE_DIR = Path("data/source/dataset-preventad_version-8.1internal_pipeline-gigapreprocess2/mri/wave1/fmriprep-20.2.8lts")
DEFAULT_INTERIM_DIR = Path("data/interim")
DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_RESOURCE_DIR = Path("resource")


@invoke.task(
    help={
        "local-dir": "Directory to download models to (default: ./models/brainlm)",
    }
)
def models(c, local_dir="./models/brainlm"):
    """Download BrainLM pre-trained models from HuggingFace.

    Downloads the 111M and 650M parameter ViT-MAE models.

    Example:
        inv prepare.models
        inv prepare.models --local-dir ./my-models/brainlm
    """
    c.run("HF_HUB_ENABLE_HF_TRANSFER=1")
    c.run(f"huggingface-cli download vandijklab/brainlm --local-dir {local_dir}")


@invoke.task(
    help={
        "resource-dir": f"Resource directory for atlases (default: {DEFAULT_RESOURCE_DIR})",
    }
)
def atlas(c, resource_dir=None):
    """Prepare brain atlases (A424 and Schaefer 400).

    Resamples A424 atlases to PreventAD space and downloads Schaefer 400 atlas.

    Example:
        inv prepare.atlas
    """
    resource_dir = Path(resource_dir) if resource_dir else DEFAULT_RESOURCE_DIR

    c.run(f"mkdir -p {resource_dir}/preventad")
    resample_atlas(f"{resource_dir}/brainlm/atlases/A424+4mm.nii.gz", f"{resource_dir}/preventad")
    resample_atlas(f"{resource_dir}/brainlm/atlases/A424+2mm.nii.gz", f"{resource_dir}/preventad")
    brain_region_coord_to_arrow()

    print("Fetching Schaefer 400 atlas (17 networks, 2mm resolution)...")
    atlas_data = fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=17, resolution_mm=2)
    resample_atlas(atlas_data.maps, "/tmp")


@invoke.task(
    help={
        "source-dir": f"Source fMRIPrep directory (default: {DEFAULT_SOURCE_DIR})",
        "output-dir": "Output directory for T1 data",
    }
)
def t1(c, source_dir=None, output_dir=None):
    """Preprocess T1-weighted structural MRI images.

    Skull-strips T1 images and saves as PyTorch tensors for BrainHarmonix.

    Example:
        inv prepare.t1
        inv prepare.t1 --source-dir /path/to/fmriprep
    """
    source_dir = Path(source_dir) if source_dir else DEFAULT_SOURCE_DIR
    output_dir = Path(output_dir) if output_dir else DEFAULT_INTERIM_DIR / "dataset-preventad.t1"

    preprocess_t1_images(source_dir, output_dir)


@invoke.task(
    help={
        "source-dir": f"Source fMRIPrep directory (default: {DEFAULT_SOURCE_DIR})",
        "zscore": "Standardize per run with z score (default: True)",
        "output-dir": "Output Arrow dataset path",
    }
)
def fmri(c, source_dir=None, output_dir=None, zscore=True):
    """Denoise fMRI data wit or without z-scoring per run.

    Example:
        inv prepare.fmri
    """
    source_dir = Path(source_dir) if source_dir else DEFAULT_SOURCE_DIR
    if zscore:
        output_dir = Path(output_dir) if output_dir else DEFAULT_INTERIM_DIR / "dataset-preventad.fmri.zscored"
    else:
        output_dir = Path(output_dir) if output_dir else DEFAULT_INTERIM_DIR / "dataset-preventad.fmri.NoZscore"

    denoise_dataset(source_dir, output_dir, standarsization=zscore)


@invoke.task(
    help={
        "fmri-source-dir": "Source directory created from prepare.fmri",
        "t1-source-dir": "created from prepare.t1",
        "ts-output-dir": "Time series TSV.",
        "output-dir": "Output Arrow dataset path",
        "workflow": "gigaconnectome or brainlm; if using brainlm, the only atlas applied would be a424",
        "atlas": "schaefer400 or a424"
    }
)
def timeseries(c, fmri_source_dir=None, t1_source_dir=None, ts_output_dir=None, output_dir=None, workflow='gigaconnectome', atlas='schaefer400'):
    """Extract time series with a given atlas. Save as arrow dataset.
    The default option will generate the output for BrainHarmonix

    Example:
        inv prepare.timeseries
    """
    if workflow == 'gigaconnectome':
        fmri_source_dir = Path(fmri_source_dir) if fmri_source_dir else DEFAULT_INTERIM_DIR / "dataset-preventad.fmri.zscored"
        t1_source_dir = Path(t1_source_dir) if ts_output_dir else DEFAULT_INTERIM_DIR / "dataset-preventad.t1"
        ts_output_dir = Path(ts_output_dir) if ts_output_dir else DEFAULT_INTERIM_DIR / f"{fmri_source_dir.name}.{workflow}.{atlas}"
        output_dir = Path(output_dir) if output_dir else DEFAULT_PROCESSED_DIR / f"{fmri_source_dir.name}.{workflow}.{atlas}.arrow"

        extract_timeseries_from_nifti(fmri_source_dir, ts_output_dir, atlas)
        convert_to_arrow_dataset(t1_source_dir, ts_output_dir, output_dir, atlas)
    elif workflow == 'brainlm':
        if atlas != 'a424':
            print("Warning: BrainLM workflow only supports A424 atlas. Overriding atlas to a424.")
        atlas = 'a424'  # BrainLM only uses A424 atlas
        # Default should be the non z scored version
        fmri_source_dir = Path(fmri_source_dir) if fmri_source_dir else DEFAULT_INTERIM_DIR / "dataset-preventad.fmri.NoZscore"
        ts_output_dir = DEFAULT_INTERIM_DIR / f"{fmri_source_dir.name}.{workflow}.{atlas}"
        output_dir = DEFAULT_PROCESSED_DIR / f"{fmri_source_dir.name}.{workflow}.{atlas}.arrow"

        output_dir = Path(output_dir) if output_dir else DEFAULT_PROCESSED_DIR / f"{fmri_source_dir.name}.{workflow}.{atlas}.arrow"
        c.run(f"mkdir -p {ts_output_dir}")
        convert_fMRIvols_to_A424(str(fmri_source_dir), ts_output_dir, dataset_name='preventad')
        c.run(f"mkdir -p {output_dir}")
        convert_to_brainlm_arrow_datasets(str(ts_output_dir), str(output_dir), dataset_name='preventad', ts_min_length=TIMESERIES_LENGTH, compute_Stats=True)
    else:
        raise ValueError(f"unsupported workflow {workflow}.")
