import re

import numpy as np
import pandas as pd
from nilearn.datasets import fetch_atlas_schaefer_2018
from pathlib import Path
from nilearn import image
from datasets import Dataset, load_from_disk
import torch
from tqdm import tqdm

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer

from preventad_benchmark.evaluation.targets import load_prediction_targets
from preventad_benchmark.config import (
    BRAINLM_COORD_DATASET,
    BRAINLM_COORD_FILE,
    EXAMPLE_SUBJECT_MNI,
    PHENOTYPE_SEXAGE,
    DENOISE_STRATEGY_NAME,
    TIMESERIES_LENGTH,
    QC_FILTERS,
)

TS_MIN_LENGTH = TIMESERIES_LENGTH


def resample_atlas(nii_file, output_dir, mni_space="MNI152NLin2009cAsym"):
    """Resample an atlas NIfTI to match the MNI space of the example subject."""
    example_mni_path = str(EXAMPLE_SUBJECT_MNI)
    if mni_space != "MNI152NLin2009cAsym":
        example_mni_path = example_mni_path.replace("MNI152NLin2009cAsym", mni_space)

    if not Path(example_mni_path).exists():
        raise FileNotFoundError(f"Example MNI file not found: {example_mni_path}")

    fname = Path(nii_file).name
    example_nii = image.load_img(example_mni_path)
    downsample_data = image.resample_img(
        nii_file,
        target_affine=example_nii.affine,
        target_shape=example_nii.shape,
        interpolation='nearest',
        force_resample=True,
        copy_header=True
    )
    downsample_data.to_filename(Path(output_dir) / f'resample_{fname}')
    return Path(output_dir) / f'resample_{fname}'


def fetch_and_prepare_schaefer400_atlas():
    """Fetch Schaefer 400 atlas and prepare for use."""
    print("Fetching Schaefer 400 atlas (17 networks, 2mm resolution)...")
    atlas = fetch_atlas_schaefer_2018(
        n_rois=400, yeo_networks=17, resolution_mm=2
    )

    # Resample to match PreventAD data space
    print("Resampling atlas to match PreventAD data space...")
    resampled_atlas = resample_atlas(atlas.maps, "/tmp")
    return resampled_atlas, atlas.labels


def compute_normalization_params(dataset):
    """Compute ROI median and IQR over per-sample ROI *means*.

    This has to be revisited as there are multiple ways to compute normalization
    parameters provided in the BrainHarmonix code.

    Args:
        dataset: Huggingface Dataset with 'raw_timeseries' column.
    Returns:
        dict: Dictionary with keys 'mean', 'std', 'median', 'iqr'.
    """
    all_mean_data = [
        np.nanmean(d["raw_timeseries"], axis=0)  # (number of ROIs, )
        for d in dataset
    ]
    overall_mean = np.nanmean(all_mean_data)
    imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=overall_mean)
    imp.fit(all_mean_data)
    all_mean_data = imp.transform(all_mean_data)

    mean = np.nanmean(all_mean_data, axis=0)
    std = np.std(all_mean_data, axis=0)
    median = np.median(all_mean_data, axis=0)
    q75 = np.percentile(all_mean_data, 75, axis=0)
    q25 = np.percentile(all_mean_data, 25, axis=0)
    iqr = q75 - q25
    return {
        'mean': mean,
        'std': std + 1e-9,  # avoid 0
        'median': median,
        'iqr': iqr + 1e-9  # avoid 0
    }


def brain_region_coord_to_arrow(
    coord_file=None,
    output_path=None,
):
    """Save Brain Region Coordinates into an Arrow Dataset."""
    coord_file = coord_file or BRAINLM_COORD_FILE
    output_path = output_path or BRAINLM_COORD_DATASET
    coords_dat = np.loadtxt(Path(coord_file)).astype(np.float32)
    coords_pd = pd.DataFrame(coords_dat, columns=["Index", "X", "Y", "Z"])
    coords_dataset = Dataset.from_pandas(coords_pd)
    coords_dataset.save_to_disk(dataset_path=Path(output_path))


def load_phenotype(phenotype_path=None, apply_qc=True):
    """Load PreventAD phenotype TSV and optionally apply QC filters.

    Args:
        phenotype_path: Path to phenotype TSV. Defaults to PHENOTYPE_SEXAGE.
        apply_qc: If True, filter by cerebellum coverage, proportion kept,
            baseline session, and first run (using QC_FILTERS from config).

    Returns:
        pd.DataFrame indexed by identifier.
    """
    phenotype_path = phenotype_path or str(PHENOTYPE_SEXAGE)
    phenotype = pd.read_csv(phenotype_path, index_col="identifier", sep="\t")
    if apply_qc:
        phenotype = phenotype.loc[phenotype["cerebellum_coverage"] > QC_FILTERS["cerebellum_coverage_min"], :]
        phenotype = phenotype.loc[phenotype["proportion_kept"] > QC_FILTERS["proportion_kept_min"], :]
        phenotype = phenotype.loc[phenotype["ses"] == QC_FILTERS["session"], :]
        phenotype = phenotype.loc[phenotype["run"] == QC_FILTERS["run"], :]
        if phenotype.participant_id.duplicated().any():
            # keep the first session for duplicated participants (with both BL00A and BL00B)
            phenotype = phenotype.drop_duplicates(subset="participant_id", keep="first")
    return phenotype


def convert_to_arrow_dataset(t1_output_dir, output_ts_dir, output_arrow_dir, seg_name):
    """Convert TSV time series files to Arrow dataset with filtering and scaling."""
    print("\nConverting time series to Arrow dataset...")

    # Load phenotype data with QC filters
    # schafer400 does not cover cerebellum but I want to keep consistent with other pipelines
    phenotype = load_phenotype(apply_qc=True)
    print(f"After filtering: {len(phenotype)} entries")

    # Build list of time series files
    timeseries_files = []
    valid_identifiers = []
    for identifier in phenotype.index:
        # identifier format: sub-MTL0001_ses-BL00A_task-rest_run-1
        ts_filename = f"{identifier}_seg-{seg_name}_desc-{DENOISE_STRATEGY_NAME}_timeseries.tsv"
        ts_path = output_ts_dir / ts_filename
        if ts_path.exists():
            timeseries_files.append(ts_path)
            valid_identifiers.append(identifier)
        else:
            print(f"Warning: Missing time series file for {identifier}")

    print(f"Found {len(timeseries_files)} time series files matching phenotype")

    # Build dataset dictionary
    dataset_dict = {
        "raw_timeseries": [],
        "t1_filepath": [],
        "ts_filename": [],
        "participant_id": [],
    }

    if t1_output_dir is None:
        dataset_dict.pop("t1_filepath")

    # Process each time series file
    for file_path, identifier in tqdm(
        zip(timeseries_files, valid_identifiers),
        total=len(timeseries_files),
        desc="Converting to Arrow",
    ):
        # Load time series
        seg_ts = pd.read_csv(file_path, sep="\t", header=0, na_values="n/a")
        seg_ts = seg_ts.values.astype(np.float32)

        # Skip if time series is too short
        if seg_ts.shape[0] < TS_MIN_LENGTH:
            print(f"Skipping {identifier}: time series too short ({seg_ts.shape[0]} < {TS_MIN_LENGTH})")
            continue

        if t1_output_dir:
            # Load corresponding T1 image
            # Extract subject_id from identifier (e.g., "sub-MTL0001_ses-..." -> "sub-MTL0001")
            subject_id = identifier.split("_ses-")[0]
            t1_path = t1_output_dir / f"{subject_id}.pt"
            dataset_dict["t1_filepath"].append(str(t1_path))
            if not t1_path.exists():
                print(f"Skipping {identifier}: no T1 image for {subject_id}")
                continue

        # t1_tensor = torch.load(t1_path, weights_only=True)

        # Extract participant_id from filename
        participant_id = Path(file_path).stem.split("_seg")[0]

        # Add to dataset
        dataset_dict["raw_timeseries"].append(seg_ts)
        dataset_dict["ts_filename"].append(str(file_path.name))
        dataset_dict["participant_id"].append(participant_id)

    print(f"\nDataset contains {len(dataset_dict['participant_id'])} samples")

    # Create and save Arrow dataset at template path
    dir_tmp = output_arrow_dir.parent / "tmp.arrow"
    dir_tmp.mkdir(exist_ok=True, parents=True)
    tmp = Dataset.from_dict(dataset_dict)
    tmp.save_to_disk(dataset_path=dir_tmp)

    # get other phenotypes
    phenotypes = load_prediction_targets(dir_tmp)
    dataset_dict.update(phenotypes)

    # update Arrow dataset with phenotypes
    output_arrow_dir.mkdir(exist_ok=True, parents=True)
    arrow_dataset = Dataset.from_dict(dataset_dict)
    arrow_dataset.save_to_disk(dataset_path=output_arrow_dir)
    print(f"Saved Arrow dataset to {output_arrow_dir}")

    # Print dataset info
    print(f"\nDataset info:")
    print(f"  - Number of samples: {len(arrow_dataset)}")
    if len(arrow_dataset) > 0:
        sample_ts = arrow_dataset[0]["raw_timeseries"]
        sample_t1 = torch.load(arrow_dataset[0]["t1_filepath"], weights_only=True)
        print(f"  - Time series shape: {np.array(sample_ts).shape}")
        print(f"  - T1 image shape: {np.array(sample_t1).shape}")


def create_train_test_split(
    phenotype_path=None,
    test_size=0.2,
    random_state=42,
    n_splits=20,
):
    """Create stratified train/test splits.

    Stratifies using sex, progess2mci labels.

    Args:
        phenotype_path: Path to subject-level targets TSV. Defaults to
            PHENOTYPE_TARGETS.
        test_size: Fraction of data for test set. Default 0.2.
        random_state: Random seed for reproducibility. Default 42.
        n_splits: Number of shuffle splits to generate. Default 20.

    Returns:
        list of dicts, each with 'train' and 'test' keys containing
        lists of participant IDs (format: "sub-MTL0001_ses-BL00A_task-rest_run-1").
    """
    phenotype = load_phenotype(apply_qc=True)
    identifiers = phenotype.index.tolist()
    # Load prediction targets aligned to participant order
    targets = load_prediction_targets(
        participant_ids=identifiers,
        phenotype_path=phenotype_path,
    )

    # Build composite stratification key from categorical targets
    categorical_keys = ["sex", "progess2mci"]
    n = len(identifiers)
    composite = []
    for i in range(n):
        parts = []
        for key in categorical_keys:
            val = targets[key][i]
            parts.append(str(val) if val is not None and val == val else "missing")
        composite.append("_".join(parts))

    composite = np.array(composite)
    indices = np.arange(n)

    try:
        splitter = StratifiedShuffleSplit(
            n_splits=n_splits, test_size=test_size, random_state=random_state
        )
        splits = list(splitter.split(indices, composite))
    except ValueError:
        print("Fall back to sex")
        # Fall back to sex-only stratification if composite groups are too small
        sex_labels = np.array([
            str(v) if v is not None else "missing" for v in targets["sex"]
        ])
        splitter = StratifiedShuffleSplit(
            n_splits=n_splits, test_size=test_size, random_state=random_state
        )
        splits = list(splitter.split(indices, sex_labels))

    return [
        {
            "train": [identifiers[i] for i in train_idx],
            "test": [identifiers[i] for i in test_idx],
        }
        for train_idx, test_idx in splits
    ]
