"""A proper standardization workflow by neuroimaging data processing standard."""
from nilearn import datasets
from nilearn.maskers import NiftiMasker, NiftiLabelsMasker
from nilearn.interfaces.fmriprep import load_confounds_strategy
from pathlib import Path
from tqdm import tqdm
import os
import numpy as np
import pandas as pd

from preventad_benchmark.config import (
    BRAINLM_DATASET_CONFIGS,
    DENOISE_STRATEGY,
    DENOISE_STRATEGY_NAME,
)
from preventad_benchmark.data.utils import resample_atlas, fetch_and_prepare_schaefer400_atlas, load_phenotype


from preventad_benchmark.config import A424_NPARCELS, BRAINHARMONIX_SEG_NAME, BRAINHARMONIX_SCHAEFER_ROIS, PHENOTYPE_SEXAGE


_preventad_config = BRAINLM_DATASET_CONFIGS["preventad"]
seg_name = _preventad_config["seg_name"]
denoise_strategy_name = DENOISE_STRATEGY_NAME
denoise_strategy = DENOISE_STRATEGY
phenotype_mapper = _preventad_config["phenotype"]

def denoise_dataset(sourcedata_dir, processed_dir, standarsization=True):
    """This is extracted from giga connectome.

    post fmriprep processing details
    Denoising: Simple strategy with 6 motion parameters.
    standardization: z score per run
    Mask: generic MNI152 whole brain mask.
    """
    phenotype = pd.read_csv(phenotype_mapper['filepath'], index_col=phenotype_mapper['index_col'], sep='\t')
    func_paths = []
    for idx in phenotype.index:
        cur = Path(sourcedata_dir) / '/'.join(idx.split('_')[:2]) / f"func/{idx}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
        if 'ses-BL' in idx:
            func_paths.append(str(cur))
    print(f"Found {len(func_paths)} fMRI files to denoise.")
    # Check if the processed directory exists
    raw_to_preproc = []
    niis_preproc_path = []
    for path_raw in func_paths:
        nii_name = path_raw.split('/')[-1].replace('preproc', denoise_strategy_name)
        nii_path = Path(f"{processed_dir}/{nii_name}")

        if not nii_path.exists():
            raw_to_preproc.append(path_raw)
            niis_preproc_path.append(nii_path)

    if len(raw_to_preproc) == 0:
        print("No raw data to preprocess.")
        return
    mni_mask = datasets.fetch_icbm152_2009()['mask']
    mni_mask = resample_atlas(mni_mask, os.environ["SLURM_TMPDIR"])
    print(f"Using MNI mask at {mni_mask}")

    Path(f"{processed_dir}").mkdir(exist_ok=True, parents=True)
    masker = NiftiMasker(
        mask_img=mni_mask,
        standardize=standarsization,
        smoothing_fwhm=None,
        verbose=0
    ).fit()
    confounds, sample_masks = load_confounds_strategy(img_files=raw_to_preproc, **denoise_strategy)
    for preproc_path, raw, conf, sm in tqdm(zip(niis_preproc_path, raw_to_preproc, confounds, sample_masks), desc="Save denoising data..."):
        fd = masker.transform(raw, confounds=conf, sample_mask=sm)
        nii = masker.inverse_transform(fd)
        nii.to_filename(preproc_path)


def extract_timeseries_from_nifti(source_denoised_dir, output_ts_dir, atlas="schaefer400"):
    """Extract time series from denoised NIfTI files using Schaefer 400 atlas or A424."""
    # Fetch and prepare atlas
    if atlas not in ["schaefer400", "a424"]:
        raise ValueError(f"We only support two atlases: schaefer400, a424. Your input is {atlas}.")
    elif atlas == "schaefer400":
        atlas_path, _ = fetch_and_prepare_schaefer400_atlas()
        atlas_labels = (np.arange(BRAINHARMONIX_SCHAEFER_ROIS) + 1).tolist()
        seg_name = BRAINHARMONIX_SEG_NAME
    else:
        atlas_path = str(_preventad_config["atlas_file"])
        atlas_labels = (np.arange(A424_NPARCELS)+1).tolist()
        seg_name = str(_preventad_config["seg_name"])

    # Fetch MNI mask
    mni_mask = datasets.fetch_icbm152_2009()["mask"]
    mni_mask = resample_atlas(mni_mask, "/tmp")

    # Get list of denoised NIfTI files
    phenotype = load_phenotype(PHENOTYPE_SEXAGE, apply_qc=True)
    denoised_niis = [Path(source_denoised_dir) / f'{identifier}_space-MNI152NLin2009cAsym_desc-{DENOISE_STRATEGY_NAME}_bold.nii.gz' for identifier in phenotype.index]
    # denoised_niis = list(source_denoised_dir.glob(f"*_desc-{DENOISE_STRATEGY_NAME}_bold.nii.gz"))
    # denoised_niis.sort()
    print(f"Found {len(denoised_niis)} denoised NIfTI files")

    # Determine which files need processing
    niis_to_extract = []
    ts_file_paths = []
    for nii_path in denoised_niis:
        # Extract identifier from filename
        # sub-MTL0001_ses-BL00A_task-rest_run-1_space-MNI152NLin2009cAsym_desc-simple+gsr_bold.nii.gz
        # -> sub-MTL0001_ses-BL00A_task-rest_run-1
        identifier = nii_path.name.split("_space-")[0]
        ts_filename = f"{identifier}_seg-{seg_name}_desc-{DENOISE_STRATEGY_NAME}_timeseries.tsv"
        ts_path = output_ts_dir / ts_filename

        if not ts_path.exists():
            niis_to_extract.append(nii_path)
            ts_file_paths.append(ts_path)

    if len(niis_to_extract) == 0:
        print("All time series already extracted.")
        return

    print(f"Extracting time series from {len(niis_to_extract)} NIfTI files...")

    # Create output directory
    output_ts_dir.mkdir(exist_ok=True, parents=True)

    # Setup masker for atlas
    atlas_masker = NiftiLabelsMasker(
        labels_img=atlas_path,
        labels=atlas_labels,
        mask_img=mni_mask,
        verbose=0,
    ).fit()

    # Extract time series
    for ts_path, nii_path in tqdm(
        zip(ts_file_paths, niis_to_extract),
        total=len(niis_to_extract),
        desc="Extracting time series",
    ):
        seg_ts = atlas_masker.transform(nii_path)
        seg_ts = pd.DataFrame(seg_ts, columns=[int(l) for l in atlas_masker.labels_])
        seg_ts = seg_ts.reindex(columns=atlas_labels)
        seg_ts.to_csv(ts_path, sep="\t", na_rep="n/a", index=False)
