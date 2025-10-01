from nilearn import datasets, image
from nilearn.maskers import NiftiMasker, NiftiLabelsMasker
from nilearn.interfaces.fmriprep import load_confounds_strategy
from pathlib import Path
from tqdm import tqdm
import os
import numpy as np
from datasets import Dataset
from importlib.resources import files
import pandas as pd
from sklearn.preprocessing import RobustScaler
import nibabel as nib

seg_name = 'A424+2mm'
ATLAS_FILE = f'resource/preventad/resample_{seg_name}.nii.gz'
denoise_strategy_name = 'simple+gsr'
denoise_strategy = {
    'denoise_strategy': 'simple',
    'motion': 'basic',
    'global_signal': 'basic',
}
ts_min_length = 150
phenotype_mapper = {
    'filepath': "data/source/dataset-preventad_version-8.1internal_pipeline-gigapreprocess2/dataset-preventad81internal_desc-sexage_pheno.tsv",
    'index_col': "identifier",
    'conver_data': ['Sex', 'Candidate_Age']
}
example_subject_MNI = "data/source/dataset-preventad_version-8.1internal_pipeline-gigapreprocess2/mri/wave1/fmriprep-20.2.8lts/sub-MTL0001/ses-BL00A/func/sub-MTL0001_ses-BL00A_task-enc_space-MNI152NLin2009cAsym_boldref.nii.gz"

def denoise_dataset(sourcedata_dir, processed_dir, grand_mean_scale=False):
    """Download and preprocess the nilearn development dataset.

    This is an extremely lazy version as for code implementation.

    post fmriprep processing details
    Denoising: Simple strategy with 6 motion parameters.
    Scaling: None.
    Mask: generic MNI152 whole brain mask.
    """
    phenotype = pd.read_csv(phenotype_mapper['filepath'], index_col=phenotype_mapper['index_col'], sep='\t')
    func_paths = []
    for idx in phenotype.index:
        cur = Path(sourcedata_dir) / '/'.join(idx.split('_')[:2]) / f"func/{idx}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
        func_paths.append(str(cur))

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
    mni_mask = resample_atlas(mni_mask, '/tmp/')

    Path(f"{processed_dir}").mkdir(exist_ok=True, parents=True)
    # I did not do signal normalisation here. It will throw brainlm results off.
    masker = NiftiMasker(
        mask_img=mni_mask,
        standardize=grand_mean_scale,
        smoothing_fwhm=None, # voxel: 4 mm, smoothing kernel: 1.5 - 3 times
        verbose=2
    )
    confounds, sample_masks = load_confounds_strategy(img_files=raw_to_preproc, **denoise_strategy)
    for preproc_path, raw, conf, sm in tqdm(zip(niis_preproc_path, raw_to_preproc, confounds, sample_masks), desc="Save denoising data..."):
        fd = masker.fit_transform(raw, confounds=conf, sample_mask=sm)
        nii = masker.inverse_transform(fd)
        nii.to_filename(preproc_path)



def gigaconnectome_dataset(sourcedata_dir, processed_dir, arrow_dir):
    """Extract time series to arrow dataset.

    This mirrors gigaconnectome.
    """
    phenotype = pd.read_csv(phenotype_mapper['filepath'], index_col=phenotype_mapper['index_col'], sep='\t')
    func_paths = []
    for idx in phenotype.index:
        cur = Path(sourcedata_dir) / '/'.join(idx.split('_')[:2]) / f"func/{idx}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
        func_paths.append(str(cur))
    mni_mask = datasets.fetch_icbm152_2009()['mask']
    mni_mask = resample_atlas(mni_mask, '/tmp/')
    # quick preprocessing

    # Check if the processed directory exists
    niis_to_extract = []
    ts_file_paths = []
    for path_raw in func_paths:
        nii_name = path_raw.split('/')[-1].replace('preproc', denoise_strategy_name)
        nii_path = Path(f"{processed_dir}/{nii_name}")
        matches = nii_name.split('_space-')[0]
        ts_filename = f"{matches}_seg-{seg_name}_desc-{denoise_strategy_name}_timeseries.tsv"
        ts_path = Path(f"{processed_dir}.a424/{ts_filename}")
        if not ts_path.exists():
            niis_to_extract.append(nii_path)
            ts_file_paths.append(ts_path)
    if len(niis_to_extract) == 0:
        print("all time series extracted.")

    else:
        Path(f"{processed_dir}.a424").mkdir(exist_ok=True, parents=True)
        complete_labels = (np.arange(424)+1).tolist()

        atlas_masker = NiftiLabelsMasker(
            labels_img=files('hfplayground') / ATLAS_FILE,
            labels=complete_labels,
            mask_img=mni_mask, verbose=3
        ).fit()  # no scaling here

        for ts_path, seg_ts in tqdm(zip(ts_file_paths, niis_to_extract), desc="save time series..."):
            seg_ts = atlas_masker.transform(seg_ts)
            seg_ts = pd.DataFrame(seg_ts, columns=[int(l) for l in atlas_masker.labels_])
            seg_ts = seg_ts.reindex(columns=complete_labels)
            seg_ts.fillna("n/a", inplace=True)
            seg_ts.to_csv(ts_path, sep='\t', index=False)

    if not arrow_dir:
        print("No arrow_dir provided, skipping arrow conversion.")
        return None

    convert_data = ['Sex', 'Candidate_Age']
    phenotype = pd.read_csv("data/source/dataset-preventad_version-8.1internal_pipeline-gigapreprocess2/dataset-preventad81internal_desc-sexage_pheno.tsv", index_col="identifier", sep='\t')
    # hard coded filter....
    phenotype = phenotype.loc[phenotype['cerebellum_coverage']>0.75, :]
    phenotype = phenotype.loc[phenotype['proportion_kept']>0.5, :]
    phenotype = phenotype.loc[phenotype['ses'] == "BL00", :]
    phenotype = phenotype.loc[phenotype['run'] == 1, :]
    timeseries_files = [Path(f"{processed_dir}.a424") / f'{identifier}_seg-{seg_name}_desc-{denoise_strategy_name}_timeseries.tsv' for identifier in phenotype.index]
    # timeseries_files = list(Path(f"{processed_dir}.a424").glob('*seg-*_timeseries.tsv'))
    timeseries_files.sort()
    dataset_dict = {
        "robustscaler_timeseries": [],
        "raw_timeseries": [],
        "zscore_timeseries": [],
        "filename":[],
        "participant_id":[]
    }
    for col in convert_data:
        dataset_dict[col] = []

    for file_path in tqdm(timeseries_files, desc="convert to arrow"):
        seg_ts = pd.read_csv(file_path, sep='\t', header=0, na_values="n/a").values.astype(np.float32)
        # apply robust scaling to the time series
        scaler = RobustScaler()
        seg_ts_robustscaler = scaler.fit_transform(seg_ts)
        # they filled missing values with 0, so we do the same..... this is bad
        seg_ts_robustscaler = np.nan_to_num(seg_ts_robustscaler, nan=0.0, posinf=0.0, neginf=0.0)
        seg_ts_z = (seg_ts - np.mean(seg_ts, axis=0)) / np.std(seg_ts, axis=0)
        # participant_id = file_path.stem.split('_')[0]
        participant_id = Path(file_path).stem.split('_seg')[0]
        dataset_dict["raw_timeseries"].append(seg_ts)
        dataset_dict["robustscaler_timeseries"].append(seg_ts_robustscaler)
        dataset_dict["zscore_timeseries"].append(seg_ts_z)
        dataset_dict["filename"].append(str(file_path.name))
        dataset_dict["participant_id"].append(participant_id)
        for col in convert_data:
            dataset_dict[col].append(phenotype.loc[participant_id, col])
    arrow_train_dataset = Dataset.from_dict(dataset_dict)
    arrow_train_dataset.save_to_disk(
        dataset_path=Path(arrow_dir)
    )
    print("Done.")


def brain_region_coord_to_arrow():
    """Save Brain Region Coordinates Into Another Arrow Dataset"""
    coords_dat = np.loadtxt(files('hfplayground') / "resource/brainlm/atlases/A424_Coordinates.dat").astype(np.float32)
    coords_pd = pd.DataFrame(coords_dat, columns=["Index", "X", "Y", "Z"])
    coords_dataset = Dataset.from_pandas(coords_pd)
    coords_dataset.save_to_disk(
        dataset_path=files('hfplayground') / "resource/brainlm/atlases/brainregion_coordinates.arrow")


def resample_atlas(nii_file, output_dir):
    fname = os.path.basename(nii_file)
    exmple_nii = image.load_img(example_subject_MNI)
    downsample_data = image.resample_img(
        nii_file,
        target_affine=exmple_nii.affine,
        target_shape=exmple_nii.shape,
        interpolation='nearest',
        force_resample=True,
        copy_header=True
    )
    downsample_data.to_filename(Path(output_dir) / f'resample_{fname}')
    return Path(output_dir) / f'resample_{fname}'