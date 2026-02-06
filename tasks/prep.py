import invoke
import os
from nilearn.datasets import fetch_development_fmri, fetch_atlas_schaefer_2018
from pathlib import Path
from hfplayground.data.prepare import denoise_development_dataset, gigaconnectome_development_dataset, downsample_for_tutorial, brain_region_coord_to_arrow
from hfplayground.data.gigaconnectome import denoise_dataset, gigaconnectome_dataset
from hfplayground.data.brainlm import convert_fMRIvols_to_A424, convert_to_arrow_datasets
from hfplayground.data import brainharmonix as bh
from hfplayground.data.structural import preprocess_t1_images
from hfplayground.data.utils import fetch_and_prepare_schaefer400_atlas, resample_atlas

@invoke.task
def models(c):
    c.run("HF_HUB_ENABLE_HF_TRANSFER=1")
    c.run("huggingface-cli download vandijklab/brainlm --local-dir ./models/brainlm")


@invoke.task
def atlas(c):
    c.run("mkdir -p ./hfplayground/resource/preventad")
    resample_atlas("hfplayground/resource/brainlm/atlases/A424+4mm.nii.gz", "hfplayground/resource/preventad")
    resample_atlas("hfplayground/resource/brainlm/atlases/A424+2mm.nii.gz", "hfplayground/resource/preventad")
    brain_region_coord_to_arrow()
    print("Fetching Schaefer 400 atlas (17 networks, 2mm resolution)...")
    atlas = fetch_atlas_schaefer_2018(
        n_rois=400, yeo_networks=17, resolution_mm=2
    )
    # Resample to match PreventAD data space
    resample_atlas(atlas.maps, "/tmp")


@invoke.task
def t1(c):
    preprocess_t1_images(
        Path("data/source/dataset-preventad_version-8.1internal_pipeline-gigapreprocess2/mri/wave1/fmriprep-20.2.8lts"),
        Path("data/interim/dataset-preventad.t1")
    )


@invoke.task
def brainharmonix_gigaconnectome_preventad(c):
    bh.extract_timeseries_from_nifti(
        Path("data/interim/dataset-preventad.gigaconnectome"),  # with grand mean scaling
        Path("data/interim/dataset-preventad.gigaconnectome.schaefer400")
    )
    bh.convert_to_arrow_dataset(
        Path("data/interim/dataset-preventad.t1"),
        Path("data/interim/dataset-preventad.gigaconnectome.schaefer400"),
        Path("data/processed/dataset-preventad.brainharmonix.gigaconnectome.arrow")
    )


@invoke.task
def brainharmonix_nograndmeanscaling_preventad(c):
    # denoise_dataset(
    #     "./data/source/dataset-preventad_version-8.1internal_pipeline-gigapreprocess2/mri/wave1/fmriprep-20.2.8lts",
    #     os.environ["SLURM_TMPDIR"] + "/dataset-preventad.NoGrandmeanScaling",
    #     grand_mean_scale=False
    # )

    # # c.run("mkdir -p ./data/interim/dataset-preventad.NoGrandmeanScaling")
    # # c.run("rysnc -av " + os.environ["$SLURM_TMPDIR"] + "dataset-preventad.NoGrandmeanScaling/ ./data/interim/dataset-preventad.NoGrandmeanScaling/")

    # bh.extract_timeseries_from_nifti(
    #     # Path("data/interim/dataset-preventad.NoGrandmeanScaling"),  # no grand mean scaling
    #     Path(os.environ["SLURM_TMPDIR"] + "/dataset-preventad.NoGrandmeanScaling"),
    #     Path("data/interim/dataset-preventad.NoGrandmeanScaling.schaefer400")
    # )
    bh.convert_to_arrow_dataset(
        Path("data/interim/dataset-preventad.brainharmonix.t1"),
        Path("data/interim/dataset-preventad.NoGrandmeanScaling.schaefer400"),
        Path("data/processed/dataset-preventad.brainharmonix.NoGrandmeanScaling.arrow")
    )


@invoke.task
def brainlm_workflow_timeseries_preventad(c):
    denoise_dataset(
        "./data/source/dataset-preventad_version-8.1internal_pipeline-gigapreprocess2/mri/wave1/fmriprep-20.2.8lts",
        "data/interim/dataset-preventad.brainlm",
        grand_mean_scale=False
    )
    c.run("mkdir -p ./data/interim/dataset-preventad.brainlm.a424")
    convert_fMRIvols_to_A424(
        "./data/source/dataset-preventad_version-8.1internal_pipeline-gigapreprocess2/mri/wave1/fmriprep-20.2.8lts",
        "./data/interim/dataset-preventad.brainlm.a424",
        dataset_name='preventad'
    )
    c.run("mkdir -p ./data/processed/dataset-preventad.brainlm.arrow")
    convert_to_arrow_datasets(
        "./data/interim/dataset-preventad.brainlm.a424",
        "./data/processed/dataset-preventad.brainlm.arrow",
        dataset_name='preventad',
        ts_min_length=150, compute_Stats=True
    )



@invoke.task
def gigaconnectome_workflow_timeseries_preventad(c):
    denoise_dataset(
        "./data/source/dataset-preventad_version-8.1internal_pipeline-gigapreprocess2/mri/wave1/fmriprep-20.2.8lts",
        "data/interim/dataset-preventad.gigaconnectome",
        grand_mean_scale=True
    )
    gigaconnectome_dataset(
        "./data/source/dataset-preventad_version-8.1internal_pipeline-gigapreprocess2/mri/wave1/fmriprep-20.2.8lts",
        "data/interim/dataset-preventad.gigaconnectome",
        "data/processed/dataset-preventad.gigaconnectome.arrow"
    )


@invoke.task
def testdata(c):
    c.run("mkdir -p ./hfplayground/resource/development_fmri")
    downsample_for_tutorial("hfplayground/resource/brainlm/atlases/A424+4mm.nii.gz", "hfplayground/resource/development_fmri")
    downsample_for_tutorial("hfplayground/resource/brainlm/atlases/A424+2mm.nii.gz", "hfplayground/resource/development_fmri")
    brain_region_coord_to_arrow()

    print("Downloading the nilearn development fMRI dataset...")
    fetch_development_fmri(data_dir="data/external")
    print("Preprocessing the development dataset...")
    c.run("mkdir -p ./data/interim/development_fmri")


@invoke.task
def testdata_brainlm_workflow_timeseries(c):
    denoise_development_dataset(
        "data/external",
        "data/interim/development_fmri.brainlm",
        grand_mean_scale=False
    )
    c.run("mkdir -p ./data/interim/development_fmri.brainlm.a424")
    convert_fMRIvols_to_A424(
        "./data/interim/development_fmri.brainlm",
        "./data/interim/development_fmri.brainlm.a424"
    )
    c.run("mkdir -p ./data/processed/development_fmri.brainlm.arrow")
    convert_to_arrow_datasets(
        "./data/interim/development_fmri.brainlm.a424",
        "./data/processed/development_fmri.brainlm.arrow",
        ts_min_length=160, compute_Stats=True
    )

@invoke.task
def testdata_gigaconnectome_workflow_timeseries(c):
    denoise_development_dataset(
        "data/external",
        "data/interim/development_fmri.gigaconnectome",
        grand_mean_scale=True
    )
    gigaconnectome_development_dataset(
        "data/external",
        "data/interim/development_fmri.gigaconnectome",
        "data/processed/development_fmri.gigaconnectome.arrow"
    )
