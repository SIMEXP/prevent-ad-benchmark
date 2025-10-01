import invoke
from nilearn.datasets import fetch_development_fmri
from hfplayground.data.prepare import denoise_development_dataset, gigaconnectome_development_dataset, resample_atlas, downsample_for_tutorial, brain_region_coord_to_arrow
from hfplayground.data.gigaconnectome import denoise_dataset, gigaconnectome_dataset
from hfplayground.data.brainlm import convert_fMRIvols_to_A424, convert_to_arrow_datasets

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


@invoke.task
def brainlm_workflow_timeseries(c):
    # denoise_dataset(
    #     "./data/source/dataset-preventad_version-8.1internal_pipeline-gigapreprocess2/mri/wave1/fmriprep-20.2.8lts",
    #     "data/interim/dataset-preventad.brainlm",
    #     grand_mean_scale=False
    # )
    # c.run("mkdir -p ./data/interim/dataset-preventad.brainlm.a424")
    # convert_fMRIvols_to_A424(
    #     "./data/source/dataset-preventad_version-8.1internal_pipeline-gigapreprocess2/mri/wave1/fmriprep-20.2.8lts",
    #     "./data/interim/dataset-preventad.brainlm.a424",
    #     dataset_name='preventad'
    # )
    c.run("mkdir -p ./data/processed/dataset-preventad.brainlm.arrow")
    convert_to_arrow_datasets(
        "./data/interim/dataset-preventad.brainlm.a424",
        "./data/processed/dataset-preventad.brainlm.arrow",
        dataset_name='preventad',
        ts_min_length=150, compute_Stats=True
    )


@invoke.task
def gigaconnectome_workflow_timeseries(c):
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
