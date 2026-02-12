from nilearn.maskers import NiftiMasker
from pathlib import Path
import torch
from tqdm import tqdm


def preprocess_t1_images(t1_source_dir, t1_output_dir):
    """Skull strip T1 images and save  to tensors for BrainHarmonix.

    Processing steps:
    1. Load T1 NIfTI in MNI152NLin6Asym space / whatever fmriprep outputs
    2. Mask just the brain with the corresponding brain mask (fmriprep outputs)
    3. Save as .pt tensor
    """
    print("\nPreprocessing T1 images for BrainHarmonix...")

    t1_output_dir.mkdir(exist_ok=True, parents=True)

    anat_dirs = list(t1_source_dir.glob("sub-MTL*/anat"))
    anat_dirs.sort()
    print(f"Found {len(anat_dirs)} subject anat directories")

    # determine which subjects need processing
    to_process = []
    for anat_dir in anat_dirs:
        subject_id = anat_dir.parent.name  # e.g., "sub-MTL0001"
        output_path = t1_output_dir / f"{subject_id}.pt"
        if not output_path.exists():
            to_process.append((anat_dir, subject_id, output_path))

    if len(to_process) == 0:
        print("All T1 images already preprocessed.")
        return

    print(f"Processing {len(to_process)} T1 images...")

    skipped_count = 0

    for anat_dir, subject_id, output_path in tqdm(to_process, desc="Processing T1"):
        # find T1 file in MNI152NLin6Asym spacedata/external
        t1_files = list(
            anat_dir.glob("*_space-MNI152NLin6Asym_desc-preproc_T1w.nii.gz")
        )
        t1_mask_files = list(
            anat_dir.glob("*_space-MNI152NLin6Asym_desc-brain_mask.nii.gz")
        )
        if not t1_files:
            skipped_count += 1
            continue

        t1_masker = NiftiMasker(mask_img=t1_mask_files[0], standardize=False).fit()
        t1_img_flat = t1_masker.transform(t1_files[0])
        img = t1_masker.inverse_transform(t1_img_flat).dataobj[:, :, :, 0]

        # Save as tensor
        tensor = torch.tensor(img, dtype=torch.float32)
        torch.save(tensor, output_path)

    print(f"Processed {len(to_process) - skipped_count} T1 images, skipped {skipped_count}")
