"""Load and prepare prediction targets for downstream evaluation."""

import re

import numpy as np
import pandas as pd
from datasets import load_from_disk

from preventad_benchmark.config import PHENOTYPE_TARGETS


def load_prediction_targets(features_path, phenotype_path=None):
    """Load phenotype targets and match order with a features Arrow dataset.

    Works for both BrainLM and BrainHarmonix datasets — any Arrow dataset
    that contains 'participant_id' column. This is mapped to the subject-level
    targets TSV.

    The data will be returned in the same order as the features dataset, so
    that they can be used together for downstream evaluation.

    Args:
        features_path: Path to Arrow dataset with participant_id.
        phenotype_path: Path to subject-level targets TSV. Defaults to PHENOTYPE_TARGETS.

    Returns:
        dict with keys 'sex', 'age', 'splifhalfage', 'progess2mci', 'aps'.
    """
    phenotype_path = phenotype_path or str(PHENOTYPE_TARGETS)
    features = load_from_disk(str(features_path))

    # Load phenotype and match participant order
    pheno_df = pd.read_csv(
        phenotype_path, sep="\t", header=0, index_col=0, na_values="n/a"
    )
    participant_ids = [
        re.search(r"sub-(MTL\d{4})_", stem)[1]
        for stem in features["participant_id"]
    ]
    pheno_df = pheno_df.loc[participant_ids]

    # there are multiple sessions per subject, keep the first session
    pheno_df = pheno_df.groupby(level=0).first()

    # Age: months -> years
    age_years = (pheno_df["Candidate_Age"].astype(float) / 12.0).tolist()

    # Binary age split at median
    age_med = np.median(age_years)
    young_old = [
        "young" if age <= age_med else "old"
        for age in age_years
    ]

    # MCI progression: binary from MCI_onset_age
    pheno_df["progess_to_mci"] = pheno_df["MCI_onset_age"].apply(
        lambda x: "yes" if not np.isnan(x) else "no"
    )

    # centiloid -> 20 as the cutoff
    centiloid_bin = []
    for val in pheno_df["centiloid_s0_WhlCbl"].astype(float).tolist():
        if np.isnan(val):
            centiloid_bin.append(None)
        elif val > 20:
            centiloid_bin.append('positive')
        else:
            centiloid_bin.append('negative')
    

    # β-amyloid (Aβ) positive SUVR ->1.26 to determine Ab-positivity
    ab_bin = []
    for val in pheno_df["amyloid_index_SUVR"].astype(float).tolist():
        if np.isnan(val):
            ab_bin.append(None)
        elif val > 1.26:
            ab_bin.append('positive')
        else:
            ab_bin.append('negative')

    return {
        "sex": pheno_df["Sex"].tolist(),
        "age": age_years,
        "splifhalfage": young_old,
        "progess2mci": pheno_df["progess_to_mci"].tolist(),
        "centiloid": pheno_df["centiloid_s0_WhlCbl"].astype(float).tolist(),
        "centiloidbin": centiloid_bin,
        "abSUVR": pheno_df["amyloid_index_SUVR"].astype(float).tolist(),
        "abSUVRbin": ab_bin,
    }
