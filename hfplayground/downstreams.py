import re
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from datasets import load_from_disk
from nilearn.connectome import ConnectivityMeasure
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from pathlib import Path

# we are using the raw recording features here. All the outputs from 'extract.py should contain it'
BASELINE_FEAT = "./outputs/brainlm.vitmae_650M.direct_transfer.gigaconnectome"
ALL_PHENO = "./data/source/dataset-preventad_version-8.1internal_pipeline-gigapreprocess2/dataset-preventad81internal_desc-subjlvltargets_pheno.tsv"
N_SPLITS = 100


def load_prediction_targets(features_path):
    """Load Phenotype Information for Downstream Tasks and match the Order with Features Dataset."""
    features = load_from_disk(features_path)

    # load phenotype
    pheno_df = pd.read_csv(ALL_PHENO, sep='\t', header=0, index_col=0, na_values="n/a")
    arrow_feature_participant_id = [re.search(r"sub-(MTL\d{4})_", stem)[1] for stem in features['participant_id']]
    pheno_df = pheno_df.loc[arrow_feature_participant_id, :]  # match order and subject

    # find median age
    age_med = np.median(features['Candidate_Age'])
    young_old = ['young' if age <= age_med else 'old' for age in features['Candidate_Age']]

    # convert MCI_onset_age to binary labels indicating whether subject progress to MCI
    pheno_df['progess_to_mci'] = pheno_df['MCI_onset_age'].apply(lambda x: 'yes' if not np.isnan(x) else 'no')

    # return all labels
    return {
        'sex': features['Sex'], 
        'age': (np.array(features['Candidate_Age']) / 12).tolist(),
        'splifhalfage': young_old,
        'progess2mci': pheno_df['progess_to_mci'].tolist(),
    }

def get_baseline_data(timeseries_length=140):
    # load data
    features = load_from_disk(BASELINE_FEAT)

    # connectome and timeseries
    ts_flatten = [np.array(example).reshape(3, 424, timeseries_length)[0].T.flatten() for example in features['padded_recording']]

    correlation_baseline = ConnectivityMeasure(
        kind="correlation", vectorize=True, discard_diagonal=True
    )
    ts = [np.array(example).reshape(3, 424, timeseries_length)[0].T for example in features['padded_recording']]
    fc = correlation_baseline.fit_transform(ts)
    return ts_flatten, fc


def svm_pipeline(x, y):
   
    if isinstance(y[0], str):
        le = LabelEncoder()
        y = le.fit_transform(y)

        pipe = Pipeline([
            ('scaler', StandardScaler()), 
            ('pca', PCA(n_components=75)), 
            ('svc', SVC(C=1, class_weight='balanced'))
        ])
        scoring = {
            'acc': 'accuracy', 
            'auc': 'roc_auc', 
            'f1': 'f1'
        }
        cv = StratifiedShuffleSplit(n_splits=N_SPLITS, random_state=1)
        return cross_validate(pipe, x, y, cv=cv, scoring=scoring)
    else:
        pipe = Pipeline([
            ('scaler', StandardScaler()), 
            ('pca', PCA(n_components=75)), 
            ('svc', SVR())
        ])
        scoring = {
            'nrmse': 'neg_root_mean_squared_error', 
            'nmae': 'neg_mean_absolute_error',
            'r2': 'r2'
        }
        cv = ShuffleSplit(n_splits=N_SPLITS, random_state=1)
        return cross_validate(pipe, x, y, cv=cv, scoring=scoring)


def linear_pipeline(x, y):
   
    if isinstance(y[0], str):
        le = LabelEncoder()
        y = le.fit_transform(y)

        pipe = Pipeline([
            ('scaler', StandardScaler()), 
            ('pca', PCA(n_components=75)), 
            ('logistic_reg', LogisticRegression())
        ])
        scoring = {
            'acc': 'accuracy', 
            'auc': 'roc_auc', 
            'f1': 'f1'
        }
        cv = StratifiedShuffleSplit(n_splits=N_SPLITS, random_state=1)
        return cross_validate(pipe, x, y, cv=cv, scoring=scoring)
    else:
        pipe = Pipeline([
            ('scaler', StandardScaler()), 
            ('pca', PCA(n_components=75)), 
            ('linear_reg', LinearRegression())
        ])
        scoring = {
            'nrmse': 'neg_root_mean_squared_error', 
            'nmae': 'neg_mean_absolute_error',
            'r2': 'r2'
        }
        cv = ShuffleSplit(n_splits=N_SPLITS, random_state=1)
        return cross_validate(pipe, x, y, cv=cv, scoring=scoring)


def baseline_experiment() -> None:
    ts, fc = get_baseline_data(timeseries_length=140)
    labels = load_prediction_targets(BASELINE_FEAT)
    for bn_feat, feat_name in zip((ts, fc), ('timeseries', 'connectivity')):
        for target_name in ['sex', 'age', 'splifhalfage', 'progess2mci']:
            svm_output_path = f'outputs/downstreams/baseline/x-{feat_name}_y-{target_name}_svm_prediction.tsv'
            if Path(svm_output_path).exists():
                print(f"{svm_output_path} exists, skip")
                continue
            svm_scores = svm_pipeline(bn_feat, labels[target_name])
            pd.DataFrame(svm_scores).to_csv(svm_output_path, sep='\t')

            linear_output_path = f'outputs/downstreams/baseline/x-{feat_name}_y-{target_name}_linear_prediction.tsv'
            if Path(linear_output_path).exists():
                print(f"{linear_output_path} exists, skip")
                continue
            linear_scores = svm_pipeline(bn_feat, labels[target_name])
            pd.DataFrame(linear_scores).to_csv(linear_output_path, sep='\t')


def brainlm_experiment(features_path: Path, feat_name: str) -> None:
    features = np.array(load_from_disk(features_path)['feat_name']).squeeze()
    if not Path(f'outputs/downstreams/{p.name}/').exists():
        Path(f'outputs/downstreams/{p.name}/').mkdir(parents=True)

    labels = load_prediction_targets(features_path)
    for target_name in ['sex', 'age', 'splifhalfage', 'progess2mci']:
        svm_output_path = f'outputs/downstreams/{p.name}/x-{feat_name.replace('_', '-')}_y-{target_name}_svm_prediction.tsv'
        if Path(svm_output_path).exists():
            print(f"{svm_output_path} exists, skip")
            continue
        svm_scores = svm_pipeline(features, labels[target_name])
        pd.DataFrame(svm_scores).to_csv(svm_output_path, sep='\t')

        linear_output_path = f'outputs/downstreams/{p.name}/x-{feat_name.replace('_', '-')}_y-{target_name}_linear_prediction.tsv'
        if Path(linear_output_path).exists():
            print(f"{linear_output_path} exists, skip")
            continue
        linear_scores = svm_pipeline(features, labels[target_name])
        pd.DataFrame(linear_scores).to_csv(linear_output_path, sep='\t')


if __name__ == "__main__":
    baseline_experiment()
    for p in Path("./outputs/").glob("*finetune*"):
        for feat_name in ['cls_token', 'cls_embedding']:
            brainlm_experiment(p, feat_name)