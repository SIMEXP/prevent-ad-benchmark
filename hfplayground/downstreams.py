import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from datasets import load_from_disk
from nilearn.connectome import ConnectivityMeasure
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate


# we are using the raw recording features here. All the outputs from 'extract.py should contain it'
BASELINE_FEAT = "./outputs/brainlm.vitmae_650M.direct_transfer.gigaconnectome"  


def get_baseline_data(timeseries_length=140):
    # load data
    features_direct = load_from_disk(BASELINE_FEAT)

    ts_flatten = [np.array(example).reshape(3, 424, timeseries_length)[0].T.flatten() for example in features_direct['padded_recording']]

    correlation_baseline = ConnectivityMeasure(
        kind="correlation", vectorize=True, discard_diagonal=True
    )
    ts = [np.array(example).reshape(3, 424, timeseries_length)[0].T for example in features_direct['padded_recording']]
    fc = correlation_baseline.fit_transform(ts)
    labels = features_direct['Sex'], (np.array(features_direct['Candidate_Age']) / 12).tolist()
    return ts_flatten, fc, labels


def svm_pipeline(x, y):
   
    if isinstance(y[0], str):
        le = LabelEncoder()
        y = le.fit_transform(y)

        pipe = Pipeline([
            ('scaler', StandardScaler()), 
            ('pca', PCA(n_components=25)), 
            ('svc', SVC(C=1, class_weight='balanced'))
        ])
        scoring = {
            'acc': 'accuracy', 
            'auc': 'roc_auc', 
            'f1': 'f1'
        }
        cv = StratifiedShuffleSplit(n_splits=100, random_state=1)
        return cross_validate(pipe, x, y, cv=cv, scoring=scoring)
    else:
        pipe = Pipeline([
            ('scaler', StandardScaler()), 
            ('pca', PCA(n_components=25)), 
            ('svc', SVR())
        ])
        scoring = {
            'nrmse': 'neg_root_mean_squared_error', 
            'r2': 'r2'
        }
        cv = ShuffleSplit(n_splits=100, random_state=1)
        return cross_validate(pipe, x, y, cv=cv, scoring=scoring)


def create_baseline_data():
    ts, fc, (sex, age) = get_baseline_data(timeseries_length=140)
    for bn_feat, feat_name in zip((ts, fc), ('timeseries', 'connectivity')):
        for target, target_name in zip((sex, age), ('sex', 'age')):
            output_path = f'outputs/x-{feat_name}_y-{target_name}_prediction.tsv'
            scores = svm_pipeline(bn_feat, target)
            pd.DataFrame(scores).to_csv(output_path, sep='\t')
