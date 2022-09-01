from sklearn.model_selection import StratifiedKFold
import copy
import numpy as np
import gc
import pandas as pd

from .pkl_utils import *
from .metrics import get_res, metric_lst
from .utils import df_to_file

def create_KFCV_index(k, x, stratify):
    cv_indexes = {}
    skf = StratifiedKFold(n_splits=k)
    for i, (train_index, test_index) in enumerate(skf.split(x, stratify)):
        cv_indexes[i]={}
        cv_indexes[i]['train_index'] = train_index
        cv_indexes[i]['test_index'] = test_index

    dump(file_name='%d_folds.pkl'%k, data=cv_indexes)
    print('"%d_folds.pkl" is created successfully.'%k)


def do_kfold(model, X, y,
             CV_file, round_=5, metric_lst=metric_lst, save_plot=None):
    reg = model
    print(reg.__class__.__name__)
    cv_indexes = load_pkl(CV_file)
    result_df = pd.DataFrame(y.values, columns=['y_true'])

    y_pred_col = 'y_pred'
    result_df[y_pred_col] = -1
    result_df['fold'] = -1
    results = []

    for fold_id in sorted(cv_indexes.keys()):
        train_index = cv_indexes[fold_id]['train_index']
        test_index = cv_indexes[fold_id]['test_index']

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        m = copy.deepcopy(reg)

        try:
            m.fit(X_train, y_train, verbose = 0)
        except Exception as exc:
            m.fit(X_train, y_train)

        test_pred = m.predict(X_test)

        result_df.loc[test_index, [y_pred_col]] = test_pred.reshape(-1, 1)
        result_df.loc[test_index, ['fold']] = fold_id
        results.append(get_res(y_test, test_pred))


        del m
        gc.collect()

    df = pd.DataFrame(np.array(results), columns=[m.__name__ for m in metric_lst])
    a = df.describe()
    a = a.loc[['mean', 'std'], :]

    fold_results = pd.concat([df,a])

    fold_col = ['Fold_%d'%(1+fold_id) for fold_id in sorted(cv_indexes.keys())] + ['Mean', 'Std']

    if True:
        df_to_file(
            fold_results,
            round_=round_,
            fold_col = fold_col
        )

    return result_df, fold_results