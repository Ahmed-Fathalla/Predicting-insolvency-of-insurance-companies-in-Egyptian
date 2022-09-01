from sklearn.metrics import mean_absolute_error as MAE_sklearn,\
                            mean_squared_error as MSE_sklearn,\
                            r2_score as R2_score_sklearn
from sklearn.metrics.regression import _check_reg_targets
import numpy as np
import pandas as pd

# from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.metrics import RootMeanSquaredError as RMSE
from tensorflow.keras import backend as K

def RMSE_sklearn(y_true, y_pred):
    return np.sqrt(MSE_sklearn(y_true, y_pred))

# def RMSE(y_true, y_pred):
#     return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def MAPE(y_true, y_pred, multioutput='uniform_average'):
    '''
    This function is implemented by: Ahmed Fathalla <a.fathalla@science.suez.edu.eg>,
         The implementation of MAPE function follows sklearn/metrics regression metrics
    Parameters
    ----------
        y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
            Ground truth (correct) target values.

        y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
            Estimated target values.
    '''
    return K.mean(K.abs((y_true - y_pred) / (y_true+ K.epsilon()))) * 100

def MAPE_Other(y_true, y_pred, multioutput='uniform_average'):
    '''
    This function is implemented by: Ahmed Fathalla <a.fathalla@science.suez.edu.eg>,
         The implementation of MAPE function follows sklearn/metrics regression metrics
    Parameters
    ----------
        y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)
            Ground truth (correct) target values.

        y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)
            Estimated target values.
    '''
    _, y_true, y_pred, _ = _check_reg_targets(y_true, y_pred, multioutput)
    y_true[y_true == 0.0] =  0.000001
    assert not(0.0 in y_true), 'MAPE arrises an Error, cannot calculate MAPE while y_true has 0 element(s). Check \"utils\_scoring_metrics.MAPE"'
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def POCID(y_true, y_pred):
    # print('-------- 2')
    df = pd.DataFrame(np.hstack([y_true.reshape(-1, 1), y_pred.reshape(-1, 1)]), columns=['y_true','y_pred']).diff()
    # print('-------- 3')
    df['total'] = df['y_true']*df['y_pred']
    def a(x):return 1 if x > 0 else 0
    df['total'] = df['total'].apply(a)
    return (100*df['total'].sum())/(df.shape[0]-1)

# def POCID_orange(y_true, y_pred):
#     """
#     Prediction on change of direction
#         Function implementation: https://github.com/biolab/orange3-timeseries/blob/master/orangecontrib/timeseries/functions.py
#                                  https://orange3-timeseries.readthedocs.io/en/latest/reference.html#functions.pocid
#     """
#     nobs = len(y_pred)
#     print('nobs = ', nobs)
#     # print('y_true.len ----------', len(y_true))
#     # print('np.diff(y_true[-nobs:].len ----------', len(np.diff(y_true[-nobs:])))
#     # print('np.diff(y_pred).len ----------', len(np.diff(y_pred)))
#     return 100 * np.mean((np.diff(y_true[-nobs:]) * np.diff(y_pred)) > 0)

def u_theil(y_true, y_pred):
    """
    Theil - U of Theil Statistics:
        Function implementation: https://github.com/domingos108/time_series_functions/blob/master/src/time_series_functions.py
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    error_sup = np.square(np.subtract(y_true, y_pred)).sum()
    error_inf = np.square(np.subtract(y_true[:- 1], y_true[1:])).sum()

    return error_sup / error_inf