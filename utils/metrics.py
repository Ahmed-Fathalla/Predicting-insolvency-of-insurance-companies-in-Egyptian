import tensorflow.keras.backend as K

# return K.sqrt(K.mean(K.square(y_pred -y_true), axis=-1))

def POCID(y_true, y_pred):
    """
    Prediction on change of direction
        Function implementation: https://github.com/biolab/orange3-timeseries/blob/master/orangecontrib/timeseries/functions.py
                                 https://orange3-timeseries.readthedocs.io/en/latest/reference.html#functions.pocid
    """
    return 100 * K.mean(((y_true[1:]-y_true[:-1])*
                        (y_pred[1:]-y_pred[:-1]))> 0.0,
                        axis=-1)

def u_theil(y_true, y_pred):
    """
    Theil - U of Theil Statistics:
        Function implementation: https://github.com/domingos108/time_series_functions/blob/master/src/time_series_functions.py
    """
    error_sup = K.sum(K.square(y_true - y_pred))
    error_inf = K.sum(K.square(y_true[:- 1]- y_true[1:]))

    return error_sup / error_inf



import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error as MAE,\
                            mean_squared_error as MSE,\
                            r2_score as R2
from tensorflow.keras.metrics import RootMeanSquaredError# as rmse

def RMSE(y_true, y_pred):
    return np.sqrt(MSE(y_true, y_pred))

def POCID_np(y_true, y_pred):
    df = pd.DataFrame(np.hstack([np.array(y_true).reshape(-1, 1), np.array(y_pred).reshape(-1, 1)]), columns=['y_true','y_pred']).diff()
    df['total'] = df['y_true']*df['y_pred']
    def a(x):return 1 if x > 0 else 0
    df['total'] = df['total'].apply(a)
    return (100*df['total'].sum())/(df.shape[0]-1)

def u_theil_np(y_true, y_pred):
    """
    Theil - U of Theil Statistics:
        Function implementation: https://github.com/domingos108/time_series_functions/blob/master/src/time_series_functions.py
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    error_sup = np.square(np.subtract(y_true, y_pred)).sum()
    error_inf = np.square(np.subtract(y_true[:- 1], y_true[1:])).sum()

    return error_sup / error_inf

# from .metrics import POCID, POCID_np, u_theil, u_theil_np
met_dic = {POCID:POCID_np,
           u_theil:u_theil_np,
           RootMeanSquaredError:RMSE
           }

metric_lst = [MAE, MSE, RMSE, R2]
def get_res(y_true, y_pred, round_=7,metric_lst=metric_lst):
    # y_true = np.array(y_true)
    # y_pred = np.array(y_pred)
    results = []
    for metric in metric_lst:
        if isinstance(metric, str):
            if metric=='mae':metric=MAE
            elif metric=='mse':metric=MSE
            elif metric=='rmse':metric=RMSE
            elif metric=='R2':metric=R2
        elif metric in met_dic.keys():metric=met_dic[metric]
        results.append( round(metric( y_true=y_true, y_pred=y_pred),round_) )
    return results