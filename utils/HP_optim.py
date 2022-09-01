import tensorflow as tf
# from tensorflow import keras
import traceback, gc
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dropout, Dense, GRU
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
# from keras_adamw import AdamW  # pip install keras-adamw
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import hyperopt, time
from hyperopt import fmin, tpe, STATUS_OK, Trials, STATUS_FAIL
from hyperopt import hp
from tensorflow.keras.metrics import RootMeanSquaredError# as rmse
# from .scoring import RMSE, MAPE

hyperopt_space_ = \
    {
    'nb_epochs' : hp.choice('nb_epochs',[50]),
    'GRU_cells' : hp.choice('GRU_cells',[2,4,8]),
    'd1_nodes' : hp.choice('d1_nodes',[4,8]),
    'd2_nodes' : hp.choice('d2_nodes',[0,4]),
    'batch_size' : hp.choice('batch_size',[16,32]),
    'dropout1': hp.choice('dropout1',[0,0.1,0.2]),
    'dropout2': hp.choice('dropout2',[0,0.1,0.2]),
    'activation_1': hp.choice('activation_1',['relu']),
    'activation_2': hp.choice('activation_2',['relu']),
    'hyperopt_max_trials': 20,
    }
    
def get_model_architecture_str(best_architecture_dict):
    # print(best_architecture_dict)
    s = '\n--------------------------\nGRU:%d \ndenes_1:%d Activation:%s\tDropout:%-.1f \ndenes_2:%d Activation:%s\tDropout:%-.1f \n batch_size:%d'%(
            best_architecture_dict['GRU_cells'],
            best_architecture_dict['d1_nodes'],
            best_architecture_dict['activation_1'],
            best_architecture_dict['dropout1'],
            best_architecture_dict['d2_nodes'],
            best_architecture_dict['activation_2'],
            best_architecture_dict['dropout2'],
            best_architecture_dict['batch_size']
            )
    try:s+='  epoches:%d'%best_architecture_dict['nb_epochs']
    except:pass
    s += '\n--------------------------'
    return s

def get_best_architecture(X_train,y_train,X_valid,y_valid,trails = 1, setup_dict=None, epoches=None):
    # print('-------- 1', setup_dict)
    def create_model_hyperopt(params):
        K.clear_session()
        # print('GRU_cells = ', int(params['GRU_cells']),'---------')
        p_epochs=int(params['nb_epochs'])
        GRU_cells =int(params['GRU_cells'])
        d1_nodes = int(params['d1_nodes'])
        d2_nodes = int(params['d2_nodes'])
        p1_dropout = params['dropout1']
        p2_dropout = params['dropout2']
        act_1 = params['activation_1']
        act_2 = params['activation_2']
        batch_size = int(params['batch_size'])
        
        
        start_time = time.time()

        # use 'try except', in case some parameters are wrong, the script will fail and stop
        try:
            input_layer = Input(shape=( X_train.shape[1], X_train.shape[2]))
            out_model = GRU(GRU_cells, return_sequences=False, )(input_layer)
            
            out_model = Dense(d1_nodes, activation=act_1, name='denes_1')(out_model)
            out_model = Dropout(p1_dropout) (out_model)
            
            if d2_nodes:
                out_model = Dense(d2_nodes, activation=act_2, name='denes_2')(out_model)
                out_model = Dropout(p2_dropout) (out_model)
                
            out_model = Dense(1, activation='linear')(out_model)
            
            model = Model( inputs=[input_layer], outputs = [out_model] )
            model.compile( loss = 'mse', optimizer = 'adam', metrics=['mae'] ) #AdamW()
            
            es = EarlyStopping(monitor='val_loss', mode='min', patience=3)
            history = model.fit(
                                            x = X_train,
                                            y = y_train ,
                                            batch_size = batch_size,
                                            epochs = p_epochs,
                                            verbose = 0,
                                            callbacks = [es],
                                            validation_data=(X_valid, y_valid),
                                            shuffle=False
                                    )

            ind = history.history['val_loss'].index(np.min(history.history['val_loss']))
            val_loss = history.history['val_loss'][ind]

            end_time = time.time()
            execution_time = end_time - start_time

            K.clear_session()
            del model
            gc.collect()

            return {'loss': val_loss,
                    'status': STATUS_OK,
                    'history': history.history,
                    'params': params,
                    'execution_time': execution_time}


        except:
            print('\n**** Create_model_hyperopt Err Hyperopt:\n',traceback.format_exc())
            import sys;sys.exit()
        return {'loss': 0, 'status': STATUS_FAIL}

    if epoches is not None:
        setup_dict['nb_epochs'] = hp.choice('nb_epochs',[epoches])
    search_space = setup_dict
    max_trials = trails if trails is not None else setup_dict['hyperopt_max_trials']
    best = fmin( fn = create_model_hyperopt,
                 space = search_space,
                 algo = tpe.suggest,
                 max_evals = max_trials,
                 trials = Trials())
    best_architecture_dict = hyperopt.space_eval(space = search_space,
                                                 hp_assignment = best)
    return best_architecture_dict, get_model_architecture_str(best_architecture_dict)

from utils.metrics import POCID, u_theil
metrics =['mae','mse']#,POCID, u_theil]

def model_from_dict(best_architecture_dict, shape):
    assert best_architecture_dict is not None, '\nmodel architecture is not initialized yet'
    
    p_epochs=int(best_architecture_dict['nb_epochs'])
    GRU_cells =int(best_architecture_dict['GRU_cells'])
    d1_nodes = int(best_architecture_dict['d1_nodes'])
    d2_nodes = int(best_architecture_dict['d2_nodes'])
    p1_dropout = best_architecture_dict['dropout1']
    p2_dropout = best_architecture_dict['dropout2']
    act_1 = best_architecture_dict['activation_1']
    act_2 = best_architecture_dict['activation_2']
    batch_size = int(best_architecture_dict['batch_size'])
    
    input_layer = Input(shape=( shape[1], shape[2]))
    out_model = GRU(GRU_cells, return_sequences=False )(input_layer)
    
    out_model = Dense(d1_nodes, activation=act_1, name='denes_1')(out_model)
    out_model = Dropout(p1_dropout) (out_model)
    
    if d2_nodes:
        out_model = Dense(d2_nodes, activation=act_2, name='denes_2')(out_model)
        out_model = Dropout(p2_dropout) (out_model)
        
    out_model = Dense(1, activation='linear')(out_model)
    
    model = Model( inputs=[input_layer], outputs = [out_model] )
    model.compile( loss = 'mse',
                   optimizer = 'adam',
                   metrics=metrics )
    return model