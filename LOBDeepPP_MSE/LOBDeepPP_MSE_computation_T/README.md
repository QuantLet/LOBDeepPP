[<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/banner.png" width="888" alt="Visit QuantNet">](http://quantlet.de/)

## [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/qloqo.png" alt="Visit QuantNet">](http://quantlet.de/) **LOBDeepPP_MSE_computation_T** [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/QN2.png" width="60" alt="Visit QuantNet 2.0">](http://quantlet.de/)

```yaml


Name of Quantlet: LOBDeepPP_MSE_computation_T

Published in: ''

Description: 'Joint Limit order book ask and bid price predition for multiple predicton horizons with a neural networks. This script computes the MSE loss for the trained model for the fixed order book level application for training, validation and test dataset.'

Keywords: 'Limit order book, finance, forecasting, prediction, MSE, evaluation'

Author: 'Marius Sterling'

See also: 'LOBDeepPP'

Submitted:  ''


```

### PYTHON Code
```python

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 14:49:33 2019

@author: ms
"""
import os
import json_tricks
import numpy as np
import sys
sys.path.append('/home/ms/github/ob_nw')
from LOB.LOB_keras_train_class import LOB_keras_train_class

# %%
PATH_results = '/home/ms/github/ob_nw/results/innvestigate_analyse/'
PATH_NN = '/home/ms/github/ob_nw/results/nn_model_training/'
MODEL_NAME = 'LOB_keras_model17a'
PATH_NN_MODEL = os.path.join(PATH_NN, MODEL_NAME)
model = 'params_N'
path_save = '.'
# %%

yhats = {i: dict() for i in ['train', 'valid', 'test']}
ys_dict = {}

for f in {'___'.join(f.split('___')[:2]) for f in os.listdir(PATH_NN_MODEL)
          if 'z___' in f and model in f}:
    params, time_start = f.split('___')
    path_nn_model = PATH_NN_MODEL

    f = sorted([i for i in os.listdir(path_nn_model) if (
        params + '___' in i and time_start in i)
        and 'model' in i
        and '.h5' in i],
        key=lambda x:
            int(x.replace(f, '').replace('___model_', '')
                .replace('.h5', '')))[-1]
    f = os.path.join(path_nn_model, f)
    print(f)

    a = LOB_keras_train_class(os.path.basename(path_nn_model))
    a.set_model_params(os.path.join('LOB/params_files', params + '.json'))

    if hasattr(a.model_params, 'set_submodel_weights'):
        del a.model_params['set_submodel_weights']
    a.model_params['load_model'] = f
    a.model_params['tcn']['base']['dropout_rate'] = 0
    a.set_up_model()

    for data_split in ['train', 'valid', 'test']:
        ids = range(0, len(a.gen[data_split]), 1)
        xs = np.concatenate([a.gen[data_split][i][0] for i in ids])
        if 'ys' not in locals() or data_split not in ys_dict:
            ys = np.concatenate([a.gen[data_split][i][1] for i in ids])
            ys_dict.update({data_split: ys})
        yhats[data_split].update({params: a.model.predict(x=xs, verbose=1)})
        del xs, ids
    del ys
    break
for k, v in yhats.items():
    with open(f'{path_save}/preds_T_yhats_{k}.json', 'w') as outfile:
        json_tricks.dump(v, outfile)
for k, v in ys_dict.items():
    with open(f'{path_save}/preds_T_ys_{k}.json', 'w') as outfile:
        json_tricks.dump(v, outfile)

# %%


for data_split in ['train', 'valid', 'test']:
    print(data_split)
    res = {}
    with open(f'{path_save}/preds_T_yhats_{data_split}.json') as json_file:
        yhats = json_tricks.load(json_file)
    with open(f'{path_save}/preds_T_ys_{data_split}.json') as json_file:
        ys = json_tricks.load(json_file)

    for k, v in yhats.items():
        mse = np.mean((v - ys) ** 2, axis=0)
        mde = 1 - np.mean((ys > 0) == (v > 0), axis=0)

        res.update({k: {'mse': mse, 'mde': mde}})
    with open(f'{path_save}/preds_T_errors_{data_split}.json', 'w') as outfile:
        json_tricks.dump(res, outfile)

```

automatically created on 2020-03-07