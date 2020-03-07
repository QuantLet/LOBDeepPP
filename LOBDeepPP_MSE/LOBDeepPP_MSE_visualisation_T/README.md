[<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/banner.png" width="888" alt="Visit QuantNet">](http://quantlet.de/)

## [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/qloqo.png" alt="Visit QuantNet">](http://quantlet.de/) **LOBDeepPP_MSE_visualisation_T** [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/QN2.png" width="60" alt="Visit QuantNet 2.0">](http://quantlet.de/)

```yaml


Name of Quantlet: LOBDeepPP_MSE_visualisation_T

Published in: ''

Description: 'Joint Limit order book ask and bid price predition for multiple predicton horizons with a neural networks. This script creates plots meant for analysis of the fixed order book level application for training, validation and test dataset.'

Keywords: 'Limit order book, finance, forecasting, prediction, MSE, evaluation, plotting'

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
import matplotlib.pyplot as plt

# %%
model = 'params_N'
path_save = '.'

# %%
errors = dict()
for ds in ['train', 'valid', 'test']:
    with open(os.path.join('../LOBDeepPP_MSE_computation_T',
                           f'preds_T_errors_{ds}.json')) as json_file:
        tmp = json_tricks.load(json_file)
        errors.update({ds: tmp})
# %% Extracting MSE from errors
mse = {k0: {k1: v1['mse'] for k1, v1 in v0.items()}
       for k0, v0 in errors.items()}
# %% MSE for ask and bid for train, test and validation data

ax = plt.subplot(111)
linestyle = {'train': '-', 'test': '--', "valid": '-.'}
for ds, _ in mse.items():
    res = np.empty([0, 3])
    for k, v in sorted(mse[ds].items(),
                       key=lambda k: int(k[0].replace(model, '')[:-1])):
        L = int(k.replace(model, '')[:-1])
        v = v.mean(axis=0)
        tmp = np.array([L, v[0], v[1]]).reshape([1, 3])
        res = np.concatenate((res, tmp), axis=0)
    ax.plot(res[:, 0], list(res[:, 1]**1), c='red',
            linestyle=linestyle[ds], label=f'ask {ds}')
    ax.plot(res[:, 0], list(res[:, 2]**1), c='green',
            linestyle=linestyle[ds], label=f'bid {ds}'
            )
    plt.xlabel('Time lags $T$')
    plt.xticks(res[:, 0])
    plt.ylabel('MSE')
#    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig(f'{path_save}/mse_ol.pdf', bbox_inches='tight', dpi=300)
plt.show()

# %% MSE vs prediction horizon with order levels for ask and bid seperated
for ds in mse.keys():
    ax = plt.subplot(111)
    for k, v in sorted(mse[ds].items(),
                       key=lambda k: int(k[0].replace(model, '')[:-1])):
        lev = int(k.replace(model, '')[:-1])
        ax.plot(range(1, 31), (v.mean(axis=1)), label=lev)
    plt.xlabel('Prediction horizon $h$')
#    plt.ylim([0.005, 0.01])
    plt.ylabel(f'MSE ({ds if ds != "valid" else "validation"})')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
              title='Time lags $T$')
#    plt.ylim([0.0035, 0.02])
    plt.savefig(f'{path_save}/ph_mse_ol_{ds}.pdf',
                bbox_inches='tight', dpi=300)
#    plt.ylim([0.0035, 0.01])
#    plt.savefig(f'{path_save}/ph_mse_ol_{ds}_zoom.pdf',
#                bbox_inches='tight', dpi=300)
    plt.show()

# %% MSE vs prediction horizon with order levels for ask and bid seperated
for ds in mse.keys():
    for bid in [True, False]:
        ax = plt.subplot(111)
        for k, v in sorted(mse[ds].items(),
                           key=lambda k: int(k[0].replace(model, '')[:-1])):
            lev = int(k.replace(model, '')[:-1])
            ax.plot(range(1, 31), (v[:, int(bid)]), label=lev)
        plt.xlabel('Prediction horizon $h$')
        plt.ylabel(f'MSE ({ds if ds != "valid" else "validation"})')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
                  title='Time lags $T$')
        if ds == 'test': 
            plt.ylim([0.0035, 0.014])
        plt.savefig(f'{path_save}/ph_mse_ol_{ds}_{"bid" if bid else "ask"}.pdf',
                    bbox_inches='tight', dpi=300)
#        plt.ylim([0.0035, 0.01])
#        plt.savefig(f'{path_save}/ph_mse_ol_{ds}_{"bid" if bid else "ask"}_zoom.pdf',
#                    bbox_inches='tight', dpi=300)
        plt.show()

```

automatically created on 2020-03-07