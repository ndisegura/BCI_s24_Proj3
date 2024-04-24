# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 10:03:15 2024

@author: asegura
"""

import scipy.io as sio
import numpy as np

mat_fname='C:/Users/asegura/Downloads/MM05/all_features_simple.mat'

mat_contents = sio.loadmat(mat_fname)


all_features=mat_contents['all_features']
print(all_features.dtype)
"""all_fetures contains the following 4 structures:
    -eeg_features
    -wav_features
    -face_features
    -feature_labels
    -promps
"""


eeg_features=all_features['eeg_features']

eeg_features_thinking=eeg_features[0][0]['thinking_feats']
eeg_features_thinking=eeg_features_thinking[0][0]
eeg_features_clearing=eeg_features[0][0]['clearing_feats']
eeg_features_clearing=eeg_features_clearing[0][0]
eeg_features_stimuli_feats=eeg_features[0][0]['stimuli_feats']
eeg_features_stimuli_feats=eeg_features_stimuli_feats[0][0]
eeg_features_speaking_feats=eeg_features[0][0]['speaking_feats']
eeg_features_speaking_feats=eeg_features_speaking_feats[0][0]


