# -*- coding: utf-8 -*-
"""


@author: Piper Welch, Emily Ertle, Andres Segura
BME 6770: BCI's Project 3
Dr. David Jangraw
4/27/2024

"""


#from loadmat import loadmat #Not sure where to find this module. I tried uses pip install loadmat and returned an error. Using scipy instead
import scipy.io as sio
import numpy as np


def load_affpac_data(subject,data_directory):
    
    data_file=f'{data_directory}S{subject}.mat'
    
    # Load dictionary
    data_dict = sio.loadmat(data_file)
    #Parse channel information
    channels=[]
    channels_count=len(data_dict['chann'][0])
    for channel_index in range (channels_count):
        channels.append(data_dict['chann'][0][channel_index][0])
        #print( channels[channel_index])
    channels=np.squeeze(channels) #remove object arrays and convert to regular str array
    
    
    return data_dict, channels

def plot_affpac_data(data,subject):
    
    channel=data['chann'][0][:][0]
    information_array=data['I']
    id_labels=data['id_lab']
    markers=data['markers']
    X_data=data['X']
    Y_data=data['Y']

