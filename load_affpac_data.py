# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 17:04:19 2024

@author: Piper Welch, Emily Ertle, Andres Segura
BME 6770: BCI's Project 3
Dr. David Jangraw
4/27/2024

"""


#from loadmat import loadmat #Not sure where to find this module. I tried uses pip install loadmat and returned an error. Using scipy instead
import scipy.io as sio


def load_affpac_data(subject,data_directory):
    
    data_file=f'{data_directory}S{subject}.mat'
    
    # Load dictionary
    data_dict = sio.loadmat(data_file)
    
    return data_dict


