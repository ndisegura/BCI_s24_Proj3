# -*- coding: utf-8 -*-
"""
@author: Piper Welch, Emily Ertle, Andres Segura
BME 6770: BCI's Project 3
Dr. David Jangraw
4/27/2024

"""

# Inport the necessry modules
import os
import sys
import matplotlib.pyplot as plt
import load_affpac_data as afp
import numpy as np

#Close previosly drawn plots
plt.close('all')

#Build data file string
data_directory='./BNCI_data/'
for subject in [0, 1, 2, 4, 6, 7, 9, 10, 11]:

    #%% Load and plot AffPac data

    channels, information_array, eeg_data, Y_data = afp.load_affpac_data(subject,data_directory)

    #afp.plot_raw_data(data,subject, eeg_data, Y_data, information_array,channels,['Fz','Oz','F4'],[15,80])
    # afp.plot_raw_data(subject, eeg_data, Y_data, information_array,channels,channels_to_plot=['Fz','Oz','F4'])
    #%% Epoch eeg data for normal and frustrated events (Marker==22,23,24,25)

    #Note: This function could be changed to include markers to epoch as an argument
    eeg_epoch_normal, eeg_epoch_frustrated = afp.epoch_eeg_data(eeg_data,Y_data)

    afp.plot_epoch_data(eeg_epoch_normal, eeg_epoch_frustrated, channels, subject, channel_to_plot=['Fz'])