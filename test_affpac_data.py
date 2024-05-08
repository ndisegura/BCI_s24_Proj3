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

    channels, information_array, eeg_data, Y_data = afp.load_affpac_data(subject,data_directory)
    fs = int(1/(information_array[0][1]-information_array[0][0]))

    low_cutoff=5
    high_cutoff=45
    filter_type='hann'
    filter_order=1001
    
    filter_coefficients_band_pass=afp.make_bandpass_filter(low_cutoff,high_cutoff,filter_type,filter_order,fs)
    eeg_data=afp.filter_data(eeg_data,filter_coefficients_band_pass)
    time_data, left_finger_correct_epoch, left_finger_LOC_epoch, right_finger_correct_epoch, right_finger_LOC_epoch = afp.individual_press_epoch_eeg_data(eeg_data, Y_data, information_array, fs)
    left_normal_LOC_channels = afp.check_channel_significance(right_finger_correct_epoch, right_finger_LOC_epoch)

    right_normal_LOC_channels = afp.check_channel_significance(right_finger_correct_epoch, right_finger_LOC_epoch)
    left_normal_right_normal_channels =afp.check_channel_significance(right_finger_correct_epoch, left_finger_correct_epoch)


    afp.plot_LOC(subject, time_data, left_finger_correct_epoch, left_finger_LOC_epoch, right_finger_correct_epoch, right_finger_LOC_epoch, channels, left_normal_LOC_channels)
    afp.plot_topographic(subject, left_finger_correct_epoch, left_finger_LOC_epoch, channels, file_name="left_correct_LOC", subtitle_1="Normal", subtitle_2="LOC", suptitle = "Comparing Left Clicks Normal and LOC")
    afp.plot_topographic(subject, right_finger_correct_epoch, right_finger_LOC_epoch, channels, file_name="right_correct_LOC", subtitle_1="Normal", subtitle_2="LOC",suptitle = "Comparing Right Clicks Normal and LOC")
    afp.plot_topographic(subject, left_finger_correct_epoch, right_finger_correct_epoch, channels, file_name="correct_right_left", subtitle_1="Left Normal", subtitle_2="Right Normal",suptitle = "Comparing Normal Left and Right Clicks")
    
    
    afp.plot_raw_data(subject, eeg_data, Y_data, information_array, channels,['Oz'],[0,2000])
    #Epoch eeg data for normal and frustrated events (Marker==22,23,24,25)

    #Note: This function could be changed to include markers to epoch as an argument
    eeg_epoch_normal, eeg_epoch_frustrated, normal_epoch_masks, frustrated_epoch_masks, epoch_time_array = afp.epoch_eeg_data(eeg_data, Y_data, information_array)
    
    afp.plot_epoch_data(eeg_epoch_normal, eeg_epoch_frustrated, epoch_time_array, channels, subject, channel_to_plot=['Oz'])

    afp.plot_topographic(subject, eeg_epoch_normal, eeg_epoch_frustrated, channels, file_name='all_data_topo_plot',subtitle_1="Normal", subtitle_2="LOC", suptitle="Normal vs LOC for all data")
    

    eeg_epochs_fft_normal,fft_frequencies=afp.get_frequency_spectrum(eeg_epoch_normal,fs)
    eeg_epochs_fft_frustrated,fft_frequencies=afp.get_frequency_spectrum(eeg_epoch_frustrated,fs)
    
    afp.plot_after_button_press(eeg_epoch_normal, eeg_epoch_frustrated, normal_epoch_masks, frustrated_epoch_masks, epoch_time_array, Y_data, channels, subject, channel_to_plot=["Fz"], start_stop_time=[-1, 2], presses="first", separate_by_side=False)