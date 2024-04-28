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
import matplotlib.pyplot as plt


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
    #Parse Information Array
    information_array=data_dict['I']
    #Parse information labels
    id_labels=[]
    labels_count=len(data_dict['id_lab'][0])
    for label_index in range(labels_count):
        
        id_labels.append(data_dict['id_lab'][0][label_index][0])
    id_labels=np.squeeze(id_labels)
    #Parse Markers
    markers=data_dict['markers']
    #Parse EEG data
    eeg_data=data_dict['X']
    Y_data=data_dict['Y']
    
    return data_dict, channels, information_array,information_array,id_labels,markers,eeg_data,Y_data

def plot_raw_data(data,subject, eeg_data, information_array,channels, channels_to_plot):
    
    
    eeg_time=information_array[0]
    
    #channels=data['channels']
    #eeg=data['eeg']
    #fs=data['fs']
    #eeg_time=np.arange(0,len(eeg[0])*1/fs,1/fs)
    eeg_time=information_array[0]
    
    # event_samples=data['event_samples']
    # event_duration=data['event_durations']
    # event_type=data['event_types']
    
    #is_channel_match=np.zeros(len(eeg_data[0]),dtype=bool)
    
    fig, axs = plt.subplots(1,sharex=True)
    fig.suptitle(f'AffPac Subject {subject} Raw Data')
    
    # #PLot Event types
    
    # for event_index, event_freq in enumerate(event_type):
    #     start_time=eeg_time[event_samples[event_index]]
    #     end_time=eeg_time[event_samples[event_index]+int(event_duration[event_index])]
    #     axs[0].plot([start_time,end_time],[event_freq,event_freq], 'b')
    # axs[0].set_ylabel('Flash Frequency')
    # axs[0].set_xlabel('Time (s)')
    # axs[0].grid()
        
    #PLot EEG Data
    for channel_index, channel_member in enumerate(channels_to_plot):
        
        is_channel_match=channels==channel_member #Boolean indexing across rows for item in list
        
        selected_channel_data=eeg_data[is_channel_match]
        
        axs.plot(eeg_time, np.squeeze(selected_channel_data),label=channel_member)
    axs.set_ylabel('Voltage (uV)')
    axs.set_xlabel('Time (s)')
    axs.legend()
    axs.grid()
    plt.tight_layout()
    
    
    pass
