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
    Y_data=data_dict['Y'].astype(int)
    
    return data_dict, channels, information_array,information_array,id_labels,markers,eeg_data,Y_data

def plot_raw_data(data,subject, eeg_data, Y_data, information_array,channels, channels_to_plot,start_stop_time=[0,-1]):
    
    
    eeg_time=information_array[0]
    sampling_period=eeg_time[1]-eeg_time[0]
    fs=1/(sampling_period)
    start_index=int(start_stop_time[0]/sampling_period) #compute integer index for start time
    stop_index=int(start_stop_time[1]/sampling_period) #compute integer index for start time
    #eeg_time=np.arange(0,len(eeg[0])*1/fs,1/fs)
    eeg_time=information_array[0]
    
    # event_samples=data['event_samples']
    # event_duration=data['event_durations']
    # event_type=data['event_types']
    
    #is_channel_match=np.zeros(len(eeg_data[0]),dtype=bool)
    
    fig, axs = plt.subplots(2,sharex=True)
    fig.suptitle(f'AffPac Subject {subject} Raw Data')
    
    # #PLot Marker values
    
    
    # for event_index, event_freq in enumerate(event_type):
    #     start_time=eeg_time[event_samples[event_index]]
    #     end_time=eeg_time[event_samples[event_index]+int(event_duration[event_index])]
    axs[0].plot(eeg_time[start_index:stop_index],np.squeeze(Y_data[0][start_index:stop_index]),label='Marker Events')
    axs[0].set_ylabel('Marker Event')
    axs[0].set_xlabel('Time (s)')
    axs[0].grid()
        
    #PLot EEG Data
    for channel_index, channel_member in enumerate(channels_to_plot):
        
        is_channel_match=channels==channel_member #Boolean indexing across rows for item in list
        
        selected_channel_data=eeg_data[is_channel_match]
        
        axs[1].plot(eeg_time[start_index:stop_index], np.squeeze(selected_channel_data[0][start_index:stop_index]),label=channel_member)
    axs[1].set_ylabel('Voltage (uV)')
    axs[1].set_xlabel('Time (s)')
    axs[1].legend()
    axs[1].grid()
    plt.tight_layout()
    
    
def epoch_eeg_data(eeg_data,Y_data):
    
    #We want to epoch the data for two particular events: Normal (Code 22/23)
    #and frustration (Code 24/25)
    eeg_epoch_normal=[]
    eeg_epoch_frustrated=[]
    
    #Determine indexes of start and end of epoch for normal
    normal_epoch_index_start=np.where(Y_data[0]==22)
    normal_epoch_index_end=np.where(Y_data[0]==23)
    frustrated_epoch_index_start=np.where(Y_data[0]==24)
    frustrated_epoch_index_end=np.where(Y_data[0]==25)
    
    #Determine how many trials there is
    epoch_count_normal=(Y_data[0]==22).sum()
    epoch_count_frustrated=(Y_data[0]==24).sum()
    
    #build normal epoch array
    for epoch_index in range(epoch_count_normal-1):
        eeg_epoch_normal.append(eeg_data[:,normal_epoch_index_start[0][epoch_index]:normal_epoch_index_end[0][epoch_index]])
    for epoch_index in range(epoch_count_frustrated-1):
        eeg_epoch_frustrated.append(eeg_data[:,frustrated_epoch_index_start[0][epoch_index]:frustrated_epoch_index_end[0][epoch_index]])
        
    #convert to numpy array before returning. Getting error when doing this because the length of third dimension is not consistent
    #Keepins as python lists for now
    #eeg_epoch_normal=np.array(eeg_epoch_normal,dtype="object")
    #eeg_epoch_frustrated=np.array(eeg_epoch_frustrated)
    
    return eeg_epoch_normal, eeg_epoch_frustrated