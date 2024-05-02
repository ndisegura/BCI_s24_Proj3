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
import loadmat 

def load_affpac_data(subject_id=0, data_path = 'data/'): 
    if subject_id < 10: data = loadmat.loadmat(f'{data_path}/S0{subject_id}')
    else: data = loadmat.loadmat(f'{data_path}/S{subject_id}')
    # print(data.keys())
    # quit()
    return data['chann'], data['I'], data['X']*1e6, data['Y']

def plot_raw_data(subject, eeg_data, Y_data, information_array, channels, channels_to_plot,start_stop_time=[0,-1]):
    
    eeg_time=information_array[0]
    sampling_period=eeg_time[1]-eeg_time[0]
    fs=1/(sampling_period)
    start_index=int(start_stop_time[0]/sampling_period) #compute integer index for start time
    stop_index=int(start_stop_time[1]/sampling_period) #compute integer index for start time

    fig, axs = plt.subplots(2, sharex=True)
    fig.suptitle(f'AffPac Subject {subject} Raw Data')
    
    axs[0].plot(eeg_time[start_index:stop_index],np.squeeze(Y_data[start_index:stop_index]),label='Marker Events')
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
    plt.xlim(start_stop_time)
    plt.tight_layout()
    plt.show()
    
    
def epoch_eeg_data(eeg_data, Y_data, information_array):
    
    #We want to epoch the data for two particular events: Normal (Code 22/23)
    #and frustration (Code 24/25)
    #Determine indexes of start and end of epoch for normal
    normal_epoch_index_start = np.where(Y_data == 22)[0]

    normal_epoch_index_end = np.where(Y_data == 23)[0]
    frustrated_epoch_index_start = np.where(Y_data == 24)[0]
    frustrated_epoch_index_end = np.where(Y_data == 25)[0]

    if normal_epoch_index_start.shape > normal_epoch_index_end.shape: 
        normal_epoch_index_start = normal_epoch_index_start[:-1]
    if frustrated_epoch_index_start.shape > frustrated_epoch_index_end.shape: 
        frustrated_epoch_index_start = frustrated_epoch_index_start[:-1]

    # Determine how many trials there are
    epoch_count_normal = (Y_data == 22).sum()
    epoch_count_frustrated = (Y_data == 24).sum()

    channels_count = eeg_data.shape[0]
    max_event_length = max(max(normal_epoch_index_end - normal_epoch_index_start), max(frustrated_epoch_index_end - frustrated_epoch_index_start))

    eeg_epoch_normal = np.full((epoch_count_normal, channels_count, max_event_length), np.nan)
    eeg_epoch_frustrated = np.full((epoch_count_frustrated, channels_count, max_event_length), np.nan)
    
    # Create empty boolean masks for each epoch
    normal_epoch_masks = np.zeros((epoch_count_normal, eeg_data.shape[1]))
    frustrated_epoch_masks = np.zeros((epoch_count_frustrated, eeg_data.shape[1]))
    
    for epoch_index in range(epoch_count_normal-1):
        # Get epoch start and end indices (for code clarity)
        epoch_start_index = normal_epoch_index_start[epoch_index]
        epoch_end_index = normal_epoch_index_end[epoch_index]
        # Find epoch length and add eeg data to the array for the current index
        curr_epoch_length = epoch_end_index - epoch_start_index
        eeg_epoch_normal[epoch_index, :, :curr_epoch_length] = eeg_data[ :, epoch_start_index : epoch_end_index]
        # Fill epoch mask with 1s where this epoch occurred
        normal_epoch_masks[epoch_index, epoch_start_index : epoch_end_index] = 1

    for epoch_index in range(epoch_count_frustrated-1):
        # Get epoch start and end indices (for code clarity)
        epoch_start_index = frustrated_epoch_index_start[epoch_index]
        epoch_end_index = frustrated_epoch_index_end[epoch_index]
        # Find epoch length and add eeg data to the array for the current index
        curr_epoch_length = epoch_end_index - epoch_start_index
        eeg_epoch_frustrated[epoch_index, :, :curr_epoch_length] = eeg_data[ :, epoch_start_index : epoch_end_index]
        # Fill epoch mask with 1s where this epoch occurred
        frustrated_epoch_masks[epoch_index, epoch_start_index : epoch_end_index] = 1
        
    epoch_time_array = information_array[0, 0 : max_event_length]
    
    return eeg_epoch_normal, eeg_epoch_frustrated, np.array(normal_epoch_masks, dtype=bool), np.array(frustrated_epoch_masks, dtype=bool), epoch_time_array


def plot_epoch_data(eeg_epoch_normal, eeg_epoch_frustrated, epoch_time_array, channels, subject, channel_to_plot): 
    plt.figure()
    
    channel_index = np.where(channels==channel_to_plot)[0][0]
    for trial in range(eeg_epoch_normal.shape[0]):
        if trial==0: 
            plt.plot(epoch_time_array, eeg_epoch_normal[trial,channel_index,:], color = 'blue', label = 'normal')
        else:
            plt.plot(epoch_time_array, eeg_epoch_normal[trial,channel_index,:], color = 'blue')

    for trial in range(eeg_epoch_frustrated.shape[0]):
        if trial==0: 
            plt.plot(epoch_time_array, eeg_epoch_frustrated[trial,channel_index,:], color = 'red', label = 'frustrated')
        else: 
            plt.plot(epoch_time_array, eeg_epoch_frustrated[trial,channel_index,:], color = 'red')

    plt.legend()
    channel = ''.join(channel_to_plot)
    plt.title(f"Channel {channel} Subject {subject}")
    plt.savefig(f'plots/subject-{subject}_channel-{channel}')
    plt.show()
    

def plot_after_button_press(eeg_epoch_normal, eeg_epoch_frustrated, normal_epoch_masks, frustrated_epoch_masks, epoch_time_array, Y_data, channels, subject, channel_to_plot, start_stop_time=[-1, 2], presses="first", separate_by_side=False):
    plt.figure()
    
    sampling_period=epoch_time_array[1]-epoch_time_array[0]
    fs=1/(sampling_period)
    length_s = abs(start_stop_time[1] - start_stop_time[0])
    stop_index=int(length_s/sampling_period)
    button_time_array = epoch_time_array[0:stop_index]
    channel_index = np.where(channels==channel_to_plot)[0][0]
    
    for trial in range(eeg_epoch_normal.shape[0] - 1):
        # Isolate the right subset of the Y_data
        trial_Y_data = Y_data[normal_epoch_masks[trial]]
        
        # Find time of button presses
        button_presses = np.array(np.where(trial_Y_data==1) or np.where(trial_Y_data==2) or np.where(trial_Y_data==3) or np.where(trial_Y_data==4))
        button_presses = button_presses.flatten()
        
        start_index=int(start_stop_time[0]/sampling_period) + button_presses[0] #compute integer index for start time
        stop_index=int(start_stop_time[1]/sampling_period) + button_presses[0] #compute integer index for start time
        time_array = epoch_time_array[start_index:stop_index]
        eeg_data = eeg_epoch_normal[trial,channel_index,start_index:stop_index]
        
        if start_index < 0:
            stop_index += abs(start_index)
            start_index = 0
        
        if trial==0:
            plt.plot(button_time_array, eeg_epoch_normal[trial,channel_index,start_index:stop_index], color = 'blue', label = 'normal')
        else:
            plt.plot(button_time_array, eeg_epoch_normal[trial,channel_index,start_index:stop_index], color = 'blue')

    for trial in range(eeg_epoch_frustrated.shape[0] - 1):
        # Isolate the right subset of the Y_data
        trial_Y_data = Y_data[frustrated_epoch_masks[trial]]
        
        # Find time of button presses
        button_presses = np.array(np.where(trial_Y_data==1) or np.where(trial_Y_data==2) or np.where(trial_Y_data==3) or np.where(trial_Y_data==4))
        button_presses = button_presses.flatten()
        
        start_index=int(start_stop_time[0]/sampling_period) + button_presses[0] #compute integer index for start time
        stop_index=int(start_stop_time[1]/sampling_period) + button_presses[0] #compute integer index for start time
        time_array = epoch_time_array[start_index:stop_index]
        eeg_data = eeg_epoch_normal[trial,channel_index,start_index:stop_index]
        
        if start_index < 0:
            stop_index += abs(start_index)
            start_index = 0
        
        if trial==0: 
            plt.plot(button_time_array, eeg_epoch_frustrated[trial,channel_index,start_index:stop_index], color = 'red', label = 'frustrated')
        else: 
            plt.plot(button_time_array, eeg_epoch_frustrated[trial,channel_index,start_index:stop_index], color = 'red')

    plt.legend()
    channel = ''.join(channel_to_plot)
    plt.grid()
    plt.title(f"Channel {channel} Subject {subject} {start_stop_time[0]} to {start_stop_time[1]}s After First Button Press")
    plt.savefig(f'plots/subject-{subject}_channel-{channel}_postpress')
    plt.show()