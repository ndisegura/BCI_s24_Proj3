# -*- coding: utf-8 -*-
"""


@author: Piper Welch, Emily Ertle, Andres Segura
BME 6770: BCI's Project 3
Dr. David Jangraw
4/27/2024

"""
import numpy as np
import matplotlib.pyplot as plt
import loadmat 
import plot_topo
from pylab import *
from scipy.signal import firwin, filtfilt,freqz,hilbert
from mne.stats import fdr_correction

np.random.seed(0)


def load_affpac_data(subject_id=0, data_path='data/'): 
    """
    Load AFFPAC dataset for a specific subject.

    Parameters:
    - subject_id (int): Subject ID (default is 0).
    - data_path (str): Path to the dataset (default is 'data/').

    Returns:
    - numpy.ndarray: Channel information.
    - numpy.ndarray: Information array.
    - numpy.ndarray: EEG data.
    - numpy.ndarray: Target labels.
    """
    if subject_id < 10:
        data = loadmat.loadmat(f'{data_path}/S0{subject_id}')
    else:
        data = loadmat.loadmat(f'{data_path}/S{subject_id}')

    return data['chann'][:32], data['I'], data['X'][:32]*1e6, data['Y']


def plot_raw_data(subject, eeg_data, Y_data, information_array, channels, channels_to_plot, start_stop_time=[0, -1], is_plot_y=False, is_plot_epochs=True):
    """
    Plot raw EEG data.

    Parameters:
    - subject (int): Subject ID.
    - eeg_data (numpy.ndarray): EEG data.
    - Y_data (numpy.ndarray): Event codes.
    - information_array (numpy.ndarray): Information array.
    - channels (numpy.ndarray): Channel information.
    - channels_to_plot (list): List of channel names to plot.
    - start_stop_time (list): List containing start and stop time for plotting (default is [0, -1]).
    - is_plot_y (bool): Whether to plot event markers (default is False).
    - is_plot_epochs (bool): Whether to plot epoch markers (default is True).

    Saves the plot as a file and closes the plot afterwards.
    """
    eeg_time = information_array[0]
    sampling_period = eeg_time[1] - eeg_time[0]
    start_index = int(start_stop_time[0] / sampling_period)
    stop_index = int(start_stop_time[1] / sampling_period)

    fig, axs = plt.subplots(2, sharex=True)
    fig.suptitle(f'AffPac Subject {subject} Raw Data')

    if is_plot_epochs: 
        normal_epoch_index_start = np.where(Y_data == 22)[0]
        normal_epoch_index_end = np.where(Y_data == 23)[0]

        frustrated_epoch_index_start = np.where(Y_data == 24)[0]
        frustrated_epoch_index_end = np.where(Y_data == 25)[0]
        if normal_epoch_index_start.shape > normal_epoch_index_end.shape: 
            normal_epoch_index_start = normal_epoch_index_start[:-1]
        if frustrated_epoch_index_start.shape > frustrated_epoch_index_end.shape: 
            frustrated_epoch_index_start = frustrated_epoch_index_start[:-1]

        for trial in range(len(normal_epoch_index_end)):
            if trial == 0: 
                axs[0].plot([eeg_time[normal_epoch_index_start[trial]], eeg_time[normal_epoch_index_end[trial]]],[0]*2, color="blue", label="Normal")
            else: 
                axs[0].plot([eeg_time[normal_epoch_index_start[trial]], eeg_time[normal_epoch_index_end[trial]]],[0]*2, color="blue")

            axs[0].scatter(eeg_time[normal_epoch_index_end[trial]], [0], color='blue')
            axs[0].scatter(eeg_time[normal_epoch_index_start[trial]], [0], color='blue')
        for trial in range(len(frustrated_epoch_index_end)):
            if trial == 0: 
                axs[0].plot([eeg_time[frustrated_epoch_index_start[trial]], eeg_time[frustrated_epoch_index_end[trial]]],[0]*2, color="red", label="Frustrated")
            else:
                axs[0].plot([eeg_time[frustrated_epoch_index_start[trial]], eeg_time[frustrated_epoch_index_end[trial]]],[0]*2, color="red")
            axs[0].scatter(eeg_time[frustrated_epoch_index_end[trial]], [0], color='red')
            axs[0].scatter(eeg_time[frustrated_epoch_index_start[trial]], [0], color='red')

    if is_plot_y: 
        axs[0].plot(eeg_time[start_index:stop_index], np.squeeze(Y_data[start_index:stop_index]), label='Marker Events')
    axs[0].set_ylabel('Marker Event')
    axs[0].set_xlabel('Time (s)')
    axs[0].grid()
    axs[0].legend()

    # Plot EEG Data
    for channel_index, channel_member in enumerate(channels_to_plot):
        is_channel_match = channels == channel_member
        selected_channel_data = eeg_data[is_channel_match]
        axs[1].plot(eeg_time[start_index:stop_index], np.squeeze(selected_channel_data[0][start_index:stop_index]), label=channel_member)
   
    axs[1].set_ylabel('Voltage (uV)')
    axs[1].set_xlabel('Time (s)')
    axs[1].legend()
    axs[1].grid()
    plt.xlim(start_stop_time)
    plt.tight_layout()
    plt.savefig(f"plots/subject_{subject}_raw_data")
    plt.close()

 
    
def epoch_eeg_data(eeg_data, Y_data, information_array):
    """
    Epoch EEG data based on event codes.

    Parameters:
    - eeg_data (numpy.ndarray): EEG data.
    - Y_data (numpy.ndarray): Event codes.
    - information_array (numpy.ndarray): Information array.

    Returns:
    - numpy.ndarray: EEG epochs data for normal events.
    - numpy.ndarray: EEG epochs data for frustrated events.
    - numpy.ndarray: Boolean masks indicating the occurrence of normal events.
    - numpy.ndarray: Boolean masks indicating the occurrence of frustrated events.
    - numpy.ndarray: Array containing time points for each epoch.
    """
    # Determine indexes of start and end of epoch for normal events
    normal_epoch_index_start = np.where(Y_data == 22)[0]
    normal_epoch_index_end = np.where(Y_data == 23)[0]

    # Determine indexes of start and end of epoch for frustrated events
    frustrated_epoch_index_start = np.where(Y_data == 24)[0]
    frustrated_epoch_index_end = np.where(Y_data == 25)[0]

    # Adjust the end indexes if they are shorter than start indexes
    if normal_epoch_index_start.shape > normal_epoch_index_end.shape:
        normal_epoch_index_end = np.append(normal_epoch_index_end, len(eeg_data[0])-1)
    if frustrated_epoch_index_start.shape > frustrated_epoch_index_end.shape:
        frustrated_epoch_index_end = np.append(frustrated_epoch_index_end, len(eeg_data[0])-1)

    # Determine the number of trials
    epoch_count_normal = (Y_data == 22).sum()
    epoch_count_frustrated = (Y_data == 24).sum()

    channels_count = 32 
    max_event_length = max(max(normal_epoch_index_end - normal_epoch_index_start), max(frustrated_epoch_index_end - frustrated_epoch_index_start))

    eeg_epoch_normal = np.zeros((epoch_count_normal, channels_count, max_event_length))
    eeg_epoch_frustrated = np.zeros((epoch_count_frustrated, channels_count, max_event_length))
    
    # Create empty boolean masks for each epoch
    normal_epoch_masks = np.zeros((epoch_count_normal, eeg_data.shape[1]))
    frustrated_epoch_masks = np.zeros((epoch_count_frustrated, eeg_data.shape[1]))
    
    for epoch_index in range(epoch_count_normal):
        # Get epoch start and end indices
        epoch_start_index = normal_epoch_index_start[epoch_index]
        epoch_end_index = normal_epoch_index_end[epoch_index]
        # Find epoch length and add EEG data to the array for the current index
        curr_epoch_length = epoch_end_index - epoch_start_index
        eeg_epoch_normal[epoch_index, :, :curr_epoch_length] = eeg_data[:32, epoch_start_index:epoch_end_index]
        # Fill epoch mask with 1s where this epoch occurred
        normal_epoch_masks[epoch_index, epoch_start_index:epoch_end_index] = 1

    for epoch_index in range(epoch_count_frustrated):
        # Get epoch start and end indices
        epoch_start_index = frustrated_epoch_index_start[epoch_index]
        epoch_end_index = frustrated_epoch_index_end[epoch_index]
        # Find epoch length and add EEG data to the array for the current index
        curr_epoch_length = epoch_end_index - epoch_start_index
        eeg_epoch_frustrated[epoch_index, :, :curr_epoch_length] = eeg_data[:32, epoch_start_index:epoch_end_index]
        # Fill epoch mask with 1s where this epoch occurred
        frustrated_epoch_masks[epoch_index, epoch_start_index:epoch_end_index] = 1
        
    epoch_time_array = information_array[0, 0:max_event_length]
    
    return eeg_epoch_normal, eeg_epoch_frustrated, np.array(normal_epoch_masks, dtype=bool), np.array(frustrated_epoch_masks, dtype=bool), epoch_time_array


def plot_epoch_data(eeg_epoch_normal, eeg_epoch_frustrated, epoch_time_array, channels, subject, channel_to_plot, fs=128): 
    """
    Plot EEG epoch data for a specific channel.

    Parameters:
    - eeg_epoch_normal (numpy.ndarray): EEG epoch data for normal condition.
    - eeg_epoch_frustrated (numpy.ndarray): EEG epoch data for frustrated condition.
    - epoch_time_array (numpy.ndarray): Array containing time points for each epoch.
    - channels (numpy.ndarray): Channel information.
    - subject (int): Subject ID.
    - channel_to_plot (str): Name of the channel to plot.
    - fs (int): Sampling frequency (default is 128).

    Saves the plot as a file and closes the plot afterwards.
    """
    plt.figure()
    
    channel_index = np.where(channels == channel_to_plot)[0][0]

    time = np.linspace(0, eeg_epoch_frustrated.shape[2]*(1/fs), eeg_epoch_frustrated.shape[2])
    for trial in range(eeg_epoch_normal.shape[0]):
        if trial == 0: 
            plt.plot(epoch_time_array, eeg_epoch_normal[trial, channel_index, :], color='blue', label='normal', alpha=0.5)
        else:
            plt.plot(epoch_time_array, eeg_epoch_normal[trial, channel_index, :], color='blue', alpha=0.5)

    for trial in range(eeg_epoch_frustrated.shape[0]):
        if trial == 0: 
            plt.plot(epoch_time_array, eeg_epoch_frustrated[trial, channel_index, :], color='red', label='frustrated', alpha=0.5)
        else: 
            plt.plot(epoch_time_array, eeg_epoch_frustrated[trial, channel_index, :], color='red', alpha=0.5)

    plt.legend()
    plt.title(f"Channel {channel_to_plot} Subject {subject}")
    plt.grid(True)
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (uV)")

    plt.savefig(f'plots/subject_{subject}_channel_{channel_to_plot}_epoch')
    plt.close()

    

def plot_after_button_press(eeg_epoch_normal, eeg_epoch_frustrated, normal_epoch_masks, frustrated_epoch_masks, epoch_time_array, Y_data, channels, subject, channel_to_plot, start_stop_time=[-1, 2], presses="first", separate_by_side=False):
    plt.figure()
    
    sampling_period=epoch_time_array[1] - epoch_time_array[0]
    length_s = abs(start_stop_time[1] - start_stop_time[0])
    stop_index=int(length_s/sampling_period)
    button_time_array = epoch_time_array[0:stop_index]
    channel_index = np.where(channels==channel_to_plot)[0][0]
    
    for trial in range(eeg_epoch_normal.shape[0]):
        # Isolate the right subset of the Y_data
        trial_Y_data = Y_data[normal_epoch_masks[trial]]
        
        # Find time of button presses
        button_presses = np.array(np.where(trial_Y_data==1) or np.where(trial_Y_data==2) or np.where(trial_Y_data==3) or np.where(trial_Y_data==4))
        button_presses = button_presses.flatten()
        
        start_index=int(start_stop_time[0]/sampling_period) + button_presses[0] #compute integer index for start time
        stop_index=int(start_stop_time[1]/sampling_period) + button_presses[0] #compute integer index for start time
        
        if start_index < 0:
            stop_index += abs(start_index)
            start_index = 0
        
        if trial==0:
            plt.plot(button_time_array, eeg_epoch_normal[trial,channel_index,start_index:stop_index], color = 'blue', label = 'normal')
        else:
            plt.plot(button_time_array, eeg_epoch_normal[trial,channel_index,start_index:stop_index], color = 'blue')

    for trial in range(eeg_epoch_frustrated.shape[0]):
        # Isolate the right subset of the Y_data
        trial_Y_data = Y_data[frustrated_epoch_masks[trial]]
        
        # Find time of button presses
        button_presses = np.array(np.where(trial_Y_data==1) or np.where(trial_Y_data==2) or np.where(trial_Y_data==3) or np.where(trial_Y_data==4))
        button_presses = button_presses.flatten()
        
        start_index=int(start_stop_time[0]/sampling_period) + button_presses[0] #compute integer index for start time
        stop_index=int(start_stop_time[1]/sampling_period) + button_presses[0] #compute integer index for start time
        
        if start_index < 0:
            stop_index += abs(start_index)
            start_index = 0
        
        if trial==0: 
            plt.plot(button_time_array, eeg_epoch_frustrated[trial,channel_index,start_index:stop_index], color = 'red', label = 'frustrated')
        else: 
            plt.plot(button_time_array, eeg_epoch_frustrated[trial,channel_index,start_index:stop_index], color = 'red')

    plt.legend()
    channel = ''.join(channel_to_plot)
    plt.grid(True)
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (uV)")
    plt.title(f"Channel {channel}, Subject {subject} \n {start_stop_time[0]} to {start_stop_time[1]}s After First Button Press")
    plt.savefig(f'plots/subject_{subject}_channel_{channel}_postpress')
    plt.close()


def plot_topographic(subject, eeg_epoch_normal, eeg_epoch_frustrated, channels, file_name, subtitle_1, subtitle_2, suptitle):
    """
    Plot topographic maps for normal and frustrated EEG epochs.

    Parameters:
    - subject (int): Subject ID.
    - eeg_epoch_normal (numpy.ndarray): EEG epochs data for normal condition.
    - eeg_epoch_frustrated (numpy.ndarray): EEG epochs data for frustrated condition.
    - channels (numpy.ndarray): Channel information.
    - file_name (str): Name of the file to save the plot.
    - subtitle_1 (str): Subtitle for the first subplot.
    - subtitle_2 (str): Subtitle for the second subplot.
    - suptitle (str): Main title for the plot.

    Saves the plot as a file and closes the plot afterwards.
    """
    fig, axes = plt.subplots(1, 2, figsize=(9,5))
    axes = axes.flatten()

    channel_indices = np.arange(0, 32, 1)

    #first topo map 
    channel_data1 = np.nanmedian(eeg_epoch_normal[:,channel_indices,:], axis=0)
    plot_topo.plot_topo(axes=axes[0], channel_names=list(channels[:32]), channel_data=channel_data1, title=subtitle_1)

    # second topo map 
    channel_data = np.nanmedian(eeg_epoch_frustrated[:,channel_indices,:], axis=0)
    plot_topo.plot_topo(axes=axes[1], channel_names=list(channels[:32]), channel_data=channel_data, title=subtitle_2)

    plt.suptitle(f"Subject {subject} \n {suptitle}")
    plt.tight_layout()
    plt.savefig(f"plots/subject_{subject}_{file_name}")
    plt.close()

    

def get_frequency_spectrum(eeg_epochs,fs):
    """
    Function to compute the fast fourier transform of epoch'd eeg data'
    
    Parameters
    ----------
    eeg_epochs : numpy array of floats of size T x C x S where T is the number 
    of event TRIALS, C is the number of channels
    and S is the numnber of EEG samples
 
    fs : integer.Input describing the sampling rate of EEG data in units of samples per second

    Returns
    -------
    eeg_epochs_fft : numpy array of floats of dimension T x C x S where T is the number of event TRIALS, C is the number of channels
    and S is the numnber of FFT points computed from EEG samples. eeg_epochs_fft contains the complex number spectrum of the eeg data
        
    fft_frequencies : numpuy array of float of size (n,) where n is the number of frequency number from 0 (DC) up to the nyquest rate.
        

    """
    
    # Take FFT of signal
    eeg_epochs_fft=np.fft.rfft(eeg_epochs)
    #Compute FFT Magnitude from Complex values
    eeg_epochs_fft_magnitude=np.absolute(eeg_epochs_fft)
    #Compute Frequencies
    n=np.shape(eeg_epochs_fft_magnitude)[2]
    fft_frequencies=np.arange(0,fs/2,(fs/2)/eeg_epochs_fft_magnitude.shape[2])
    if n<len(fft_frequencies):
        fft_frequencies=fft_frequencies[0:-1]
    #fft_frequencies=np.fft.rfftfreq(n,d=1/fs)
    return eeg_epochs_fft,fft_frequencies


def plot_power_spectrum(eeg_epochs_fft_normal,eeg_epochs_fft_frustrated,fft_frequencies,channels,channels_to_plot,subject=1):
    
    eeg_epochs_fft_magnitude_normal=np.absolute(eeg_epochs_fft_normal)
    eeg_epochs_fft_magnitude_frustrated=np.absolute(eeg_epochs_fft_frustrated)
    
    #Compute the power
    #Generate power array
    power_array=np.zeros(eeg_epochs_fft_magnitude_normal.shape)
    power_array=2 #Array of dimension m,n,l with value=2
    #Compute the power by squaring each element
    eeg_epochs_fft_power_normal=np.power(eeg_epochs_fft_magnitude_normal,power_array)
    eeg_epochs_fft_power_frustrated=np.power(eeg_epochs_fft_magnitude_frustrated,power_array)
    #Compute the mean
    eeg_epochs_fft_mean_normal=np.mean(eeg_epochs_fft_power_normal, axis=0)
    eeg_epochs_fft_mean_frustrated=np.mean(eeg_epochs_fft_power_frustrated, axis=0)
    #Normalize to the highest power. Use array broadcasting to handle dimensions mismatch
    eeg_epochs_fft_normalized_normal=eeg_epochs_fft_mean_normal/np.max(eeg_epochs_fft_mean_normal,axis=1)[:,np.newaxis]

    eeg_epochs_fft_normalized_frustrated=eeg_epochs_fft_mean_frustrated/np.max(eeg_epochs_fft_mean_frustrated,axis=1)[:,np.newaxis]
    
    #Compute the FFT power in dB
    eeg_epochs_fft_db_normal= np.log10(eeg_epochs_fft_normalized_normal)
    eeg_epochs_fft_db_frustrated= np.log10(eeg_epochs_fft_normalized_frustrated)
    

    #Plot the spectrum
    plot_count=len(channels_to_plot)
    fig, axs = plt.subplots(plot_count,figsize = (10, 5), sharex=True)
    
    
    for channel_index, channel_name in enumerate(channels_to_plot):
    
       is_channel_to_plot = channels==channel_name
    #    print(fft_frequencies.shape, subject)
       axs[channel_index].plot(fft_frequencies,np.squeeze(eeg_epochs_fft_db_normal[is_channel_to_plot]),label='Normal Trials')
       axs[channel_index].axvline(x=12,linewidth=1, color='b')
       axs[channel_index].plot(fft_frequencies,np.squeeze(eeg_epochs_fft_db_frustrated[is_channel_to_plot]),label='Frustrated Trials')
       axs[channel_index].axvline(x=15,linewidth=1,  color="orange")
       axs[channel_index].set_ylabel('Power (dB)')
       axs[channel_index].set_xlabel('Frequency (Hz)')
       axs[channel_index].set_title(f'Channel {channel_name}')
       axs[channel_index].legend()
       axs[channel_index].grid()
    
    plt.suptitle(f'Frequency Content Subject {subject}')
    plt.tight_layout()
    plt.savefig(f"plots/subject_{subject}_fft_plot")
    plt.close()
    return eeg_epochs_fft_db_normal,eeg_epochs_fft_db_frustrated   


def make_bandpass_filter(low_cutoff,high_cutoff,filter_type='hann',filter_order=10,fs=1000):
    """
    Generate a bandpass FIR filter.

    Parameters:
        low_cutoff (float): Lower cutoff frequency of the filter.
        high_cutoff (float): Higher cutoff frequency of the filter.
        filter_type (str, optional): Type of window to use in FIR filter design. Defaults to 'hann'.
        filter_order (int, optional): Order of the FIR filter. Defaults to 10.
        fs (int, optional): Sampling frequency in Hz. Defaults to 1000.

    Returns:
        array: Coefficients of the FIR filter.
    """
    
    if filter_type==None: filter_type='hann'
    fNQ = fs/2                                     #Compute the Niqyst rate
    taps_number = filter_order                     # Define the filter order
    #Wn = [low_cutoff/fNQ ,high_cutoff/fNQ]         # ... and specify the cutoff frequency normalized to Nyquest rate
    Wn = [low_cutoff ,high_cutoff]         # ... and specify the cutoff frequency normalized to Nyquest rate
    #filter_coefficients  = firwin(taps_number, Wn, window=filter_type, pass_zero='bandpass')              # ... build lowpass FIR filter,
    filter_coefficients  = firwin(taps_number, Wn, window=filter_type, pass_zero=False,fs=fs)              # ... build lowpass FIR filter,
    
    w, h = freqz(filter_coefficients)                      #Compute the frequency response
    
    response_time=np.arange(0,filter_order/fs*+1,1/fs)
    
    fig, axs = plt.subplots(2)
    axs[1].set_title('Digital filter frequency response')
    axs[1].plot(w*fNQ/3.1416, 20 * np.log10(abs(h)), 'b')
    axs[1].set_ylabel('Amplitude [dB]', color='b')
    axs[1].set_xlabel('Frequency [Hz]')
    ax2 = axs[1].twinx()
    angles = np.unwrap(np.angle(h))
    ax2.plot(w*fNQ/3.1416, angles, 'g')
    ax2.set_ylabel('Angle (radians)', color='g')
    ax2.grid(True)
    ax2.axis('tight')
    axs[0].set_title('Digital filter impulse response')
    axs[0].plot(response_time,filter_coefficients, 'b')
    axs[0].set_ylabel('Amplitude ', color='b')
    axs[0].set_xlabel('Time [s]')
    axs[0].grid(True)

    plt.tight_layout()

    #save figure to a file
    plt.savefig(f"plots/{filter_type}_filter_{low_cutoff}-{high_cutoff}Hz_order{filter_order}.")
    plt.close()
    
    return filter_coefficients


def filter_data(data,b):
    """
    Filter input data using "b" FIR filter coefficients.
    "a" filter coefficients is always 1 (FIR)

    Parameters:
        data (array): Numpy array containing EEG data.
        b (array): Coefficients of the FIR filter.

    Returns:
        array: Filtered EEG data.
    """
    
    filtered_data=filtfilt(b, a=1, x=data,axis=data.ndim-1)
    
    return filtered_data

   
def individual_press_epoch_eeg_data(eeg_data, Y_data, information_array, fs, epoch_start_time=-0.5, epoch_end_time=1):
    """
    Extracts individual press epochs from EEG data.

    Parameters:
    - eeg_data (numpy.ndarray): EEG data.
    - Y_data (numpy.ndarray): Target labels for each sample.
    - information_array (numpy.ndarray): Information array.
    - fs (int): Sampling frequency.
    - epoch_start_time (float): Start time of each epoch in seconds (default is -0.5).
    - epoch_end_time (float): End time of each epoch in seconds (default is 1).

    Returns:
    - numpy.ndarray: Time data.
    - numpy.ndarray: Epochs data for left finger correct responses.
    - numpy.ndarray: Epochs data for left finger LOC responses.
    - numpy.ndarray: Epochs data for right finger correct responses.
    - numpy.ndarray: Epochs data for right finger LOC responses.
    """
    channels_count = eeg_data.shape[0]
    left_finger_correct = np.where(Y_data == 1)[0]
    left_finger_LOC = np.where(Y_data == 3)[0]
    right_finger_correct = np.where(Y_data == 2)[0]
    right_finger_LOC = np.where(Y_data == 4)[0]

    samples_per_epoch = int(fs * (epoch_end_time - epoch_start_time))

    left_finger_correct_epoch = np.zeros((len(left_finger_correct), channels_count, samples_per_epoch))
    left_finger_LOC_epoch = np.zeros((len(left_finger_LOC), channels_count, samples_per_epoch))
    right_finger_correct_epoch = np.zeros((len(right_finger_correct), channels_count, samples_per_epoch))
    right_finger_LOC_epoch = np.zeros((len(right_finger_LOC), channels_count, samples_per_epoch))

    time_data = np.arange(epoch_start_time, epoch_end_time, 1 / fs)

    for sample_index, sample in enumerate(left_finger_correct):
        epoch_start_index = sample + int(epoch_start_time * fs)
        epoch_end_index = epoch_start_index + samples_per_epoch
        left_finger_correct_epoch[sample_index, :, :] = eeg_data[:, epoch_start_index:epoch_end_index]

    for sample_index, sample in enumerate(right_finger_correct):
        epoch_start_index = sample + int(epoch_start_time * fs)
        epoch_end_index = epoch_start_index + samples_per_epoch
        right_finger_correct_epoch[sample_index, :, :] = eeg_data[:, epoch_start_index:epoch_end_index]

    for sample_index, sample in enumerate(right_finger_LOC):
        epoch_start_index = sample + int(epoch_start_time * fs)
        epoch_end_index = epoch_start_index + samples_per_epoch
        right_finger_LOC_epoch[sample_index, :, :] = eeg_data[:, epoch_start_index:epoch_end_index]

    for sample_index, sample in enumerate(left_finger_LOC):
        epoch_start_index = sample + int(epoch_start_time * fs)
        epoch_end_index = epoch_start_index + samples_per_epoch
        left_finger_LOC_epoch[sample_index, :, :] = eeg_data[:, epoch_start_index:epoch_end_index]

    return time_data, left_finger_correct_epoch, left_finger_LOC_epoch, right_finger_correct_epoch, right_finger_LOC_epoch


def plot_LOC(subject, time_data, left_finger_correct_epoch, left_finger_LOC_epoch, right_finger_correct_epoch, right_finger_LOC_epoch, channels, channels_to_plot):
    """
    Plot comparison of left and right finger responses across normal and LOC conditions.

    Parameters:
    - subject (int): Subject ID.
    - time_data (numpy.ndarray): Time data.
    - left_finger_correct_epoch (numpy.ndarray): Epochs data for left finger correct responses.
    - left_finger_LOC_epoch (numpy.ndarray): Epochs data for left finger LOC responses.
    - right_finger_correct_epoch (numpy.ndarray): Epochs data for right finger correct responses.
    - right_finger_LOC_epoch (numpy.ndarray): Epochs data for right finger LOC responses.
    - channels (list): List of channel names.
    - channels_to_plot (list): List of channel indices to plot.

    Saves the plot as a file and closes the plot afterwards.
    """
    channels_to_plot_count = len(channels_to_plot)
    fig, axs = plt.subplots(2, channels_to_plot_count, figsize=(14, 10), sharex=True, sharey=True)

    # plotting values
    data_sets = [
        (left_finger_correct_epoch, left_finger_LOC_epoch, "Left Finger Responses", "Left", 'blue', 'navy'),
        (right_finger_correct_epoch, right_finger_LOC_epoch, "Right Finger Responses", "Right", 'red', 'maroon')
    ]

    for channel_index, channel_loc in enumerate(channels_to_plot):
        for idx, (correct_epoch, LOC_epoch, title, side, correct_color, LOC_color) in enumerate(data_sets):
            #get p values
            p_vals = calculate_bootstrap_pval(correct_epoch, LOC_epoch)

            axs[idx,channel_index].grid(True)
            correct_se = np.std(correct_epoch, axis=0) / np.sqrt(correct_epoch.shape[0])
            LOC_se = np.std(LOC_epoch, axis=0) / np.sqrt(LOC_epoch.shape[0])

            axs[idx,channel_index].plot(time_data, np.mean(correct_epoch[:,channel_loc,:], axis=0), color=correct_color, label=f"{side} Normal")
            # plot CI normal
            upper_bound = np.mean(correct_epoch[:,channel_loc,:], axis=0) + correct_se[channel_loc,:] * 2
            lower_bound = np.mean(correct_epoch[:,channel_loc,:], axis=0) - correct_se[channel_loc,:] * 2
            axs[idx,channel_index].fill_between(time_data, upper_bound, lower_bound, alpha=0.3, color=correct_color) 

            axs[idx,channel_index].plot(time_data, np.mean(LOC_epoch[:,channel_loc,:], axis=0), color=LOC_color, label=f"{side} LOC")
            # plot CI LOC
            upper_bound = np.mean(LOC_epoch[:,channel_loc,:], axis=0) + LOC_se[channel_loc,:] * 2
            lower_bound = np.mean(LOC_epoch[:,channel_loc,:], axis=0) - LOC_se[channel_loc,:] * 2
            axs[idx,channel_index].fill_between(time_data, upper_bound, lower_bound, alpha=0.3, color=LOC_color) 

            axs[idx,channel_index].set_title(title + f' Subject {subject} Channel {channels[channel_loc]}')
            plotted_first_p_val = False
            # mark locations where bootstrapping leads to a signif. p value 
            for p_val_index, p_val in enumerate(p_vals[channel_loc,:]):
                if p_val < 0.05: 
                    if plotted_first_p_val ==False:
                        axs[idx,channel_index].scatter(time_data[p_val_index], 0,  color="fuchsia", label = "p < 0.05")
                        plotted_first_p_val = True
                    else:
                        axs[idx,channel_index].scatter(time_data[p_val_index], 0,  color="fuchsia")

            axs[idx,channel_index].legend()
            axs[idx,channel_index].axvline(x=0, linestyle='--', color='black', linewidth=1, label="Finger Press")
            axs[idx,0].set_ylabel("Voltage (uV)")

            axs[1,channel_index].set_xlabel("Time (s)")
    # save plot 
    plt.suptitle("Comparison of Left and Right Presses Across Normal and LOC")
    plt.tight_layout()
    plt.savefig(f"plots/subject_{subject}_left_right_normal_LOC_comparison")
    plt.close()


def check_channel_significance(epoch_type1, epoch_type2):
    """
    Checks the significance of channels between two types of epochs.

    Parameters:
    - epoch_type1 (numpy.ndarray): Epochs data for type 1.
    - epoch_type2 (numpy.ndarray): Epochs data for type 2.

    Returns:
    - numpy.ndarray: Indices of channels with the highest significance.
    """
    p_vals = calculate_bootstrap_pval(epoch_type1, epoch_type2)
    p_vals = fdr_correction(p_vals)[0]

    counts = np.sum(p_vals < 0.05, axis=1)

    # Find the index of the dimension with the highest count
    highest_signif_channels = np.argsort(counts)[-3:]
    return highest_signif_channels


def resample_eeg(epoch_data, resample_count):
    '''
    Randomly resample (with replacement) epoched EEG data.
   
    Parameters
    ----------
    epoch_data : array of float, size NxTxM where N is the number of events, 
            T is the number of samples, and M is the number of channels
            Epoched data for every event.
    resample_count : int
        Number of epochs (i.e., events, trials) to include in the resampled data. 

    Returns
    -------
    epoch_resampled : array of float, size RxTxM where R is the number of events 
            corresponding to the resample count, T is the number of samples, 
            and M is the number of channels
            Resampled epoched data. 

    Sources:
        Kramer and Eden (2020) "A Bootstrap Test to Compare ERPs" https://mark-kramer.github.io/Case-Studies-Python/02.html
        
    '''
    trial_index = np.random.randint(epoch_data.shape[0],size=resample_count)
    epoch_resampled = epoch_data[trial_index,:,:]

    return epoch_resampled


def get_bootstrap_erp(epoch_type1, epoch_type2):
    '''
    Calculate the difference between resampled event-related potentials (ERPs), 
    where the trials used to calculate the ERP are from a combined distribution
    that includes both target and nontarget events.

    Parameters
    ----------
    target_epochs : array of float, size NxTxM where N is the number of target events, 
        T is the number of samples, and M is the number of channels
        Epoched data for every target event..
    nontarget_epochs : array of float, size NxTxM where N is the number of nontarget events, 
        T is the number of samples, and M is the number of channels
        Epoched data for every nontarget event.

    Returns
    -------
    bootstrap_difference : array of float, size TxM where T is the number of samples and M is the number of channels
        Difference in voltage of resampled ERPs.

    Sources:
        Kramer and Eden (2020) "A Bootstrap Test to Compare ERPs" https://mark-kramer.github.io/Case-Studies-Python/02.html
        
    '''
    #merge target and nontarget distributions
    all_data = np.concatenate((epoch_type1, epoch_type2), axis = 0)

    type1_trial_count = epoch_type1.shape[0]
    type2_trial_count = epoch_type2.shape[0]
    
    # resample from merged distrubution, keeping same trial and nontrial count as original
    type1_sampled = resample_eeg(all_data, type1_trial_count)
    type2_sampled = resample_eeg(all_data, type2_trial_count)
    
    # calculate resampled ERPs
    type1_erp_samples = np.mean(type1_sampled, axis=0) 
    type2_erp_samples = np.mean(type2_sampled, axis=0) 
    
    # calculate test statistic, the absolute difference between ERPs
    bootstrap_difference = np.abs(type1_erp_samples - type2_erp_samples)

    return bootstrap_difference


def calculate_boostrap_stats(epoch_type1, epoch_type2, iteration_count=3000):
    '''
    Calculate distribution of the test statistic using bootstrapping. The test
    statistic is the difference between the target and nontarget event-related 
    potentials (ERPs). The difference is calculated at every time point.

    Parameters
    ----------
    target_epochs : array of float, size NxTxM where N is the number of target events, 
        T is the number of samples, and M is the number of channels
        Epoched data for every target event..
    nontarget_epochs : array of float, size NxTxM where N is the number of nontarget events, 
        T is the number of samples, and M is the number of channels
        Epoched data for every nontarget event.
    iteration_count : int, optional. The default is 3000.
        Number of bootstrap iterations. 

    Returns
    -------
    erp_diffs : array of float, size ZxTxM where Z is the number of bootstrap iterations,
        T is the number of samples, and M is the number of channels
        Distribution of test statistic values (absolute difference between ERPs).

    Sources:
        Kramer and Eden (2020) "A Bootstrap Test to Compare ERPs" https://mark-kramer.github.io/Case-Studies-Python/02.html
        
    '''
    # set up array for distribution of statistic values

    sample_count = epoch_type1.shape[1]
    channel_count = epoch_type1.shape[2]    

    erp_diffs = np.full((iteration_count,sample_count,channel_count),np.nan)

    for iteration_index in range(iteration_count):
        # Calculat test statistic for resampled datasets
        erp_diffs[iteration_index,:,:] = get_bootstrap_erp(epoch_type1, epoch_type2)

    return erp_diffs


def calculate_bootstrap_pval(epoch_type1, epoch_type2, iteration_count=50):
    '''
    Calculate bootstrap p values for the difference between target and nontarget
    event-related potentials (ERPs) at every time point.

    Parameters
    ----------
    target_epochs : array of float, size NxTxM where N is the number of target events, 
        T is the number of samples, and M is the number of channels
        Epoched data for every target event..
    nontarget_epochs : array of float, size NxTxM where N is the number of nontarget events, 
        T is the number of samples, and M is the number of channels
        Epoched data for every nontarget event.
    iteration_count : int, optional. The default is 3000.
        Number of bootstrap iterations.

    Returns
    -------
    p_values : array of float, size TxM where T is the number of samples, and M is the number of channels
        Statistical p value. The probability of the observed difference in ERPs assuming the null hypothesis
        is true. The null hypothesis is that there is no difference between target and nontarget ERPs.
    
    Sources:
        Kramer and Eden (2020) "A Bootstrap Test to Compare ERPs" https://mark-kramer.github.io/Case-Studies-Python/02.html
        
    '''
    # calculate observed erp differences, in real data
    erp_diffs_real = abs(np.mean(epoch_type1, axis=0) - np.mean(epoch_type2, axis=0))
    # calculate bootstrap erp differences
    erp_diffs_distribution = calculate_boostrap_stats(epoch_type1,epoch_type2,iteration_count)

    is_greater_than_dist = np.greater(erp_diffs_real,erp_diffs_distribution) # compare the observed statistic to this distribution of statistic values
    exceed_count = np.sum(is_greater_than_dist,axis=0) # count how many distribution values are exceeded by the observed value
    p_values = 1-exceed_count/iteration_count # 1-(percentage of distribution values that are exceeded by the observed value) (p-value)
    return p_values


def correct_p_values(p_values):
    '''
    Preform a FDR correction on p values 

    Parameters
    ----------
    p_values : array of float, size TxM where T is the number of samples, and M is the number of channels
            Statistical p value. The probability of the observed difference in ERPs assuming the null hypothesis
            is true. The null hypothesis is that there is no difference between target and nontarget ERPs.

    Returns
    -------
    p_values_corrected : array of float, size TxM where T is the number of samples, and M is the number of channels
            Statistical p value corrected for multiple comparisons. The corrected 
            probability of the observed difference in ERPs assuming the null hypothesis
            is true. The null hypothesis is that there is no difference between 
            target and nontarget ERPs.
    '''
    _, p_values_corrected = fdr_correction(p_values)

    return p_values_corrected