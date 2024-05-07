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



def load_affpac_data(subject_id=0, data_path = 'data/'): 
    if subject_id < 10: data = loadmat.loadmat(f'{data_path}/S0{subject_id}')
    else: data = loadmat.loadmat(f'{data_path}/S{subject_id}')

    return data['chann'][:32], data['I'], data['X'][:32]*1e6, data['Y']


def plot_raw_data(subject, eeg_data, Y_data, information_array, channels, channels_to_plot,start_stop_time=[0,-1], is_plot_y=False, is_plot_epochs=True):
    
    eeg_time=information_array[0]
    sampling_period=eeg_time[1]-eeg_time[0]
    start_index=int(start_stop_time[0]/sampling_period) #compute integer index for start time
    stop_index=int(start_stop_time[1]/sampling_period) #compute integer index for start time

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
                axs[0].plot([eeg_time[normal_epoch_index_start[trial]], eeg_time[normal_epoch_index_end[trial]]],[0]*2, color = "blue", label = "Normal")
            else: 
                axs[0].plot([eeg_time[normal_epoch_index_start[trial]], eeg_time[normal_epoch_index_end[trial]]],[0]*2, color = "blue")

            axs[0].scatter(eeg_time[normal_epoch_index_end[trial]], [0], color = 'blue')
            axs[0].scatter(eeg_time[normal_epoch_index_start[trial]], [0], color='blue')
        for trial in range(len(frustrated_epoch_index_end)):
            if trial == 0: 
                axs[0].plot([eeg_time[frustrated_epoch_index_start[trial]], eeg_time[frustrated_epoch_index_end[trial]]],[0]*2, color = "red", label = "Frustrated")
            else:
                axs[0].plot([eeg_time[frustrated_epoch_index_start[trial]], eeg_time[frustrated_epoch_index_end[trial]]],[0]*2, color = "red")
            axs[0].scatter(eeg_time[frustrated_epoch_index_end[trial]], [0], color = 'red')
            axs[0].scatter(eeg_time[frustrated_epoch_index_start[trial]], [0], color='red')

    if is_plot_y: 
        axs[0].plot(eeg_time[start_index:stop_index],np.squeeze(Y_data[start_index:stop_index]),label='Marker Events')
    axs[0].set_ylabel('Marker Event')
    axs[0].set_xlabel('Time (s)')
    axs[0].grid()
    axs[0].legend()

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
    plt.savefig(f"plots/subject_{subject}_raw_data")
    plt.close()
 
    
def epoch_eeg_data(eeg_data, Y_data, information_array):
    
    #We want to epoch the data for two particular events: Normal (Code 22/23)
    #and frustration (Code 24/25)
    #Determine indexes of start and end of epoch for normal
    normal_epoch_index_start = np.where(Y_data == 22)[0]
    normal_epoch_index_end = np.where(Y_data == 23)[0]

    frustrated_epoch_index_start = np.where(Y_data == 24)[0]
    frustrated_epoch_index_end = np.where(Y_data == 25)[0]

    if normal_epoch_index_start.shape > normal_epoch_index_end.shape: 
        # normal_epoch_index_start = normal_epoch_index_start[:-1]
        normal_epoch_index_end = np.append(normal_epoch_index_end, len(eeg_data[0])-1)
    if frustrated_epoch_index_start.shape > frustrated_epoch_index_end.shape: 
        # frustrated_epoch_index_start = frustrated_epoch_index_start[:-1]
        frustrated_epoch_index_end = np.append(frustrated_epoch_index_end, len(eeg_data[0])-1)
        
    #Determine how many trials there is
    epoch_count_normal = (Y_data == 22).sum()
    epoch_count_frustrated = (Y_data == 24).sum()

    channels_count = 32 #eeg_data.shape[0]
    max_event_length = max(max(normal_epoch_index_end - normal_epoch_index_start), max(frustrated_epoch_index_end - frustrated_epoch_index_start))

    #eeg_epoch_normal = np.full((epoch_count_normal, channels_count, max_event_length), 0)
    eeg_epoch_normal=np.zeros((epoch_count_normal, channels_count, max_event_length))
    #eeg_epoch_frustrated = np.full((epoch_count_frustrated, channels_count, max_event_length), 0)
    eeg_epoch_frustrated = np.zeros((epoch_count_frustrated, channels_count, max_event_length))
    
    # Create empty boolean masks for each epoch
    normal_epoch_masks = np.zeros((epoch_count_normal, eeg_data.shape[1]))
    frustrated_epoch_masks = np.zeros((epoch_count_frustrated, eeg_data.shape[1]))
    
    for epoch_index in range(epoch_count_normal):
        # Get epoch start and end indices (for code clarity)
        epoch_start_index = normal_epoch_index_start[epoch_index]
        epoch_end_index = normal_epoch_index_end[epoch_index]
        # Find epoch length and add eeg data to the array for the current index
        curr_epoch_length = epoch_end_index - epoch_start_index
        eeg_epoch_normal[epoch_index, :, :curr_epoch_length]  = eeg_data[ :32, epoch_start_index : epoch_end_index]
        # Fill epoch mask with 1s where this epoch occurred
        normal_epoch_masks[epoch_index, epoch_start_index : epoch_end_index] = 1

    for epoch_index in range(epoch_count_frustrated):
        # Get epoch start and end indices (for code clarity)
        epoch_start_index = frustrated_epoch_index_start[epoch_index]
        epoch_end_index = frustrated_epoch_index_end[epoch_index]
        # Find epoch length and add eeg data to the array for the current index
        curr_epoch_length = epoch_end_index - epoch_start_index
        eeg_epoch_frustrated[epoch_index, :, :curr_epoch_length] = eeg_data[ :32, epoch_start_index : epoch_end_index]
        # Fill epoch mask with 1s where this epoch occurred
        frustrated_epoch_masks[epoch_index, epoch_start_index : epoch_end_index] = 1
        
    epoch_time_array = information_array[0, 0 : max_event_length]
    
    return eeg_epoch_normal, eeg_epoch_frustrated, np.array(normal_epoch_masks, dtype=bool), np.array(frustrated_epoch_masks, dtype=bool), epoch_time_array


def plot_epoch_data(eeg_epoch_normal, eeg_epoch_frustrated, epoch_time_array, channels, subject, channel_to_plot, fs = 128): 
    plt.figure()
    
    channel_index = np.where(channels==channel_to_plot)[0][0]

    time = np.linspace(0, eeg_epoch_frustrated.shape[2]*(1/128), eeg_epoch_frustrated.shape[2])
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
    plt.grid(True)
    plt.ylabel("Time (s)")
    plt.savefig(f'plots/subject_{subject}_channel_{channel}_epoch')
    plt.close()
    

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
    plt.title(f"Channel {channel}, Subject {subject} \n {start_stop_time[0]} to {start_stop_time[1]}s After First Button Press")
    plt.savefig(f'plots/subject_{subject}_channel_{channel}_postpress')
    plt.close()


def plot_topographic(subject, eeg_epoch_normal, eeg_epoch_frustrated, channels, channels_to_plot):
    fig, axes = plt.subplots(1, 2, figsize=(6,4))
    axes = axes.flatten()

    channel_indices = np.arange(0, 32, 1)

    channel_data1 = np.nanmedian(eeg_epoch_normal[:,channel_indices,:], axis=0)
    plot_topo.plot_topo(axes = axes[0], channel_names = list(channels[:32]), channel_data=channel_data1, title="Normal")

    channel_data = np.nanmedian(eeg_epoch_frustrated[:,channel_indices,:], axis=0)
    plot_topo.plot_topo(axes = axes[1], channel_names = list(channels[:32]), channel_data=channel_data, title="Frustrated")
    
    plt.suptitle(f"Subject {subject}")
    plt.savefig(f"plots/subject_{subject}_all_data_topo_plot")
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
    eeg_epochs_fft=np.fft.rfft(eeg_epochs/10e6)
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
    #Find the 12Hz trials
    #is_trial_12Hz=is_trial_15Hz==False
    #separate 12Hz and 15Hz epochs
    #eeg_epochs_fft_12Hz=eeg_epochs_fft[is_trial_12Hz]
    #eeg_epochs_fft_15Hz=eeg_epochs_fft[is_trial_15Hz]
    
    #Compute FFT Magnitude from Complex values for 12Hz
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
    # print(len(np.max(eeg_epochs_fft_mean_frustrated,axis=1)[:,np.newaxis]))
    # quit()
    eeg_epochs_fft_normalized_frustrated=eeg_epochs_fft_mean_frustrated/np.max(eeg_epochs_fft_mean_frustrated,axis=1)[:,np.newaxis]
    
    #Compute the FFT power in dB
    eeg_epochs_fft_db_normal= np.log10(eeg_epochs_fft_normalized_normal)
    eeg_epochs_fft_db_frustrated= np.log10(eeg_epochs_fft_normalized_frustrated)
    
    #is_channel_to_plot=channels==any(channels_to_plot)
    
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
    # plt.show()
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
