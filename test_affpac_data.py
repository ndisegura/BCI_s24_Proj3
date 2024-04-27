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

#Close previosly drawn plots
plt.close('all')

#Build data file string
data_directory='./BNCI_data/'
subject='00'


#%% Load and plot AffPac data

data = afp.load_affpac_data(subject,data_directory)