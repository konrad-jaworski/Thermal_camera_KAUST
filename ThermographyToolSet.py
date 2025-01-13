import numpy as np
import pandas as pd
import glob
import os
import re
from scipy.signal import savgol_filter
from scipy.fft import fft
from sklearn.decomposition import PCA


def filtering(data,n_sample,poly_order):
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            data[:,i,j] = savgol_filter(data[:,i,j],n_sample,poly_order)
    return data

def ReadingDataToStack(csv_folder):
    csv_files = glob.glob(os.path.join(csv_folder, "*.csv"))

    # Extract the numeric value from the filename using regex
    def extract_number(file_name):
        match = re.search(r'\d+', os.path.basename(file_name))
        return int(match.group()) if match else float('inf')  # Use 'inf' if no number is found

    # Sort files by the numeric value in their names
    csv_files = sorted(csv_files, key=extract_number)

    # Initialize an empty list to store 2D arrays
    frames = []

    # Loop through CSV files and load them as 2D arrays
    for csv_file in csv_files:
        data = np.loadtxt(csv_file, delimiter=",")  # Load CSV file as a NumPy array
        frames.append(data)

    # Stack all 2D arrays into a 3D array
    stacked_array = np.stack(frames, axis=0)
    cropped_frames=stacked_array[:,:,64:640-64]#Squaring an image
    print(f"3D Array shape: {cropped_frames.shape}")

    return cropped_frames

def phasegram(data):
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            data[:,i,j] = np.angle(fft(data[:,i,j]))
    return data

def PCT(data,principal_component_number):
    reshaped_data=data.reshape(data.shape[0],data.shape[1]*data.shape[2])
    reshaped_data=np.transpose(reshaped_data,(1,0))
    pca=PCA(n_components=principal_component_number)
    pca.fit(reshaped_data)
    PCT_data=pca.transform(reshaped_data)
    PCT_data=np.transpose(PCT_data,(1,0))
    PCT_data=PCT_data.reshape(PCT_data.shape[0],data.shape[1],data.shape[2])
    return PCT_data