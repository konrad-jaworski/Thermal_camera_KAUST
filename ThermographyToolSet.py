import numpy as np
import glob
import os
import re
from scipy.signal import savgol_filter
from scipy.fft import fft
from sklearn.decomposition import PCA
from scipy import stats
from scipy.linalg import svd, inv, eig
from sklearn.preprocessing import MinMaxScaler

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
    phasegram_matrix=np.zeros(((data.shape[0],data.shape[1],data.shape[2])))
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            phasegram_matrix[:,i,j] = np.angle(fft(data[:,i,j]))
    return phasegram_matrix

def PCT(data,principal_component_number):
    reshaped_data=data.reshape(data.shape[0],data.shape[1]*data.shape[2])
    reshaped_data=np.transpose(reshaped_data,(1,0))
    pca=PCA(n_components=principal_component_number)
    pca.fit(reshaped_data)
    PCT_data=pca.transform(reshaped_data)
    PCT_data=np.transpose(PCT_data,(1,0))
    PCT_data=PCT_data.reshape(PCT_data.shape[0],data.shape[1],data.shape[2])
    return PCT_data

def TSR(data,polynomial_order):
    x=np.linspace(1,data.shape[0],data.shape[0])
    x_log=np.log(x)

    coefficient_matrix=np.zeros(((polynomial_order+1,data.shape[1],data.shape[2])))
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            coefficient_matrix[:,i,j]=np.polyfit(x_log,np.log(data[:,i,j]),polynomial_order)

    return coefficient_matrix

def TSR2RGB(data):
    def normalize_channel(channel):
        channel_min = np.min(channel)
        channel_max = np.max(channel)
        normalized = (channel - channel_min) / (channel_max - channel_min) * 255
        return normalized.astype(np.uint8)

    red_channel=normalize_channel(data[0])
    green_channel=normalize_channel(data[1])
    blue_channel=normalize_channel(data[2])

    rgb_image = np.stack((red_channel, green_channel, blue_channel), axis=-1)
    return rgb_image

def HOS(data):
    HOS_matrix=np.zeros(((3,data.shape[1],data.shape[2])))

    skew_image=np.zeros((data.shape[1],data.shape[2]))
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            skew_image[i,j]=stats.skew(data[:,i,j])

    kurtosis_image=np.zeros((data.shape[1],data.shape[2]))
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            kurtosis_image[i,j]=stats.kurtosis(data[:,i,j])

    def fifth_standardized_central_moment(data):
        mean = np.mean(data)
        std_dev = np.std(data)
        standardized_data = (data - mean) / std_dev
        moment_5th = np.mean(standardized_data**5)
        return moment_5th

    fifth_moment=np.zeros((data.shape[1],data.shape[2]))

    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            fifth_moment[i,j]=fifth_standardized_central_moment(data[:,i,j])


    HOS_matrix[0,:,:]=skew_image
    HOS_matrix[1,:,:]=kurtosis_image
    HOS_matrix[2,:,:]=fifth_moment

    return HOS_matrix

def DMD(data,truncation):
    re_thermo=data.reshape(data.shape[0],data.shape[1]*data.shape[2])
    dmd_ready=np.transpose(re_thermo,(1,0))
    scaler=MinMaxScaler()
    dmd_ready=scaler.fit_transform(dmd_ready)

    x=dmd_ready[:,:-1]
    x_p=dmd_ready[:,1:]

    U,s,Vh=svd(x,full_matrices=False)

    sigma=np.zeros((Vh.shape[0],Vh.shape[1]))
    for i in range(Vh.shape[0]):
        sigma[i,i]=s[i]

    U=U[:,:truncation]
    Vh=Vh[:truncation,:]
    sigma=sigma[:truncation,:truncation]

    A_tylda=U.T@x_p@Vh.T@inv(sigma)
    eig_val,eig_vec=eig(A_tylda)

    omega=x_p@Vh.T@inv(sigma)@eig_vec
    omega_tr=np.transpose(omega,(1,0))
    dmd_data=omega_tr.reshape(omega_tr.shape[0],data.shape[1],data.shape[2])

    return dmd_data