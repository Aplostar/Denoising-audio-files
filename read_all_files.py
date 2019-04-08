# Reading the clean and noisy data from a directory
import librosa
import librosa.display
import os
import numpy as np

clean_dir = "C:/Users/Arpit Pachauri/Desktop/Clean"
noise_dir = "C:/Users/Arpit Pachauri/Desktop/Noise"
arr = os.listdir(clean_dir)
clean_arr = []
noise_arr = []
for i in arr:
    if ".wav" in i :
        name = clean_dir + "/" + i
        y,sr = librosa.load(name,duration = 2.5)
        clean_arr.append(y)
        
arr = os.listdir(noise_dir)
for i in arr:
    if ".wav" in i :
        name = noise_dir + "/" + i
        y,sr = librosa.load(name,duration = 2.5)
        print(len(y))
        noise_arr.append(y)
        
noisy_arr = np.array([])
t = np.arange(-25,30,5)
for i in clean_arr:
    for j in noise_arr:
        for k in t:
            if(k<=0):
                g = i + np.multiply(10**(-1*k/20),j)
                noisy_arr = np.append(noisy_arr,g)
            else:
                g = np.multiply(10**(k/20),i) + j
                noisy_arr = np.append(noisy_arr,g)
            