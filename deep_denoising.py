# Deep denoising autoencoder

import numpy as np
from sklearn.preprocessing import MaxAbsScaler
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

import librosa

# Data preperation part

noise = ["airport.wav","babble.wav","birds.wav","car.wav","helloo.wav"]
clean = ["hello.wav"]

cl = []
y,sr = librosa.load("hello.wav",duration = 2.5)
cl.append(y)

noise_arr = []
noise_str = ["airport.wav","babble.wav","birds.wav","car.wav","helloo.wav"]

for i in noise_str:
  y,sr = librosa.load(i,duration = 2.5)
  noise_arr.append(y)

noisy_arr = np.array([])
t = np.arange(-25,30,5)
for i in cl:
  for j in noise_arr:
    for k in t:
      if(k<=0):
        g = i + np.multiply(10**(-1*k/20),j)
        noisy_arr = np.append(noisy_arr,g)
      else:
        g = np.multiply(10**(k/20),i) + j
        noisy_arr = np.append(noisy_arr,g)
cl_arr = 55*cl  
clean_arr = np.array([])
for i in cl_arr:
  clean_arr = np.append(clean_arr,i)

clean_arr = np.array(clean_arr)
noisy_arr = shuffle(noisy_arr,random_state = 42)

def calc_rows(a):
    if a.size%200 == 0:
        return a.size//200
    else:
        return a.size//200 + 1
rows = calc_rows(clean_arr)
clean_arr.resize((rows,200))
noisy_arr.resize((rows,200))

clean_arr = MaxAbsScaler().fit_transform(clean_arr)
noisy_arr = MaxAbsScaler().fit_transform(noisy_arr)

# Making the model
from keras.models import Model
from keras.layers import Dense,Input
from keras import optimizers

input_data = Input(shape = (200,))

x = Dense(100,activation ='relu')(input_data)
x = Dense(50,activation = 'relu')(x)
encoded = Dense(25,activation = 'relu')(x)

x = Dense(50,activation = 'relu')(encoded)
x = Dense(100,activation = 'relu')(x)
decoded = Dense(200,activation = 'relu')(x) 
          
autoencoder = Model(input_data,decoded)
                
adam = optimizers.Adam(lr = 0.01)
autoencoder.compile(optimizer = optimizers.Adam(lr = 0.001),loss = 'mean_squared_error')

autoencoder.fit(noisy_arr,
               clean_arr,
               validation_split = 0.3,
               epochs = 100,
               shuffle = False,
               
               )
'''
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(['train','test'],loc = 'upper left')

'''