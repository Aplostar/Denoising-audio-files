# Convolution denoising autoencoder

# Data Preperation
import numpy as np
from sklearn.preprocessing import MaxAbsScaler
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

import librosa
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
plt.plot(noisy_arr)
plt.show()
plt.plot(clean_arr)
plt.show()
clean_arr = np.reshape(clean_arr,(1895,40,40,1))
noisy_arr = np.reshape(noisy_arr,(1895,40,40,1))

# # Making the model
from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D
from keras.models import Model
from sklearn.model_selection import train_test_split
input_sound = Input(shape =(40,40,1))
x_train,x_test,y_train,y_test = train_test_split(noisy_arr,clean_arr,test_size = 0.3,random_state = 21)
#Encoding Layers
x = Conv2D(32,(3,3),activation = 'relu',padding = 'same')(input_sound)
x = MaxPooling2D((2,2),padding = 'same')(x)
x = Conv2D(16,(3,3),activation = 'relu',padding = 'same')(x)
x = MaxPooling2D((2,2),padding = 'same')(x)
encoded = Conv2D(8,(3,3),activation = 'relu',padding = 'same')(x)

#Decoding Layers
x = Conv2D(16,(3,3),activation = 'relu',padding = 'same')(encoded)
x = UpSampling2D((2,2))(x)
x = Conv2D(32,(3,3),activation = 'relu',padding = 'same')(x)
x = UpSampling2D((2,2))(x)
decoded = Conv2D(1,(3,3),activation = 'relu',padding = 'same')(x)
autoencoder = Model(input_sound,decoded)
autoencoder.summary()

#loss = mse(input_sound,decoded) * 50
autoencoder.compile(optimizer = 'adadelta',loss = 'mean_squared_error')

autoencoder.fit(x_train,
                y_train,              
               batch_size =20,
                epochs = 25,
               shuffle = True,
                validation_split = 0.3
               
               )
