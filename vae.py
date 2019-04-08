# Variational autoencoder

from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras import backend as K
from keras.losses import mse
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def sampling(args):
  z_mean, z_log_var = args
  batch = K.shape(z_mean)[0]
  dim = K.int_shape(z_mean)[1]
  
  epsilon = K.random_normal(shape = (batch,dim))
  return z_mean + K.exp(0.5 * z_log_var) * epsilon

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
original_dim = 200
input_shape = (200,)
intermediate_dim = 150
batch_size = 128
latent_dim = 2
epochs = 50

# Vae Model
inputs = Input(shape = input_shape)
x = Dense(intermediate_dim,activation = 'relu')(inputs)
z_mean = Dense(latent_dim)(x)
z_log_var = Dense(latent_dim)(x)

z = Lambda(sampling,output_shape = (latent_dim,))([z_mean, z_log_var])

# Encoder Model
encoder = Model(inputs,[z_mean,z_log_var,z])
encoder.summary()

#Decoder Model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(intermediate_dim, activation='relu')(latent_inputs)
outputs = Dense(original_dim, activation='sigmoid')(x)
decoder = Model(latent_inputs, outputs)
decoder.summary()

outputs = decoder(encoder(inputs)[2])
vae = Model(inputs,outputs)

reconstruction_loss = mse(inputs,outputs)

reconstruction_loss = reconstruction_loss*original_dim
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss,axis = -1)

kl_loss *= -0.5

vae_loss = K.mean(reconstruction_loss + kl_loss)

vae.add_loss(vae_loss)
vae.compile(optimizer = 'adam')

vae.summary()

vae.fit(noisy_arr,
       epochs = epochs,
       batch_size = batch_size)
