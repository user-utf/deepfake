import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import numpy as np
import json
import pandas as pd
import tensorflow as tf
import cv2
import time
from tensorflow import keras
from keras import layers, losses
from keras.models import Model
#os.environ['CUDA_VISIBLE_DEVICES'] = "1"
# os.environ['TF_GPU_ALLOCATOR']="cuda_malloc_async"
# if tf.config.list_physical_devices('GPU'):
#     physical_devices = tf.config.list_physical_devices('GPU')
#     tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
#     tf.config.experimental.set_virtual_device_configuration(physical_devices[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4000)])
# with tf.device('/device:GPU:2'):
images = []
x= 0
folder = str(os.path.dirname(os.path.realpath(__file__))) + "\other"
for filename in os.listdir(folder):
    print(x)
    images.append(cv2.cvtColor(cv2.imread(os.path.join(folder, filename)),cv2.COLOR_BGR2GRAY))
    x+=1
    if x>1000:break
# for base, dirs, files in os.walk(str(os.path.dirname(os.path.realpath(__file__)))):
#   x= 0
#   for file in range(100): #len(files)
#     print(x)
#     images.append(cv2.imread(str(os.path.dirname(os.path.realpath(__file__))) + '\henri\c'+str(file)+'.png'))
#     x +=1 
x_train = images[:round(len(images)*7/8)]
# x_train = images
# x_test = x_train
x_test = images[round(len(images)*7/8)+1:]
print("rounded")
# for a in range(len(x_train)): #going through every image
#     print(a)
#     for b in range(len(x_train[a])): # every row in the image
#         for c in range(len(x_train[a][b])): #every pixel
#             x_train[a][b][c] = x_train[a][b][c][0]
#             if a < len(x_test): 
#                 x_test[a][b][c] = x_test[a][b][c][0]
x_train[:] = [x / 255 for x in x_train]
x_test[:] = [x / 255 for x in x_test]
print("/255")
print("processed")
pixel_count = (len(x_test[0])/3)* len(x_test[0][0])
latent_dim = 256 
print(len(x_test[0]), len(x_test[0][0]))
print(x_test[0])
cv2.imshow('Gray image', x_test[0])

cv2.waitKey(0)
cv2.destroyAllWindows()
# x_train.reshape(len(x_train),None,pixel_count)
# x_test.reshape(len(x_test),None,pixel_count)
x_train = np.array(x_train)
x_test = np.array(x_test)
class Autoencoder(Model):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim   
        self.encoder = tf.keras.Sequential([
            #layers.Flatten(),
            layers.Input(shape = (96, 128)),
            layers.Reshape([len(x_test[0])* len(x_test[0][0])]),
            layers.Dense(int(len(x_test[0])* len(x_test[0][0])/2), activation='relu'),
            #layers.Dense(int(len(x_test[0])* len(x_test[0][0])/4), activation='relu'),
            # layers.Reshape([int(len(x_test[0])* len(x_test[0][0])/4)]),
            layers.Dense(400, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(400, activation='relu'),
            #layers.Dense(int(len(x_test[0])* len(x_test[0][0])/4), activation='relu'),
            layers.Dense(int(len(x_test[0])* len(x_test[0][0])/2), activation='relu'),
            layers.Dense(len(x_test[0])* len(x_test[0][0]), activation='relu'),
            layers.Reshape( [96, 128])#   (len(x_test[0]), len(x_test[0][0]))
        ])
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

autoencoder = Autoencoder(latent_dim)
autoencoder.compile(optimizer='adam', loss="mse") #losses.MeanSquaredError()losses.MeanSquaredError() or  mse/mae
autoencoder.fit(x_train, x_train,
            epochs=100,
            shuffle=True,
            batch_size=10,
            validation_data=(x_test, x_test)) 
autoencoder.save("models")
#[None,x_test[0][0],x_test[0][1]] #np.resize(x_train[0],(96, 128))
#print(img[0])
for i in x_test:
    img2 = np.array(autoencoder.call(np.resize(i,(1,96, 128))))
    #cv2.destroyAllWindows()
    cv2.imshow("fake",img2[0])
    cv2.imshow("real",i)
    k = cv2.waitKey(1)
    if k == 27:         #pour quiter le program
        cv2.destroyAllWindows()
        break
    time.sleep(0.02)
    print("pass")
cv2.destroyAllWindows()


# cap = cv2.VideoCapture(0) 
# while True:
#     ret,frame = cap.read() # return a single frame in variable `frame`
#     # cv2.imshow('img1',frame) #display the captured image
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #grayscale
#     frame = cv2.resize(frame, (0,0), fx = 0.2, fy = 0.2)
#     #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #grayscale
#     frame = np.array(frame)
#     frame = (frame/255)
#     img2 = np.array(autoencoder.call(np.resize(frame,(1,96, 128))))
#     #cv2.destroyAllWindows()
#     cv2.imshow("image",img2[0])
#     k = cv2.waitKey(1)
#     if k == 27:         #pour quiter le program
#         cv2.destroyAllWindows()
#         break
# cv2.destroyAllWindows()
