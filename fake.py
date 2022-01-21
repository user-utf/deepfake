from tensorflow import keras
from keras import layers, losses
from keras.models import Model
import cv2
import numpy as np
import time
import tensorflow as tf
cap = cv2.VideoCapture(0) 
autoencoder =  tf.keras.models.load_model("models")

while True:
    ret,frame = cap.read() # return a single frame in variable `frame`
    # cv2.imshow('img1',frame) #display the captured image
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #grayscale
    frame = cv2.resize(frame, (0,0), fx = 0.2, fy = 0.2)
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #grayscale
    frame = np.array(frame)
    frame = (frame/255)
    img2 = np.array(autoencoder.predict(np.resize(frame,(1,96, 128))))
    #cv2.destroyAllWindows()
    cv2.imshow("fake",img2[0])
    cv2.imshow("real",frame)
    k = cv2.waitKey(1)
    if k == 27:         #pour quiter le program
        cv2.destroyAllWindows()
        break
cv2.destroyAllWindows()