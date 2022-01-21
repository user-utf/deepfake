import cv2
import os
import time
# Image directory
directory = str(os.path.dirname(os.path.realpath(__file__)))
# Change the current directory 
# to specified directory 
os.chdir(directory + r"\other")
cap = cv2.VideoCapture(0) # video capture source camera (Here webcam of laptop) 
for i in range(1000):
    ret,frame = cap.read() # return a single frame in variable `frame`
    # cv2.imshow('img1',frame) #display the captured image
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #grayscale
    frame = cv2.resize(frame, (0,0), fx = 0.2, fy = 0.2)
    cv2.imwrite('c'+str(i)+'.png',frame)
    time.sleep(0.02)
    # cv2.destroyAllWindows()

cap.release()