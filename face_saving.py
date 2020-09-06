# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 16:05:55 2020

@author: Nikhil Kumar
"""
import cv2
import numpy as np

cam = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

face_data = []
dataset_path = "./data/"
filename = input("Enter your name : ")

while True:
    ret, frame = cam.read()
    
    if ret==False:
        continue
    
    grey_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(grey_frame,1.3,5)
    if len(faces)==0:
        continue
    
    faces = sorted(faces,key = lambda x : x[2]*x[3])
    
    x,y,w,h = faces[-1]
    cv2.rectangle(grey_frame,(x,y),(x+w,y+h),(255,255,255),2)
    
    offset = 10
    face_section = grey_frame[y-offset:y+offset+h,x-offset:x+offset+w]
    face_section = cv2.resize(face_section,(100,100))
    face_data.append(face_section)
    
    
    #cv2.imshow("Frame",frame)
    cv2.imshow("Grey Frame",grey_frame)
    
    key_pressed = cv2.waitKey(1) & 0xFF
    
    if key_pressed == ord('q'):
        break
    
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))
np.save(dataset_path+filename+'.npy',face_data)
print("Data saved successfully")

cam.release()
cv2.destroyAllWindows()