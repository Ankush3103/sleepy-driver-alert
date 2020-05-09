#Importing relevant libraries

from Keras.models import load_model
import numpy as np
import cv2
import os
import time
from pygame import mixer


#Adding alarm to raise alert
Mixer.init()
Sound = mixer.sound('alarm.wav')

#initialising -> cascade files into cascade classifier -> OpenCV

face = cv2.CascadeClassifier('haar-cascade-files\haarcascade_frontalface_alt.xml')
left-eye = cv2.CascadeClassifier('haar-cascade-files\haarcascade_lefteye_2splits.xml')
right-eye = cv2.CascadeClassifier('haar-cascade-files\haarcascade_righteye_2splits.xml')

#start-of-execution
label=['Close','Open']

#load model and video capture

model = load_model('models/cnncat2.h5')
path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count = 0
score = 0
dep = 2
rpred = [99]
lpred = [99]

while(True):
	ret, frame = cap.read()
	height, width = frame.shape[:2]
	
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
	left_eye_gray = right-eye.detectMultiScale(gray)
	right_eye_gray = left-eye.detectMultiScale(gray)

	cv2.rectangle(frame, (0, height-50), (200, height), (0,0,0), thickness = cv2.FILLED)

	for (x,y,w,h) in faces:
		cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )

#Next few lines check if eyes are open or not ->rpred,lpred variables

#right

	for (x,y,w,h) in right_eye_gray:
        	r_eye=frame[y:y+h,x:x+w]
        	count=count+1
        	r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
        	r_eye = cv2.resize(r_eye,(24,24))
        	r_eye= r_eye/255
        	r_eye=  r_eye.reshape(24,24,-1)
        	r_eye = np.expand_dims(r_eye,axis=0)
        	rpred = model.predict_classes(r_eye)
        	if(rpred[0]==1):
          	  label='Open' 
        	if(rpred[0]==0):
           	 label='Closed'
        	break

#left

 	for (x,y,w,h) in left_eye_gray:
        	l_eye=frame[y:y+h,x:x+w]
        	count=count+1
        	l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
        	l_eye = cv2.resize(l_eye,(24,24))
        	l_eye= l_eye/255
        	l_eye=l_eye.reshape(24,24,-1)
        	l_eye = np.expand_dims(l_eye,axis=0)
        	lpred = model.predict_classes(l_eye)
        	if(lpred[0]==1):
           	 label='Open'   
        	if(lpred[0]==0):
           	 label='Closed'
        	break

# if closed eyes detected

	if(rpred[0]==0 and lpred[0]==0):
        	score=score+1
		cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
#if for some reason only one is closed, driver is not sleeping
	else:
		score=score-1
		cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)

	if(score<0):
        	score=0   
		cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)

#if person is very sleepy, high score, eyes closed
	
	if(score>15):
#path to save clicked photo IMPORTANT - add your path
		path = /Users/ankush/desktop
		cv2.imwrites(os.path.join(path,'image.jpg'),frame)
		try
			sound.play()
		except: 
			pass

		if(dep<16):
			dep=dep+2
		else:
			dep=dep-2
			if(dep<2):
				dep=2
		cv2.rectangle(frame,(0,0),(width,height),(0,0,255),dep) 
	cv2.imshow('frame',frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
        	break

cap.release()
cv2.destroyAllWindows()
