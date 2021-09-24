# <--------- Importing relevant libraries --------->
import cv2
import os

#keras uses tensorflow as backend so need to import both
import tensorflow 
from keras.models import load_model

#for display and alarm
import pygame

# <--------- start of code --------->

# Adding alarm to raise alert if driver is sleeping using pygame library
pygame.init()
sound = pygame.mixer.Sound('alarm.wav')

# initialising -> cascade files into cascade classifier to be read with grayscale function later (to find boundaries)

face = cv2.CascadeClassifier('/Users/ankush/Desktop/sleepy-driver-alert/haar-cascade-files/haarcascade_frontalface_alt.xml')
l_eye = cv2.CascadeClassifier('/Users/ankush/Desktop/sleepy-driver-alert/haar-cascade-files/haarcascade_lefteye_2splits.xml')
r_eye = cv2.CascadeClassifier('/Users/ankush/Desktop/sleepy-driver-alert/haar-cascade-files/haarcascade_righteye_2splits.xml')

label = ['Close', 'Open'] #for display

# load model and video capture <--------- basic functions --------->

model = load_model('/Users/ankush/Desktop/sleepy-driver-alert/models/cnnCat2.h5')

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL #display text font

# <----variables---->

#score var to check eye status
score = 0
count = 0
#display rectangle location
dep = 2
# initialising variable for 0/1 val of eye status
rpred = [2]
lpred = [2]

#main >>> infinite loop used to capture each frame till termination

while (True):
    frame = cap.read() #reads each frame
    height, width = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #converts color of captured frame
    #converting image to grayscale for opencv to be able to use them
    faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    left_eye_gray = r_eye.detectMultiScale(roi_gray)
    right_eye_gray = l_eye.detectMultiScale(roi_gray)

    #boundary of prompt
    cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , 2)
                                            
    #draws boundary of face >>>>>[ROI]>>>>>
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

    # Next few lines check if eyes are open or not ->rpred,lpred variables

    # right

    for (x, y, w, h) in right_eye_gray:
        r_eye = frame[y:y + h, x:x + w] #specific roi
        count = count + 1
        r_eye = cv2.cavtColor(r_eye, cv2.COLOR_BGR2GRAY) #color conv
        r_eye = cv2.resize(r_eye, (24, 24)) #24x24 pixel
        r_eye = r_eye / 255 #better convergence val b/w 0 and 1
        r_eye = r_eye.reshape(24, 24, -1)
        r_eye = np.expand_dims(r_eye, axis=0)
        rpred = model.predict_classes(r_eye) #predict_classes() fn returns index value 0 or 1
        if (rpred[0] == 1):  #checking eye status
            label = 'Open'
        if (rpred[0] == 0):
            label = 'Closed'
        break

    # left

    for (x, y, w, h) in left_eye_gray:
        l_eye = frame[y:y + h, x:x + w] #specific roi
        count = count + 1
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
        l_eye = cv2.resize(l_eye, (24, 24))
        l_eye = l_eye / 255
        l_eye = l_eye.reshape(24, 24, -1)
        l_eye = np.expand_dims(l_eye, axis=0)
        lpred = model.predict_classes(l_eye)
        if (lpred[0] == 1):
            label = 'Open'
        if (lpred[0] == 0):
            label = 'Closed'
        break
    
    # <----------- scoring system ----------->
    #these run for every frame in while loop
    
    #closed eyes - both
    if (rpred[0] == 0 and lpred[0] == 0):
        score = score + 1 
        cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
                                            
    # if for some reason only one is closed, driver is not sleeping
    else:
        score = score - 1
        cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    
    # displays score for eye status                                       
    if (score < 0):
        score = 0
        cv2.putText(frame, 'Score:' + str(score), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    # if person is very sleepy, high score ==> eyes closed

    if (score > 15):
        
        cv2.imwrites(os.path.join(path, 'image.jpg'), frame) #saves frame in which driver is sleeping
        
        try: #basic exception handler if sound cannot be played.
            sound.play()
        except:
            pass

        #increases thickness of rectanglular boundary to caution driver                                 
        if (dep < 16):
            dep = dep + 2

        else:
            dep = dep - 2
            if (dep < 2):
                dep = 2
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), dep)

    cv2.imshow('frame', frame) #displays captured frame
    if cv2.waitKey(1) & 0xFF == ord('q'): #displays frame for 1ms and breaks if 'q' ley is pressed - cv2 convention uses q w/ unicode to end while loop
        break

#when capture is stopped and while loop exited
cap.release()
cv2.destroyAllWindows()

# <------------------- End of code ------------------->
