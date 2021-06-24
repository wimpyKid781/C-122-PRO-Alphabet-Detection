import cv2
import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os, ssl, time

x = np.load("image (1).npz")['arr_0']
y = pd.read_csv("labels.csv")["labels"]
print(pd.Series(y).value_counts())
classes = ['A','B','C','D','E', 'F','G','H','I','J','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
nClasses = len(classes)  

xtrain, xtest, ytrain, ytest = train_test_split(x,y, random_state=9, train_size=7500, test_size=2500)
xtrain_scaled = xtrain/255.0
xtest_scaled = xtest/255.0
clf = LogisticRegression(solver = "saga", multi_class="multinomial").fit(xtrain_scaled,ytrain)
ypred = clf.predict(xtest_scaled) 
accuracy = accuracy_score(ytest, ypred)
print(accuracy)

cap = cv2.VideoCapture(0)
while(True):
    try:
        ret, frame = cap.read()
        gray = cv2.CVTColor(frame,cv2.COLOR_BGR2GRAY)
    #drawing a box in the center of the video
        height , width = gray.shape()
        upper_left = (int(width/2-56),int(height/2-56))
        bottom_right = (int(width/2+56),int(height/2+56))
        cv2.rectangle(gray, upper_left, bottom_right,(0,255,0),2)
    #to only consider the area inside the box for detecting the digit
    #roi = region of interest
        roi = gray[upper_left[1]:bottom_right[1],upper_left[0], bottom_right[0]]
    #converting cv2 image to pil format
        im_pill = Image.fromarray(roi)
    #convert to greyscale image. L format means each pixel is represented by a single value from 0 to 255
        imageBW = im_pill.convert("L")
        imageBW_resized = imageBW.resize((28,28),Image.ANTIALIAS())
    #antialias is to smooth the jagged edges and reduce distortion
        imageBW_resized_inverted = PIL.ImageOps.invert(imageBW_resized)
        pixelFilter = 20
        minpixel = np.percentile(imageBW_resized_inverted, pixelFilter)
        imageBW_resized_inverted_scaled = np.flip(imageBW_resized_inverted-minpixel,0,255)
        maxpixel = np.max(imageBW_resized_inverted)
        imageBW_resized_inverted_scaled = np.asarray(imageBW_resized_inverted_scaled)/maxpixel
        testSample = np.array(imageBW_resized_inverted_scaled.reshape(1,784))
        testPred = clf.predict(testSample)
        print("Predicted Classes =",testPred)
    #display the resulting frame
        cv2.imshow("frame",gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        pass

cap.release()
cap.destroyAllWindows()