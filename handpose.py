import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
data = pd.read_csv("train.csv")
X = data[data.columns[2:44]]
Y = data["label"]
params_grid = {
    "kernel":["rbf"],"gamma":[1e-4,1e-5],"C":[1,10,100],"coef0":[1,5]
}

grid_model = GridSearchCV(SVC(),params_grid,cv=5)
grid_model.fit(np.array(X),np.array(Y))

import cv2

import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands

hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
label = ["fullpalm","thumbsup"]

while True:
        success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        #print(results.multi_hand_landmarks)
        
        pred = []

        if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                        for id, lm in enumerate(handLms.landmark):
                                h,w,c = img.shape
                                cx, cy = int(lm.x*w), int(lm.y*h)
                                pred.append(lm.x)
                                pred.append(lm.y)
                                cv2.putText(img,str(id),(cx,cy), cv2.FONT_HERSHEY_DUPLEX,0.5,(255,0,255))
                        mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
        
        if(len(pred) == 42):
                cv2.putText(img,str(label[int(grid_model.predict([pred]))-1]),(10,70), cv2.FONT_HERSHEY_DUPLEX,3,(255,0,0))

        #print(label[int(grid_model.predict([pred]))-1])

        #cv2.putText(img,str(label[int(grid_model.predict([pred]))-1]),(10,70), cv2.FONT_HERSHEY_DUPLEX,3,(255,0,255))

        cv2.imshow("Image",img)

        cv2.waitKey(1)
