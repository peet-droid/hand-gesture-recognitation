import cv2

import mediapipe as mp
import pandas as pd
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands

hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

Dict  = {}

for i in range(21*2):
        Dict[i] = []



ptime = time.time()
#ctime = time.time()

while True:
        success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        #print(results.multi_hand_landmarks)

        if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                        for id, lm in enumerate(handLms.landmark):
                                #land mark positions in hands
                                h,w,c = img.shape
                                cx, cy = int(lm.x*w), int(lm.y*h)
                                print(id,cx,cy)
                                Dict[2*id+0].append(lm.x)
                                Dict[2*id+1].append(lm.y)
                                cv2.putText(img,str(id),(cx,cy), cv2.FONT_HERSHEY_DUPLEX,0.5,(255,0,255))
                        mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

        ctime = time.time()

        #fps = 1/(ctime-ptime)

        if(ctime-ptime > 15 ):
                break


        #cv2.putText(img,str(int(fps)),(10,70), cv2.FONT_HERSHEY_DUPLEX,3,(255,0,255))

        cv2.imshow("Image",img)

        cv2.waitKey(1)

df = pd.DataFrame(Dict)

df.to_csv("congress.csv")

print(df)
