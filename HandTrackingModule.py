import cv2
import mediapipe as mp
import time
import os
import math
import numpy as np
from google.protobuf.json_format import MessageToDict

class handDetector():
    fingersTipsIDs = {"pinky":20, "ring":16, "middle":12, "index":8, "thumb":4}
    minLenFingers,maxLenFinger = 50,250
    numberDetected = 0
    previous_length=[135,135,135,135,135,135]
    def __init__(self, mode=False, maxHands=2, detectionConfidence=0.5, trackConfidence=0.5):
        self.mode=mode
        self.maxHands=maxHands
        self.detectionConfidence=detectionConfidence
        self.trackConfidence=trackConfidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.detectionConfidence,self.trackConfidence)
        self.mpDraw = mp.solutions.drawing_utils


    def findHands(self, img, draw = True):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #hands object use RGB image
        self.results = self.hands.process(imgRGB)
        handSide=[]
        if self.results.multi_hand_landmarks:
            for idx, hand_handedness in enumerate(self.results.multi_handedness):
                    handedness_dict = MessageToDict(hand_handedness)
                    handSide.append(handedness_dict['classification'][0]['label'])
            if draw:
                for handLms in self.results.multi_hand_landmarks:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        try:
            return img, handSide
        except:
            return img, 'NO HAND DETECTED'

    def findPosition(self, img, handNumber=0, draw=True):
        lmList=[]
        if self.results.multi_hand_landmarks:
            # myHand = self.results.multi_hand_landmarks
            for handLms in self.results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    h,w,c = img.shape
                    cx,cy = int(lm.x*w),int(lm.y*h)
                    lmList.append([id,cx,cy])
                    if draw:
                        cv2.circle(img,(cx,cy),15,(255,255,0),cv2.FILLED)
        return lmList

    def drawAllVolumeBars(self,numberDetected, level,img):
        self.previous_length[numberDetected] = level
        for id in range(0,6):
            self.drawVolumeBar(self.previous_length[id],img,barID=id)

    def getbarLength(self, lmlist, draw=True):
        x1,y1 = lmlist[4][1],lmlist[4][2]
        x2,y2 = lmlist[8][1],lmlist[8][2]
        cx,cy = (x1+x2)//2,(y1+y2)//2
        length = math.hypot(x2-x1,y2-y1)
        return length,x1,y1,x2,y2,cx,cy

    def drawRectangles(self,img):
        for barID in range (0,6):
            x1Rect = 20 + 50*barID
            x2Rect = 55 + 50*barID
            y1Rect = 10
            y2Rect = 260
            cv2.rectangle(img,(x1Rect,y1Rect),(x2Rect,y2Rect),(0,255,0),3)
            cv2.putText(img,'#'+ str(barID),(x1Rect,y2Rect+20),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),2)


    def drawVolumeBar(self, length, img,barID=0):
        x1Rect = 20 + 50*barID
        x2Rect = 55 + 50*barID
        y1Rect = 10
        y2Rect = 260
        volBar = int(np.interp(length,[handDetector.minLenFingers,handDetector.maxLenFinger],[y2Rect,y1Rect]))
        volPercentage = np.interp(length,[handDetector.minLenFingers,handDetector.maxLenFinger],[0,100])
        self.drawRectangles(img)
        cv2.rectangle(img,(x1Rect,volBar),(x2Rect,y2Rect),(0,255,0),cv2.FILLED)
        cv2.putText(img,f'{int(volPercentage)}%',(x1Rect,y2Rect+40),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),2)

    def drawFingersBar(self, length,x1,y1,x2,y2,cx,cy,img):
        if length<50:
            cv2.circle(img,(cx,cy),15,(0,255,0),cv2.FILLED)
        else:
            cv2.circle(img,(x1,y1),15,(255,0,0),cv2.FILLED)
            cv2.circle(img,(x2,y2),15,(255,0,0),cv2.FILLED)
            cv2.circle(img,(cx,cy),15,(255,0,0),cv2.FILLED)
            cv2.line(img,(x1,y1),(x2,y2),(255,0,0),3)


    def checkFingerUp(self,fingerID,lmlist,hand="Left"):
        fingersTipsIDs=handDetector.fingersTipsIDs
        if fingerID != "thumb":
            if lmlist[fingersTipsIDs[fingerID]][2] < lmlist[fingersTipsIDs[fingerID]-2][2]: #Check y coordinate of tip compared to pip
                return 1 #finger is open
            else:
                return 0 #Finger is closed
        elif fingerID == "thumb":
            if hand == "Right": #left hand
                if lmlist[fingersTipsIDs[fingerID]][1] < lmlist[fingersTipsIDs[fingerID]-1][1]: #Check x coordinate of tip compared to ip
                    return 1 #finger is open
                else:
                    return 0 #Finger is closed
            else: #rigth hand
                if lmlist[fingersTipsIDs[fingerID]][1] > lmlist[fingersTipsIDs[fingerID]-1][1]: #Check x coordinate of tip compared to ip
                    return 1 #finger is open
                else:
                    return 0 #Finger is closed

        else:
            return 2 #fingerID is incorrect

    def checkFingersState(self,lmlist,hand="Left"): #array containing 1 for a every finger up, in the same order as the dictionnary
        totalFingersUp = []
        for id in handDetector.fingersTipsIDs: #iterate over the fingers
            totalFingersUp.append(self.checkFingerUp(id,lmlist,hand))
        return totalFingersUp

    def countFingersUp(self,lmlist,hand="Left"): #returns the total number of open fingers
        totalFingersUp = self.checkFingersState(lmlist,hand)
        handDetector.numberDetected = totalFingersUp.count(1)
        return totalFingersUp.count(1)

    def prepareImages(self,folderPath = "FingerImages"):
        myList=os.listdir(folderPath)
        overlayList=[]
        for imPath in myList:
            image = cv2.imread(f'{folderPath}/{imPath}')
            new_image = cv2.resize(image,(200, 300))
            overlayList.append(new_image)
        return overlayList

    def assignLists(self,lmlist,handSides):
        lmlist1,lmlist2=[],[]
        try: #got both hands
            for i in range(0,21):
                lmlist1.append(lmlist[i]) #Left
                lmlist2.append(lmlist[21+i]) #Right
        except:
            if handSides[0] == 'Left':
                lmlist1=lmlist
            elif handSides[0] == 'Right':
                lmlist2=lmlist
        return lmlist1,lmlist2


def main():
    print("CREATED BY FREDERIC CHEMALI")

if __name__ == "__main__":
    main()
