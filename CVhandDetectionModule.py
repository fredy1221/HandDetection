from __future__ import print_function
from os import listdir
from os.path import isfile, join
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from yolo import load_image_pixels, make_yolov3_model, yolo_decode_boxes, draw_boxes
from weight_reader import WeightReader
from keras.models import load_model
import HandTrackingModule as htm
import time
from skimage.morphology import closing
from skimage.morphology import (square, rectangle, diamond, disk, octagon, star)
from IPython import display
from bokeh.plotting import figure
from bokeh.io import output_notebook, show, push_notebook

global yolov3, class_threshold
output_notebook()

def createMask(imgOriginal,Hmin,Hmax,Smin,Smax,Vmin,Vmax,KernelSize):
  fig=plt.figure(figsize=(12,8), dpi= 100, facecolor='w', edgecolor='k')
  plt.subplot(151).set_title('Original image'),plt.imshow(imgOriginal)

  img = imgOriginal.copy()
  hsvim = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #convert the image to HSV color space
  lower = np.array([Hmin, Smin, Vmin], dtype = "uint8")
  upper = np.array([Hmax, Smax, Vmax], dtype = "uint8")
  skinRegionHSV = cv2.inRange(hsvim, lower, upper) #create a mask using the specified HSV range

  SE =disk(3)
  skinRegionHSV = closing(skinRegionHSV, SE) #fill the gaps using closing
  blurred = cv2.blur(skinRegionHSV, (KernelSize,KernelSize)) #blur the image to prevent small irregularities
  ret,thresh = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY) #create the mas using thresholing
  plt.subplot(152).set_title('mask'),plt.imshow(blurred, cmap=plt.cm.gray)

  imgIso = cv2.bitwise_and(img,img,mask = blurred) #applying the mask on the original image using bitwise
  plt.subplot(153).set_title('Isolated image'),plt.imshow(imgIso)

  contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #find the contours on the isolated image
  contours = max(contours, key=lambda x: cv2.contourArea(x))
  cv2.drawContours(img, [contours], -1, (0,0,255), 2) #draw a line along the contour

  hull = cv2.convexHull(contours) #apply the convex hull on the islated hand
  cv2.drawContours(img, [hull], -1, (0, 255, 255), 2)
  plt.subplot(154).set_title('hull'),plt.imshow(img)

  hull = cv2.convexHull(contours, returnPoints=False)
  defects = cv2.convexityDefects(contours, hull) #check for the defects which are the fingers gaps

  if defects is not None:
    cnt = 0
  for i in range(defects.shape[0]):  # calculate the angle
    s, e, f, d = defects[i][0]
    start = tuple(contours[s][0])
    end = tuple(contours[e][0])
    far = tuple(contours[f][0])
    a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
    b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
    c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
    angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  #      cosine theorem
    if angle <= np.pi / 2:  # angle less than 90 degree, treat as fingers
      cnt += 1
      cv2.circle(img, far, 10, [255, 255, 0], -1)
  if cnt > 0:
    cnt = cnt+1 #because every two fingers have 1 gap
  if cnt>5:
    cnt = 0 #can't have more than three fingers, we have a problem here
    print ("NO NUMBER DETECTED")
  else:
    cv2.putText(img, str(cnt), (10, 50), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0) , 2, cv2.LINE_AA)
    print('Detected number: ',str(cnt))
    plt.subplot(155).set_title('number detection'),plt.imshow(img)
  plt.show()


def DetectAndCropImage(photo_filename, yolov3, class_threshold):
  image, image_w, image_h = load_image_pixels(photo_filename)
  yhat = yolov3.predict(image) # make prediction
  v_boxes, v_labels, v_scores = yolo_decode_boxes(yhat, class_threshold, image_h, image_w) #extract the bonding box from the yolo object
  img = io.imread(photo_filename)

  if (len(v_boxes) != 0):
    maxBox = v_boxes[0]
    for box in v_boxes: #save the biggest box, which is the hand
      y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
      w, h = x2 - x1, y2 - y1
      if w*h > (maxBox.xmax-maxBox.xmin)*(maxBox.ymax-maxBox.ymin):
        maxBox = box
    crop_img = img[maxBox.ymin:maxBox.ymax, maxBox.xmin:maxBox.xmax] #crop the image according to the bounding box
    return crop_img
  else:
    print ("No hand found, choose another picture from the list and wait for results")
    return img

def f(img_path,Hmin,Hmax,Smin,Smax,Vmin,Vmax,KernelSize):
  try:
      imgOriginal = io.imread(img_path)
      createMask(imgOriginal,Hmin,Hmax,Smin,Smax,Vmin,Vmax,KernelSize)
  except:
      print("ERROR")

def g(img_path,save=True):
  try:
    photo_filename = img_path
    yolov3, class_threshold = load_YOLO_model()
    crop_img = DetectAndCropImage(photo_filename, yolov3, class_threshold)
    plt.subplot(121).set_title('Originale Image'),plt.imshow(io.imread(photo_filename))
    plt.subplot(122).set_title('Cropped Image'),plt.imshow(crop_img)
    plt.show()
    if save:
      io.imsave('croppedImage.jpg', crop_img)
      print("image saved")
  except:
      print("TRY ANOTHER IMAGE")

def load_YOLO_model():
  yolov3 = make_yolov3_model() # define the yolo v3 model
  weight_reader = WeightReader('./yolo/yolov3.weights') # load the weights
  weight_reader.load_weights(yolov3) # set the weights
  class_threshold = 0.6 # define the probability threshold for detected objects
  return yolov3, class_threshold

def BGsubstract():
  cap = cv2.VideoCapture(1)
  backSub = cv2.createBackgroundSubtractorMOG2() #call the background separation function
  ret, frame = cap.read()
  frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA) # because Bokeh expects a RGBA image
  frame=cv2.flip(frame, -1) # because Bokeh flips vertically
  width=frame.shape[1]
  height=frame.shape[0]
  p = figure(x_range=(0,width), y_range=(0,height), output_backend="webgl", width=width, height=height)
  myImage = p.image_rgba(image=[frame], x=0, y=0, dw=width, dh=height)
  show(p, notebook_handle=True)
  while True:
    ret, frame = cap.read()
    if ret==False:
      print("CAMERA NOT DETECTED")
      break
    fgMask = backSub.apply(frame)
    frame = cv2.cvtColor(fgMask, cv2.COLOR_BGR2RGBA)
    frame = cv2.flip(frame, -1)
    myImage.data_source.data['image']=[frame]
    push_notebook()
    time.sleep(0.05)

def DPhandDetection():
  wCam,hCam = 640,480
  cap = cv2.VideoCapture(1)
  cap.set(3,wCam)
  cap.set(4,hCam)

  pTime = 0
  detector = htm.handDetector(detectionConfidence=0.65, trackConfidence=0.5) #create the hand detection object
  numberDetected,length = 0,0
  pNumber,cNumber = 0,0
  while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    if success==False:
        print("CAMERA NOT DETECTED")
        break
    img,handSides = detector.findHands(img) #handSide is an array containng the sides of hands in order
    lmlist = detector.findPosition(img,draw=False) #42 elements containg coordinates of hands

    if len(lmlist)!=0:
        lmlist1,lmlist2 = detector.assignLists(lmlist,handSides) #separate the landmarks list in left and right hand
        for handSide in handSides:
            if handSide=='Right':
                length,x1,y1,x2,y2,cx,cy = detector.getbarLength(lmlist2) #return the requested height from the right hand
                detector.drawAllVolumeBars(numberDetected, length, img)
                detector.drawFingersBar(length,x1,y1,x2,y2,cx,cy,img)
            if handSide=='Left':
                cNumber = detector.countFingersUp(lmlist1,hand="Left") #return the detected number
                if cNumber != pNumber:
                    pNumber = cNumber
                    pTime = time.time()
                else:
                    cTime = time.time()
                    if cTime-pTime > 2: #only change the number after two seconds to prevent abrupt movements
                        numberDetected = cNumber
                        pTime =0
                    else:
                        cv2.putText(img,f'hold {int(2-cTime+pTime)} more seconds',(20,340),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),2) #countdown for the user

        cv2.rectangle(img,(20,350),(130,500),(0,255,0),cv2.FILLED)
        cv2.putText(img,str(numberDetected),(45,450),cv2.FONT_HERSHEY_COMPLEX,3,(255,0,0),10) #display the detected number

    cv2.imshow("image",img)
    keyboard = cv2.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break


def WaterShed2():
  wCam,hCam = 640,480
  cap = cv2.VideoCapture(1)
  cap.set(3,wCam)
  cap.set(4,hCam)
  ret, frame = cap.read()

  frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA) # because Bokeh expects a RGBA image
  frame=cv2.flip(frame, -1) # because Bokeh flips vertically
  width=frame.shape[1]
  height=frame.shape[0]
  p = figure(x_range=(0,width), y_range=(0,height), output_backend="webgl", width=width, height=height)
  myImage = p.image_rgba(image=[frame], x=0, y=0, dw=width, dh=height)
  show(p, notebook_handle=True)

  while True:
    ret, img = cap.read()
    if ret==False:
      print("CAMERA NOT DETECTED")
      break
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0

    markers = cv2.watershed(img,markers)
    img[markers == -1] = [255,0,0]
#     img[markers != 3]=[0,0,0]
#     img[markers == 2]=[255,255,255]

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    img = cv2.flip(img, -1)
    myImage.data_source.data['image']=[img]
    push_notebook()
    time.sleep(0.05)


def finalComparison():
  my_frame = pd.DataFrame(data={'Image Type':['white background static','colored background static','open environment static','real time images'],
                                'HSV mask':['YES','SOMETIMES','NO','NO'],
                                'Background separation':['NO','NO','NO','SOMETIMES'],
                                'Watershed':['YES','NO','NO','NO'],
                                'yolov3 isolation + HSV mask':['YES','YES','NO','NO'],
                                'Deep learning':['YES','YES','YES','YES']})
  fig = plt.figure(figsize = (16, 1))
  ax = fig.add_subplot(111)
  ax.table(cellText = my_frame.values,
            rowLabels = my_frame.index,
            colLabels = my_frame.columns,
            loc = "best")
  ax.set_title("Different hand isolation techniques comparaison")
  ax.axis("off");

def WaterShedStatic(img_path='.\FingerImages\env1.jpeg'):
  fig=plt.figure(figsize=(12,8), dpi= 100, facecolor='w', edgecolor='k')
  img = imgOriginal = io.imread(img_path)
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
  print(ret)
  kernel = np.ones((3,3),np.uint8)
  opening1 = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
  sure_bg = cv2.dilate(opening1,kernel,iterations=3)
  dist_transform = cv2.distanceTransform(opening1,cv2.DIST_L2,5)
  ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
  sure_fg = np.uint8(sure_fg)
  unknown = cv2.subtract(sure_bg,sure_fg)
  ret, markers = cv2.connectedComponents(sure_fg)
  markers = markers+1
  markers[unknown==255] = 0
  markers = cv2.watershed(img,markers)

  img[markers == -1] = [255,0,0]
  img[markers != 2]=[0,0,0]
  img[markers == 2]=[255,255,255]

  SE = rectangle(10,9)
  img1 = cv2.morphologyEx(img, cv2.MORPH_CLOSE, SE)
  img2 = cv2.erode(img1,kernel,iterations=5)
  img3 = cv2.medianBlur(img2, 5)
  img4 = img.copy()

  plt.subplot(151).set_title('Isolation using markers'),plt.imshow(img, cmap=plt.cm.gray)
  plt.subplot(152).set_title('Closing'),plt.imshow(img1, cmap=plt.cm.gray)
  plt.subplot(153).set_title('erode'),plt.imshow(img2, cmap=plt.cm.gray)
  plt.subplot(154).set_title('Media Blur'),plt.imshow(img3, cmap=plt.cm.gray)

  img3 = cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)
  contours, hierarchy = cv2.findContours(img3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #find the contours on the isolated image
  contours = max(contours, key=lambda x: cv2.contourArea(x))
  cv2.drawContours(img3, [contours], -1, (0,0,255), 2) #draw a line along the contour

  hull = cv2.convexHull(contours) #apply the convex hull on the isolated hand
  cv2.drawContours(img3, [hull], -1, (0, 255, 255), 2)

  hull = cv2.convexHull(contours, returnPoints=False)
  defects = cv2.convexityDefects(contours, hull) #check for the defects which are the fingers gaps

  if defects is not None:
      cnt = 0
  for i in range(defects.shape[0]):  # calculate the angle
      s, e, f, d = defects[i][0]
      start = tuple(contours[s][0])
      end = tuple(contours[e][0])
      far = tuple(contours[f][0])
      a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
      b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
      c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
      angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  #      cosine theorem
      if angle <= np.pi / 2:  # angle less than 90 degree, treat as fingers
        cnt += 1
        cv2.circle(img3, far, 10, [255, 255, 0], -1)
  if cnt > 0:
      cnt = cnt+1 #because every two fingers have 1 gap
  if cnt>5:
      cnt = 0 #can't have more than three fingers, we have a problem here
      print ("NO NUMBER DETECTED")
  else:
      cv2.putText(img3, str(cnt), (10, 50), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0) , 2, cv2.LINE_AA)
      print('Detected number: ',str(cnt))
  #     plt.subplot(155).set_title('number detection'),plt.imshow(img)

  plt.subplot(155).set_title('number detection'),plt.imshow(img3, cmap=plt.cm.gray)
  plt.show()

