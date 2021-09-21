# Hand detection
This repo was part of a personnal project aiming to compare different methods of numbers detections using hand gestures.
After doing so, the objective is to control the height of 5 different bars using only hand gestures.

## Workflow of the detection
1- detect the landmarks on the hands using DL 

2- recognise the number shaped with the fingers of the left hand

3- select a channel using thi specific number

4- adjust the level of this channel using gestures from the right hand

## To run the project
1- download the model and weights that are already trained and put them in the yolo folder

2- run the notebook

## Project structure
### Multi-channels level control using hand gestures.ipynb
The main notebook that can be used to run everything
### CVhandDetectionModule.py
Contains all the functions that will be used to run the notebook
### HandTrackingModule.py
This is the class used to perform the hand tracking, it contains all the detection, drawing, and calculation related to going from the fingers to the channels
### FingerImages folder
Contains images of fingers or hands that will be used as examples in the notebook
### yolo
The weights and model should be added to this folder. The model and weights can be downloaded from: https://drive.google.com/drive/folders/1PYMJFOtdRzhEg5jDG-8DMUiiRiJIKlGp?usp=sharing
### bounding_box.py
Contains functions related to getting the score and labels of a bounding box
### weight_reader.py
Load the weights
### yolo.py
Contains functions relatde to the yolo model
