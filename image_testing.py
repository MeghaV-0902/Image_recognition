# -*- coding: utf-8 -*-
"""
Created on Sat May  8 14:15:05 2021

@author: Megha
"""

import cv2
import os
import numpy as np
import face_recognition1 as fr

test_img=cv2.imread('C:/Users/Megha/Documents/GitHub/python_projects/face-recognition/test images/test.jpg')
faces_detected, gray_img=fr.faceDetection(test_img)
print("Faces detected: ",faces_detected)

#comment these linkes when ur running code from second time
#faces, faceID=fr.labels_for_training_images("C:/Users/Megha/Pictures/model/training")
#face_recognizer=fr.train_classifier(faces,faceID)
#face_recognizer.write('C:/Users/Megha/Pictures/model/trainingData.yml')


#uncomment these lines when running code for 2nd time

face_recognizer=cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("C:/Users/Megha/Documents/GitHub/python_projects/face-recognition/trainingData.yml")

name={0:"Megha"}#creatind dict containing names for label

for face in faces_detected:
      (x,y,w,h)=face
      roi_gray=gray_img[y:y+h,x:x+h]
      label, confidence=face_recognizer.predict(roi_gray)#predicts label of the image
      fr.draw_rect(test_img,face)
      predicted_name=name[label]

      if(confidence>35):
          continue
      fr.put_text(test_img, predicted_name,x,y)
      print("Confidence:",confidence)
      print('Label',label)
      resized_img=cv2.resize(test_img,(500,700))
      cv2.imshow('Face Detection', resized_img)
      cv2.waitKey(0)
      cv2.destroyAllWindows
