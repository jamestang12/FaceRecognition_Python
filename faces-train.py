import os
import numpy as np
import cv2
import pickle
from PIL import Image


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR,"images")

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()


current_id = 0
label_ids={}
y_labels = []
x_train = []



for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root , file)
            label = os.path.basename(os.path.dirname(path)).replace(" ","-").lower()
            #print(label,path)
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]
           # print(label_ids)
            #y_labels.append(label) # some number
            #x_train.append(path)# verify this image, turn into a NUMPY array, GRAY
            pil_image = Image.open(path).convert("L") # grayscale
            size = (550,550)
            final_image = pil_image.resize(size, Image.ANTIALIAS)
            image_array = np.array(final_image,"uint8") # turn it to a numpy array ,make it a list of number that relate to the image inorder to train the data
            #print(image_array)
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.3, minNeighbors=4)

            for (x,y,w,h) in faces:
                roi = image_array[y:y+h, x:x+w] #reigon of in
                x_train.append(roi)
                y_labels.append(id_)

with open("labels.pickle",'wb') as f:   #f stand for file
   pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainner.yml")
