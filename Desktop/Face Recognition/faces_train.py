import cv2
import os
import numpy as np
from PIL import Image
import pickle

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
# if you get cv2.cv2 has no attr face then follow this link 
# https://stackoverflow.com/questions/44633378/attributeerror-module-cv2-cv2-has-no-attribute-createlbphfacerecognizer

#put pics in image folder including a folder with same person image
#then the belo code will automaticaly trained by the image you've provided

#path of the code in my pc
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR,"image")

y_labels =[]
x_train = []

curr_id = 0
label_ids={}

for root, dirs, file in os.walk(IMG_DIR):
    for file in file:
        if file.endswith("JPG") or file.endswith("jpg"):
            path = os.path.join(root,file)
            #label 
            #"os.path.dirname(path)" instead we can use "root"
            label = os.path.basename(os.path.dirname(path)).replace(" ","-").lower()
            print(label,path)
            if not label in label_ids:
                label_ids[label] = curr_id
                curr_id += 1
            id = label_ids[label]
            print(label_ids)
            #y_labels.append(label) #some number instead
            #x_train.append(path) # verify image, turn into numpy arr, GRAY
            pil_image = Image.open(path).convert("L")#grayscale
            #resize
            size = (550,550)
            final_img = pil_image.resize(size, Image.ANTIALIAS)
            image_array = np.array(pil_image, "uint8")#inserting in numpy array
            #print(len(image_array))
            faces = face_cascade.detectMultiScale(image_array,scaleFactor=1.5,minNeighbors=5)

            for (x,y,w,h) in faces:
                #roi means region of interest
                roi = image_array[y:y+h,x:x+w]
                x_train.append(roi)
                y_labels.append(id)

#print(y_labels)
#print(x_train)

with open("labels.pickle",'wb') as f:
    pickle.dump(label_ids,f)
    #storing label id to a file
recognizer.train(x_train,np.array(y_labels))
recognizer.save("trainer.yml")