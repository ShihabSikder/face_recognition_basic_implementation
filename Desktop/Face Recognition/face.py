import numpy as np
import cv2
import pickle


face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_smile.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.read("trainer.yml")
#if you're using mobile cam as webcame then use cap = cv2.VideoCapture(1)
#else use cap = cv2.VideoCapture(0) for default webcam attatched to pc
#run faces_train first
cap = cv2.VideoCapture(1)
labels = {}
#need to invert labes to "Name":id to "id":"Name"
with open("labels.pickle",'rb') as f:
    orig_labels = pickle.load(f)
    labels = {v:k for k,v in orig_labels.items()}

while(True):
    #capture frame
    ret,frame = cap.read()

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)

    # region of interest

    for (x,y,w,h) in faces:
        #print(x,y,w,h)
        roi_gray = gray[y:y+h,x:x+w]
        roi_rgb = frame[y:y+h,x:x+w]

        #recognize
        #using deep learning models keras, sklearn
        
        id,conf = recognizer.predict(roi_gray)
        if conf>=85:
            print(labels[id]," confidence = ",conf)
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id]
            color = (255,255,255)
            strk = 1
            scale = 1
            cv2.putText(frame,name, (x-5,y-5) ,font,scale,color,strk,cv2.LINE_AA)

            eyes = eye_cascade.detectMultiScale(roi_gray)
            for(ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_rgb, (ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            
            smiles = smile_cascade.detectMultiScale(roi_gray)
            for(sx,sy,sw,sh) in smiles:
                cv2.rectangle(roi_rgb, (sx,sy),(sx+sw,sy+sh),(0,255,0),2)
            
        #save the image of only the face
        img_item = "my_img.png"
        cv2.imwrite(img_item,roi_rgb)

        #drawing Rect

        color =(255,0,0) #BGR (not RGB!!! DUde I'm serious)
        stroke = 2
        cv2.rectangle(frame, (x,y),(x+w,y+h),color,stroke)


    #display the result
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()