import cv2
import numpy as np
from PIL import Image
import os
import pyttsx3

def speak(command):
    engine = pyttsx3.init("sapi5")
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)
    engine.setProperty('rate', 180)
    engine.say(command)
    engine.runAndWait()

def face_train():
    # Path for face image database
    path = 'C:/Users/Nares/Desktop/Vision/face recognition/dataset'

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier("C:/Users/Nares/Desktop/Vision/face recognition/haarcascade_frontalface_default.xml")

    # function to get the images and label data
    def getImagesAndLabels(path):

        imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
        faceSamples=[]
        ids = []

        for imagePath in imagePaths:

            PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
            img_numpy = np.array(PIL_img,'uint8')
            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = detector.detectMultiScale(img_numpy)

            for (x,y,w,h) in faces:
                faceSamples.append(img_numpy[y:y+h,x:x+w])
                ids.append(id)

        return faceSamples,ids

    speak("Training faces. It will take a few seconds. Wait ...")
    faces,ids = getImagesAndLabels(path)
    recognizer.train(faces, np.array(ids))

    # Save the model into trainer/trainer.yml
    recognizer.write('C:/Users/Nares/Desktop/Vision/face recognition/trainer/trainer.yml') # recognizer.save() worked on Mac, but not on Pi

    # Print the numer of faces trained and end program
    speak("{0} faces trained. Exiting Program".format(len(np.unique(ids))))