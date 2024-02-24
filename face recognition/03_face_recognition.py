import cv2
import numpy as np
import os 
import pyttsx3

def speak(command):
    engine = pyttsx3.init("sapi5")
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)
    engine.setProperty('rate', 180)
    engine.say(command)
    engine.runAndWait()

def face_recognition():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('C:/Users/Nares/Desktop/Vision/face recognition/trainer/trainer.yml')
    cascadePath = "C:/Users/Nares/Desktop/Vision/face recognition/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)

    #iniciate id counter
    id = 0

    # names related to ids: example ==> Marcelo: id=1,  etc
    names = []
    path = 'C:/Users/Nares/Desktop/Vision/face recognition/dataset'
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]  

    for imagepath in imagePaths:
        name = os.path.split(imagepath)[-1].split(".")[2]
        names.append(name)

    # function to get unique values
    x = np.array(names)
    names = (np.unique(x))
    print(names)

    img = cv2.imread("C:/Users/Nares/Desktop/Vision/face recognition/dataset/User.1.naresh.1.jpg")

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        )

    for(x,y,w,h) in faces:

        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        # Check if confidence is less then 100 ==> "0" is perfect match 
        if (confidence < 40):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        print("The person is " + id + " with confidence of" + confidence)
        speak("The person is " + id + " with confidence of" + confidence)

face_recognition()