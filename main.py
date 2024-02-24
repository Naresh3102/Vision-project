import speech_recognition as sr
import pyttsx3
import random
from ultralytics import YOLO
import cv2
import os
import numpy as np
from PIL import Image

r = sr.Recognizer()
mic = sr.Microphone()

def setName(name):
    path = "C:/Users/Nares/Desktop/Vision/username.txt"
    with open(path,"w+") as file:
        file.write(name)
        file.close()

def getName():
    path = "C:/Users/Nares/Desktop/Vision/username.txt"
    file = open(path,"r")
    name = file.read()
    file.close()
    return name

def there_exists(terms):
    for term in terms:
        if term in speech:
            return True

def speak(command):
    engine = pyttsx3.init("sapi5")
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)
    engine.setProperty('rate', 180)
    engine.say(command)
    engine.runAndWait()

def get_speech():
    with mic as source:
        r.adjust_for_ambient_noise(source,duration=0.2)
        audio = r.listen(source)
        speech = ""
        try:
            speech = r.recognize_google(audio,language='en')
        except Exception as e:
            speech = "There is a exception named {}".format(type(e).__name__)
    return speech.lower()

def speak(command):
    engine = pyttsx3.init("sapi5")
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)
    engine.setProperty('rate', 180)
    engine.say(command)
    engine.runAndWait()

def create_dataset(name):

    cam = cv2.VideoCapture(0)
    cam.set(3, 640) # set video width
    cam.set(4, 480) # set video height

    face_detector = cv2.CascadeClassifier('C:/Users/Nares/Desktop/Vision/face recognition/haarcascade_frontalface_default.xml')

    # For each person,one numeric face id
    ids = {0}
    path = 'C:/Users/Nares/Desktop/Vision/face recognition/dataset'
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]  
    for imagepath in imagePaths:
        id = os.path.split(imagepath)[-1].split(".")[1]
        ids.add(int(id))
    ids =list(ids)
    if(max(ids) == 0):
        face_id = 0
    else:
        face_id = max(ids)

    speak("Initializing face capture. Look the camera and wait ...")
    # Initialize individual sampling face count
    count = 0

    while(True):

        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            count += 1

            # Save the captured image into the datasets folder
            cv2.imwrite("C:/Users/Nares/Desktop/Vision/face recognition/dataset/User." + str(face_id) + '.' + str(name) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

        if count == 30:
            break
    # Do a bit of cleanup
    speak("Face saved to database")
    cam.release()
    cv2.destroyAllWindows()

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
    speak("{0} faces trained.".format(len(np.unique(ids))))

def face_recognition(frame):
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

    img = cv2.imread(frame)

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
        speak("The person is " + id + " with confidence of" + confidence)

def object_detection():

    video_capture = cv2.VideoCapture(0)

    # Capture a single frame from the camera
    ret, frame = video_capture.read()

    video_capture.release()

    model = YOLO("yolov8l.pt")

    results = model.predict(frame)

    result = results[0]

    count = len(result.boxes)

    objects_list = []
    coordinates_list = []

    for box in result.boxes:
        label = result.names[box.cls[0].item()]
        cords = [round(x) for x in box.xyxy[0].tolist()]
        prob = box.conf[0].item()
        #print("Object type:",label)
        #print("Coordinates:",cords)
        #print("Probability:",prob)
        #print("----")
        objects_list.append(label)
        coordinates_list.append(cords)

    print(objects_list)
    print(coordinates_list)

    speak("There are {} objects in the captured image.".format(count))
    object_name = ""
    for i in objects_list:
        object_name = object_name + str(i) +" "

    speak("The objects are {}.".format(object_name))
    if "person" in objects_list:
        face_recognition(frame=frame)

def respond(speech):
    # 1:wake up
    if there_exists(['hello vision','wake up vision','wake up']):
        greetings = [f"hey, how can I help you {getName()}", f"hey, what's up? {getName()}", f"I'm listening {getName()}", f"how can I help you? {getName()}", f"hello {getName()}"]
        greet = greetings[random.randint(0,len(greetings)-1)]
        speak(greet)
    # 2: name
    elif there_exists(["what is your name","what's your name","tell me your name","who are you"]):
        if len(getName()) != 0:
            speak("my name is Vision. i am your virtual assistant")
        else:
            speak("my name is Vision.. i am your virtual assistant. what's your name?")
    # 3:set user name
    elif there_exists(["my name is"]):
        name = speech.split("is")[-1].strip()
        speak(f"okay, i will remember that {name}")
        setName(name)
    # 4: greeting
    elif there_exists(["how are you","how are you doing"]):
        speak(f"I'm very well, thanks for asking {getName()}")
    # 5: Object detection
    elif there_exists(["object detection","detect object"]):
        object_detection()
    # 6: Training a new face
    elif there_exists(["train face","new face"]):
        speak("Tell me the name of the person")
        face_name = get_speech()
        create_dataset(face_name)
    else:
        speak("I didn't get what you said. please use the proper command if you are trying to use any features")

if __name__ == "__main__":
    while True:
        print("I am ready. you can speak")
        speech = get_speech()
        if "hello" in speech:
            from greet import greetMe
            greetMe()

            while True:
                speech = get_speech()
                if "go to sleep" in speech:
                    speak("ok sir , you can call me anytime by just saying wake up")
                    break
                else:
                    respond(speech)