def create_dataset(name):
    import cv2
    import os
    import pyttsx3

    def speak(command):
        engine = pyttsx3.init("sapi5")
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[0].id)
        engine.setProperty('rate', 180)
        engine.say(command)
        engine.runAndWait()

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

    print("Initializing face capture. Look the camera and wait ...")
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
    print("Face saved to database")
    speak("Face saved to database")
    cam.release()
    cv2.destroyAllWindows()

create_dataset("naresh")
