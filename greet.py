import pyttsx3
import datetime
import speech_recognition as sr


r = sr.Recognizer()
mic = sr.Microphone()

engine = pyttsx3.init("sapi5")
voices = engine.getProperty("voices")
engine.setProperty("voice", voices[0].id)
engine.setProperty("rate",180)

def speak(audio):
    engine.say(audio)
    engine.runAndWait()

def getName():
    path = "C:/Users/Nares/Desktop/Vision/username.txt"
    file = open(path,"r")
    name = file.read()
    file.close()
    return name

def greetMe():
    hour = int(datetime.datetime.now().hour)
    if hour>=0 and hour<=12:
        speak(f"     Good Morning  {getName()}" )
    elif hour >12 and hour<=18:
        speak(f"    Good Afternoon  {getName()}")
    else:
        speak(f"     Good Evening  {getName()}")
    if len(getName()) != 0:
        speak("my name is vision")
        speak("Please tell me, how can i help you")
    else:
        speak("my name is vision")
        speak("I didn't know your name can you please tell your name?")
        speak("Please tell the name in the format")
        speak("my name is and add your name in last")
        speak("it is easy for me to understand this format")
