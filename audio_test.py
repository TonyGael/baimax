import speech_recognition as sr
import pyttsx3

def listen():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    with microphone as source:
        print("Calibrando el micrófono...")
        recognizer.adjust_for_ambient_noise(source)
        print("Di algo:")
        audio = recognizer.listen(source)

    try:
        print("Reconociendo...")
        text = recognizer.recognize_google(audio, language="es-ES")
        print(f"Tu dijiste: {text}")
        return text
    except sr.UnknownValueError:
        print("No pude entender lo que dijiste")
        return None
    except sr.RequestError:
        print("No pude conectarme al servicio de reconocimiento de voz")
        return None

def speak(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Velocidad de la voz
    engine.setProperty('volume', 0.9)  # Volumen de la voz

    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)  # Seleccionar voz femenina

    engine.say(text)
    engine.runAndWait()

if __name__ == "__main__":
    while True:
        text = listen()
        if text:
            speak(text)
        if text and "salir" in text.lower():
            speak("Adiós")
            break
