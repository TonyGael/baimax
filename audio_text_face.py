import speech_recognition as sr
import pyttsx3
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import cv2

# Inicializar el modelo GPT-2 y el tokenizador
model_name = 'datificate/gpt2-small-spanish'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Inicializar el reconocimiento de voz
recognizer = sr.Recognizer()
microphone = sr.Microphone()

# Inicializar el sintetizador de voz
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 0.9)
voices = engine.getProperty('voices')
for voice in voices:
    if "spanish" in voice.name.lower():
        engine.setProperty('voice', voice.id)
        break

# Inicializar el reconocimiento facial
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

def listen():
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
    engine.say(text)
    engine.runAndWait()

def ask_question(question, max_length=50):
    input_ids = tokenizer.encode(question, return_tensors='pt')
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer

if __name__ == "__main__":
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        cv2.imshow('Video', frame)

        if len(faces) > 0:
            text = listen()
            if text:
                if "salir" in text.lower():
                    speak("Adiós")
                    break

                response = ask_question(text)
                print(f"Respuesta: {response}")
                speak(response)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
