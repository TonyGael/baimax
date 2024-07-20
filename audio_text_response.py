import speech_recognition as sr
import pyttsx3
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Inicializar el modelo GPT-2 y el tokenizador
# model_name = 'gpt2'
model_name = 'datificate/gpt2-small-spanish'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

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
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 0.9)

    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)  # Voz femenina

    engine.say(text)
    engine.runAndWait()

def ask_question(question, max_length=50):
    input_ids = tokenizer.encode(question, return_tensors='pt')
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer

if __name__ == "__main__":
    while True:
        text = listen()
        if text:
            if "salir" in text.lower():
                speak("Adiós")
                break

            response = ask_question(text)
            print(f"Respuesta: {response}")
            speak(response)
