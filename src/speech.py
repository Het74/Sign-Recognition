import pyttsx3
import speech_recognition as sr
import threading

class SpeechHandler:
    def __init__(self):
        # Initialize Speech-to-Text
        self.recognizer = sr.Recognizer()

    def speak(self, text):
        """Speaks the text in a separate thread so it doesn't block the UI"""
        def run_speech():
            # Initialize locally inside the thread to avoid 'run loop already started' errors
            engine = pyttsx3.init()
            engine.setProperty('rate', 150) # Speed of speech
            engine.say(text)
            engine.runAndWait()
            
        thread = threading.Thread(target=run_speech)
        thread.start()

    def listen(self, callback):
        """Listens to microphone in a separate thread and returns text via callback"""
        def run_listen():
            with sr.Microphone() as source:
                print("Listening...")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                try:
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=5)
                    text = self.recognizer.recognize_google(audio)
                    callback(text.upper())
                except sr.WaitTimeoutError:
                    callback("Error: Timeout")
                except sr.UnknownValueError:
                    callback("Error: Could not understand")
                except sr.RequestError:
                    callback("Error: API unavailable")
                    
        thread = threading.Thread(target=run_listen)
        thread.start()

    def start_continuous_listening(self, callback):
        self.is_listening = True
        
        def run_listen():
            with sr.Microphone() as source:
                print("Started continuous listening...")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                while self.is_listening:
                    try:
                        audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)
                        text = self.recognizer.recognize_google(audio)
                        callback(text.upper())
                    except sr.WaitTimeoutError:
                        continue
                    except sr.UnknownValueError:
                        continue
                    except sr.RequestError:
                        callback("Error: API unavailable")
                        break
                        
        self.listen_thread = threading.Thread(target=run_listen)
        self.listen_thread.daemon = True
        self.listen_thread.start()

    def stop_listening(self):
        self.is_listening = False
