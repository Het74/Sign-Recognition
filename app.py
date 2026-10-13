import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk
import cv2
import threading
import time
from datetime import datetime
import os

from src.hand_detector import HandDetector
from src.feature_extractor import process_landmarks
from src.predictor import SignPredictor
from src.speech import SpeechHandler

# Configure CustomTkinter
ctk.set_appearance_mode("Dark")  
ctk.set_default_color_theme("blue")  

class BeyondWordsApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("BEYOND WORDS - Sign Language Recognition & Communication System")
        self.geometry("1300x800")
        self.minsize(1100, 700)
        
        # Core components
        self.detector = HandDetector(max_hands=2)
        self.predictor = SignPredictor()
        self.speech = SpeechHandler()
        
        # State variables
        self.cap = None
        self.is_camera_on = False
        self.is_recognition_on = False
        self.is_mic_listening = False
        
        self.current_sentence = []
        
        # Prediction stabilization
        self.prediction_buffer = []
        self.buffer_size = 5
        self.last_added_word = ""
        self.last_added_time = time.time()
        self.cooldown = 2.0
        self.min_confidence = 0.75
        
        self.setup_ui()
        
    def setup_ui(self):
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        # HEADER
        header_frame = ctk.CTkFrame(self, fg_color="#1E1E2E", corner_radius=0)
        header_frame.grid(row=0, column=0, columnspan=2, sticky="ew")
        
        title_label = ctk.CTkLabel(header_frame, text="BEYOND WORDS", font=("Helvetica", 28, "bold"), text_color="#A6ACCD")
        title_label.pack(pady=(15, 5))
        
        subtitle_label = ctk.CTkLabel(header_frame, text="Sign Language Recognition & Communication System", font=("Helvetica", 14), text_color="#717CB4")
        subtitle_label.pack(pady=(0, 15))
        
        # TABS
        self.tabview = ctk.CTkTabview(self)
        self.tabview.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=20, pady=10)
        
        self.tab_recognition = self.tabview.add("Sign Recognition")
        self.tab_reverse = self.tabview.add("Speech to Sign")
        
        self.setup_recognition_tab()
        self.setup_reverse_tab()
        
        # BOTTOM STATUS BAR
        status_frame = ctk.CTkFrame(self, height=30, corner_radius=0)
        status_frame.grid(row=2, column=0, columnspan=2, sticky="ew")
        
        model_status_text = "Model: LOADED" if self.predictor.model_loaded else "Model: NOT TRAINED"
        model_color = "#A3BE8C" if self.predictor.model_loaded else "#BF616A"
        
        self.lbl_model_status = ctk.CTkLabel(status_frame, text=model_status_text, text_color=model_color)
        self.lbl_model_status.pack(side="left", padx=20)
        
        self.lbl_cam_status = ctk.CTkLabel(status_frame, text="Camera: OFF")
        self.lbl_cam_status.pack(side="left", padx=20)
        
        self.lbl_rec_status = ctk.CTkLabel(status_frame, text="Recognition: OFF")
        self.lbl_rec_status.pack(side="left", padx=20)

    def setup_recognition_tab(self):
        self.tab_recognition.grid_rowconfigure(0, weight=1)
        self.tab_recognition.grid_columnconfigure(0, weight=6)
        self.tab_recognition.grid_columnconfigure(1, weight=4)
        
        # LEFT: CAMERA
        left_frame = ctk.CTkFrame(self.tab_recognition)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left_frame.grid_rowconfigure(1, weight=1)
        left_frame.grid_columnconfigure(0, weight=1)
        
        cam_header = ctk.CTkLabel(left_frame, text="Live Camera Feed", font=("Helvetica", 18, "bold"))
        cam_header.grid(row=0, column=0, pady=10)
        
        self.video_container = ctk.CTkFrame(left_frame, fg_color="black")
        self.video_container.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.video_container.pack_propagate(False)
        
        self.video_label = tk.Label(self.video_container, bg="black")
        self.video_label.pack(expand=True, fill="both")
        
        cam_controls = ctk.CTkFrame(left_frame, fg_color="transparent")
        cam_controls.grid(row=2, column=0, pady=10)
        
        self.btn_toggle_cam = ctk.CTkButton(cam_controls, text="Start Camera", command=self.toggle_camera, width=150)
        self.btn_toggle_cam.pack(side="left", padx=10)
        
        self.btn_toggle_rec = ctk.CTkButton(cam_controls, text="Start Recognition", command=self.toggle_recognition, width=150, state="disabled")
        self.btn_toggle_rec.pack(side="left", padx=10)
        
        # RIGHT: RECOGNITION
        right_frame = ctk.CTkScrollableFrame(self.tab_recognition)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        right_frame.grid_columnconfigure(0, weight=1)
        
        pred_card = ctk.CTkFrame(right_frame, fg_color="#2E3440", corner_radius=15)
        pred_card.grid(row=0, column=0, sticky="ew", padx=20, pady=20)
        
        ctk.CTkLabel(pred_card, text="Current Prediction", font=("Helvetica", 14)).pack(pady=(10, 0))
        self.lbl_prediction = ctk.CTkLabel(pred_card, text="---", font=("Helvetica", 42, "bold"), text_color="#88C0D0")
        self.lbl_prediction.pack(pady=10)
        
        self.lbl_confidence = ctk.CTkLabel(pred_card, text="Confidence: 0.0%", font=("Helvetica", 14))
        self.lbl_confidence.pack(pady=(0, 10))
        
        sentence_frame = ctk.CTkFrame(right_frame)
        sentence_frame.grid(row=1, column=0, sticky="ew", padx=20, pady=10)
        
        ctk.CTkLabel(sentence_frame, text="Generated Sentence", font=("Helvetica", 16, "bold")).pack(pady=(10, 5))
        
        self.txt_sentence = ctk.CTkTextbox(sentence_frame, height=80, font=("Helvetica", 20))
        self.txt_sentence.pack(fill="x", padx=10, pady=5)
        self.txt_sentence.configure(state="disabled")
        
        sentence_controls = ctk.CTkFrame(sentence_frame, fg_color="transparent")
        sentence_controls.pack(pady=10)
        
        ctk.CTkButton(sentence_controls, text="Speak", command=self.speak_sentence, fg_color="#A3BE8C", hover_color="#8FBCBB").pack(side="left", padx=5)
        ctk.CTkButton(sentence_controls, text="Delete Last", command=self.delete_last_word).pack(side="left", padx=5)
        ctk.CTkButton(sentence_controls, text="Clear", command=self.clear_sentence, fg_color="#BF616A", hover_color="#D08770").pack(side="left", padx=5)
        
        hist_frame = ctk.CTkFrame(right_frame)
        hist_frame.grid(row=2, column=0, sticky="nsew", padx=20, pady=10)
        hist_frame.grid_rowconfigure(1, weight=1)
        
        ctk.CTkLabel(hist_frame, text="Prediction History", font=("Helvetica", 14, "bold")).grid(row=0, column=0, pady=5)
        
        self.txt_history = ctk.CTkTextbox(hist_frame, height=100)
        self.txt_history.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.txt_history.configure(state="disabled")

    def setup_reverse_tab(self):
        self.tab_reverse.grid_rowconfigure(0, weight=1)
        self.tab_reverse.grid_columnconfigure(0, weight=1)
        self.tab_reverse.grid_columnconfigure(1, weight=1)

        # Left: Controls & Text
        controls_frame = ctk.CTkFrame(self.tab_reverse)
        controls_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        ctk.CTkLabel(controls_frame, text="Continuous Speech-to-Text", font=("Helvetica", 18, "bold")).pack(pady=10)
        
        self.btn_toggle_mic = ctk.CTkButton(controls_frame, text="Start Mic", command=self.toggle_mic, fg_color="#EBCB8B", text_color="black", hover_color="#D08770", width=200, height=40, font=("Helvetica", 16, "bold"))
        self.btn_toggle_mic.pack(pady=10)
        
        self.txt_speech = ctk.CTkTextbox(controls_frame, height=300, font=("Helvetica", 22))
        self.txt_speech.pack(fill="both", expand=True, padx=20, pady=10)
        self.txt_speech.configure(state="disabled")

        ctk.CTkButton(controls_frame, text="Clear Text", command=self.clear_speech_text, fg_color="#BF616A", hover_color="#D08770").pack(pady=10)

        # Right: Sign Display
        sign_frame = ctk.CTkFrame(self.tab_reverse)
        sign_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        self.lbl_sign_word = ctk.CTkLabel(sign_frame, text="Current Sign: ---", font=("Helvetica", 18, "bold"))
        self.lbl_sign_word.pack(pady=10)
        
        self.lbl_sign_image = tk.Label(sign_frame, bg="#2E3440")
        self.lbl_sign_image.pack(expand=True, fill="both", padx=20, pady=20)
        
        self.update_sign_image("NONE")

    # --- MAIN CAMERA CONTROLS ---
    def toggle_camera(self):
        if self.is_camera_on:
            self.is_camera_on = False
            self.btn_toggle_cam.configure(text="Start Camera")
            self.lbl_cam_status.configure(text="Camera: OFF")
            self.btn_toggle_rec.configure(state="disabled")
            if self.cap:
                self.cap.release()
            self.video_label.configure(image='')
        else:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.is_camera_on = True
                self.btn_toggle_cam.configure(text="Stop Camera")
                self.lbl_cam_status.configure(text="Camera: ON")
                self.btn_toggle_rec.configure(state="normal")
                self.update_camera()
            else:
                print("Error: Could not open camera.")

    def toggle_recognition(self):
        if not self.predictor.model_loaded:
            self.lbl_prediction.configure(text="Model not trained", text_color="#BF616A")
            return
            
        if self.is_recognition_on:
            self.is_recognition_on = False
            self.btn_toggle_rec.configure(text="Start Recognition")
            self.lbl_rec_status.configure(text="Recognition: OFF")
            self.lbl_prediction.configure(text="---")
            self.lbl_confidence.configure(text="Confidence: 0.0%")
        else:
            self.is_recognition_on = True
            self.btn_toggle_rec.configure(text="Stop Recognition")
            self.lbl_rec_status.configure(text="Recognition: ON")
            self.prediction_buffer.clear()

    def update_camera(self):
        if self.is_camera_on and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                
                frame = self.detector.find_hands(frame)
                
                if self.is_recognition_on:
                    landmarks = self.detector.get_landmarks()
                    if landmarks:
                        features = process_landmarks(landmarks)
                        if features:
                            word, conf = self.predictor.predict(features)
                            self.process_prediction(word, conf)
                    else:
                        self.prediction_buffer.clear()
                
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                width = self.video_container.winfo_width()
                height = self.video_container.winfo_height()
                if width > 1 and height > 1:
                    cv2image = cv2.resize(cv2image, (width, height), interpolation=cv2.INTER_LINEAR)
                    
                img = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
                
            self.after(5, self.update_camera)

    def process_prediction(self, word, confidence):
        if confidence >= self.min_confidence:
            self.lbl_prediction.configure(text=word, text_color="#88C0D0")
            self.lbl_confidence.configure(text=f"Confidence: {confidence*100:.1f}%")
            
            self.prediction_buffer.append(word)
            if len(self.prediction_buffer) > self.buffer_size:
                self.prediction_buffer.pop(0)
                
            if len(self.prediction_buffer) == self.buffer_size and len(set(self.prediction_buffer)) == 1:
                stable_word = self.prediction_buffer[0]
                current_time = time.time()
                if stable_word != self.last_added_word or (current_time - self.last_added_time) > self.cooldown:
                    self.add_word_to_sentence(stable_word)
                    self.add_to_history(stable_word, confidence)
                    self.last_added_word = stable_word
                    self.last_added_time = current_time
                    self.prediction_buffer.clear()
        else:
            self.lbl_prediction.configure(text="Not Recognized", text_color="#D08770")
            self.lbl_confidence.configure(text=f"Confidence: {confidence*100:.1f}%")
            self.prediction_buffer.clear()

    def add_word_to_sentence(self, word):
        self.current_sentence.append(word)
        self.update_sentence_display()
        
    def delete_last_word(self):
        if self.current_sentence:
            self.current_sentence.pop()
            self.update_sentence_display()
            self.last_added_word = ""
            
    def clear_sentence(self):
        self.current_sentence.clear()
        self.update_sentence_display()
        self.last_added_word = ""
        
    def update_sentence_display(self):
        text = " ".join(self.current_sentence)
        self.txt_sentence.configure(state="normal")
        self.txt_sentence.delete("1.0", tk.END)
        self.txt_sentence.insert("1.0", text)
        self.txt_sentence.configure(state="disabled")

    def speak_sentence(self):
        text = " ".join(self.current_sentence)
        if text:
            self.speech.speak(text)
            
    def add_to_history(self, word, confidence):
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"{timestamp}   {word.ljust(15)} {confidence*100:.1f}%\n"
        self.txt_history.configure(state="normal")
        self.txt_history.insert("1.0", entry)
        self.txt_history.configure(state="disabled")

    # --- REVERSE COMMUNICATION CONTROLS ---
    def toggle_mic(self):
        if self.is_mic_listening:
            self.speech.stop_listening()
            self.is_mic_listening = False
            self.btn_toggle_mic.configure(text="Start Mic", fg_color="#EBCB8B")
            self.update_sign_image("NONE")
        else:
            self.is_mic_listening = True
            self.btn_toggle_mic.configure(text="Stop Mic", fg_color="#BF616A")
            self.speech.start_continuous_listening(self.on_continuous_speech)

    def on_continuous_speech(self, text):
        def update():
            if text.startswith("Error"):
                if "API unavailable" in text:
                    self.toggle_mic() # stop automatically if API breaks
                return
                
            self.txt_speech.configure(state="normal")
            self.txt_speech.insert(tk.END, text + "\n")
            self.txt_speech.see(tk.END)
            self.txt_speech.configure(state="disabled")
            
            # Show sign for the most recently recognized phrase/word
            words = text.strip().split()
            if words:
                # First try full phrase, else fallback to last word
                success = self.update_sign_image(text.strip().upper())
                if not success:
                    self.update_sign_image(words[-1].upper())
                    
        self.after(0, update)

    def clear_speech_text(self):
        self.txt_speech.configure(state="normal")
        self.txt_speech.delete("1.0", tk.END)
        self.txt_speech.configure(state="disabled")
        self.update_sign_image("NONE")

    def update_sign_image(self, word):
        self.lbl_sign_word.configure(text=f"Current Sign: {word}")
        
        if word == "NONE":
            self.lbl_sign_image.configure(image="", text="Waiting for speech...", fg="white", font=("Helvetica", 16))
            self.lbl_sign_image.image = None
            return True
            
        file_name = word.replace(" ", "_").lower() + ".png"
        path = os.path.join("assets", "signs", file_name)
        
        if os.path.exists(path):
            img = Image.open(path)
            # Find a good size
            width = 400
            height = 400
            img = img.resize((width, height), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.lbl_sign_image.configure(image=photo, text="")
            self.lbl_sign_image.image = photo
            return True
        else:
            self.lbl_sign_image.configure(image="", text=f"Sign representation for\n'{word}'\nnot available.", fg="white", font=("Helvetica", 16))
            self.lbl_sign_image.image = None
            return False

if __name__ == "__main__":
    app = BeyondWordsApp()
    app.mainloop()
