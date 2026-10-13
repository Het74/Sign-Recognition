# BEYOND WORDS
**Sign Language Recognition and Communication System**

## Project Overview
Beyond Words is a real-time AI-powered application designed to bridge the communication gap for individuals who use sign language. It captures hand gestures using a webcam, extracts hand landmarks using MediaPipe, processes them through a custom-trained Neural Network model, and outputs the detected word as text and speech.

## Features
- **Real-Time Recognition**: Processes webcam feed instantly.
- **Gesture Stabilization**: Prevents flickering predictions.
- **Sentence Builder**: Concatenates recognized words into meaningful sentences.
- **Text-to-Speech (TTS)**: Reads the generated sentences out loud using natural speech.
- **Reverse Communication**: Type a word to see its corresponding sign language representation.
- **Customizable Dataset**: Tools included to easily record and train your own custom gestures.

## Core Technologies Used
- **Python**: Core programming language.
- **OpenCV**: Computer Vision and webcam feed handling.
- **MediaPipe**: Real-time hand landmark extraction.
- **TensorFlow & Keras**: Lightweight Artificial Neural Network (MLP) for classification.
- **CustomTkinter**: Modern, responsive, dark-mode GUI.
- **pyttsx3**: Offline Text-to-Speech generation.

---

## 🚀 Installation & Setup

1. **Clone/Download the repository**
2. **Navigate into the project directory**:
   ```bash
   cd Beyond_Words
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: You may also need to install PyAudio. If `pip install pyaudio` fails on Windows, install it using a precompiled wheel or `pipwin install pyaudio`)*.

---

## 🔧 Training Your Own Model

The system requires a trained AI model to work. Follow these steps to collect data and train the model for the 4 core words: **HELLO**, **THANK YOU**, **HELP**, **YES**.

### 1. Collect Data
Run the data collection script for each gesture:
```bash
python collect_data.py
```
- Enter the gesture name (e.g., `HELLO`).
- Specify the number of samples (e.g., `500`).
- Press `s` when you are ready to start the camera.
- Move your hand slightly to capture various angles.
- **Repeat this process** for `THANK YOU`, `HELP`, and `YES`.

### 2. Train the Model
Once you have collected data for all classes, run the training script:
```bash
python train_model.py
```
This script will:
- Load all CSV data from the `dataset/` folder.
- Train a fast Neural Network (MLP).
- Output accuracy, precision, recall, and a confusion matrix.
- Save the trained model to `models/sign_language_model.keras` and the label mapping to `models/labels.json`.

---

## 🏃 Running the Application

Once the model is successfully trained, launch the main UI:
```bash
python app.py
```
- Click **Start Camera**.
- Click **Start Recognition**.
- Perform gestures in front of the camera. The words will appear on the right side and populate the sentence builder.
- Click **Speak** to announce the sentence.

---

## 📂 Project Architecture

```
Beyond_Words/
├── app.py                # Main CustomTkinter UI & core application logic
├── collect_data.py       # Webcam script to record & save gesture landmarks
├── train_model.py        # Script to train the Keras NN model on dataset
├── requirements.txt      # Python dependencies
├── README.md             # Documentation
│
├── dataset/              # Training data stored in CSV format
│   ├── hello/
│   ├── thank_you/
│   ├── help/
│   └── yes/
│
├── models/               # Contains the trained .keras model and labels mapping
├── src/                  # Helper modules
│   ├── hand_detector.py      # MediaPipe wrapper for hand detection
│   ├── feature_extractor.py  # Flattens and normalizes landmarks
│   ├── predictor.py          # Model inference wrapper
│   └── speech.py             # TTS / STT functionality
│
└── assets/               # Stores images for "Reverse Communication"
    └── signs/
```

## 🔮 Future Improvements
While the first iteration supports exactly 4 gestures, the system is designed to scale dynamically. To add new words (e.g., `WATER`, `PLEASE`), simply run `collect_data.py` with the new word, re-run `train_model.py`, and the application will automatically recognize the new class!
