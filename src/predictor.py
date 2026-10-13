import tensorflow as tf
import json
import numpy as np
import os

class SignPredictor:
    def __init__(self, model_path="models/sign_language_model.keras", labels_path="models/labels.json"):
        self.model_loaded = False
        self.model = None
        self.labels = {}
        
        if os.path.exists(model_path) and os.path.exists(labels_path):
            try:
                self.model = tf.keras.models.load_model(model_path)
                with open(labels_path, "r") as f:
                    # JSON keys are strings, convert back to int
                    loaded_labels = json.load(f)
                    self.labels = {int(k): v for k, v in loaded_labels.items()}
                self.model_loaded = True
            except Exception as e:
                print(f"Error loading model: {e}")
                
    def predict(self, features):
        if not self.model_loaded or features is None:
            return None, 0.0
            
        features = np.array([features]) # Add batch dimension
        predictions = self.model.predict(features, verbose=0)[0]
        
        class_id = np.argmax(predictions)
        confidence = predictions[class_id]
        
        word = self.labels.get(class_id, "Unknown")
        return word, confidence
