import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import glob

DATA_DIR = 'dataset'
MODEL_DIR = 'models'

def main():
    print("=========================================")
    print("  BEYOND WORDS - Model Training Tool")
    print("=========================================")
    
    if not os.path.exists(DATA_DIR):
        print("Error: 'dataset' directory not found. Please run collect_data.py first.")
        return
        
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    X = []
    y = []
    labels_dict = {}
    label_id = 0
    
    print("Loading dataset...")
    # Iterate through each folder in dataset directory
    for folder_name in os.listdir(DATA_DIR):
        folder_path = os.path.join(DATA_DIR, folder_name)
        if os.path.isdir(folder_path):
            # Map folder name (e.g., thank_you) back to gesture name (THANK YOU)
            gesture_name = folder_name.replace("_", " ").upper()
            
            # Find all CSV files in the folder
            csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
            if not csv_files:
                continue
                
            labels_dict[label_id] = gesture_name
            class_samples = 0
            
            for csv_file in csv_files:
                try:
                    data = np.loadtxt(csv_file, delimiter=",")
                    if len(data.shape) == 1:
                        data = data.reshape(1, -1) # Handle single row
                        
                    for row in data:
                        X.append(row)
                        y.append(label_id)
                    class_samples += len(data)
                except Exception as e:
                    print(f"Error reading {csv_file}: {e}")
                    
            print(f"Loaded {class_samples} samples for class '{gesture_name}' (ID: {label_id})")
            label_id += 1
            
    if not X:
        print("Error: No data found. Please collect data first.")
        return
        
    X = np.array(X)
    y = np.array(y)
    num_classes = len(labels_dict)
    
    print(f"\nTotal Dataset: {len(X)} samples, {num_classes} classes")
    
    # Save label dictionary
    labels_path = os.path.join(MODEL_DIR, "labels.json")
    with open(labels_path, "w") as f:
        json.dump(labels_dict, f, indent=4)
    print(f"Saved labels mapping to {labels_path}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("\nBuilding model...")
    # Simple, lightweight MLP model for hand landmarks (63 features)
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X.shape[1],)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    print("Training model...")
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)
    
    print("\nEvaluating model...")
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    conf_mat = confusion_matrix(y_test, y_pred)
    
    print(f"\nModel Evaluation Metrics:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"\nConfusion Matrix:\n{conf_mat}")
    
    model_path = os.path.join(MODEL_DIR, "sign_language_model.keras")
    model.save(model_path)
    print(f"\nModel saved successfully to {model_path}")
    
if __name__ == "__main__":
    main()
