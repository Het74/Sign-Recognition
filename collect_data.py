import cv2
import os
import time
import numpy as np
from src.hand_detector import HandDetector
from src.feature_extractor import process_landmarks

# Create base dataset directory
DATA_DIR = 'dataset'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def main():
    print("=========================================")
    print("  BEYOND WORDS - Data Collection Tool")
    print("=========================================")
    
    gesture_name = input("Enter the gesture name to collect (e.g., HELLO, THANK YOU, HELP, YES): ").strip().upper()
    
    # Create directory for specific gesture
    # Replace spaces with underscores for folder names
    folder_name = gesture_name.replace(" ", "_").lower()
    gesture_dir = os.path.join(DATA_DIR, folder_name)
    if not os.path.exists(gesture_dir):
        os.makedirs(gesture_dir)
        
    num_samples = int(input("How many samples do you want to collect? (Recommended: 500): "))
    
    print(f"\nPreparation: Get ready to perform the '{gesture_name}' gesture.")
    print("Press 's' to start collecting data.")
    print("Press 'q' at any time to quit.")
    
    cap = cv2.VideoCapture(0)
    detector = HandDetector(max_hands=2)
    
    started = False
    samples_collected = 0
    data = []
    
    while True:
        success, img = cap.read()
        if not success:
            continue
            
        img = cv2.flip(img, 1) # Mirror image
        
        # Detect hands and get landmarks
        img = detector.find_hands(img)
        landmarks = detector.get_landmarks()
        
        # Display instructions/status
        if not started:
            cv2.putText(img, f"Press 's' to start collecting: {gesture_name}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(img, f"Collecting: {gesture_name} ({samples_collected}/{num_samples})", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # If hand is detected, save the normalized landmarks
            if landmarks:
                features = process_landmarks(landmarks)
                if features:
                    data.append(features)
                    samples_collected += 1
                    time.sleep(0.01) # Small delay to capture varied frames
            
        cv2.imshow("Data Collection", img)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and not started:
            print("Starting collection...")
            started = True
            
        if samples_collected >= num_samples:
            print("Collection complete!")
            break
            
    cap.release()
    cv2.destroyAllWindows()
    
    if data:
        # Save to CSV
        print(f"Saving {len(data)} samples...")
        data_np = np.array(data)
        
        # We append timestamp to the filename to allow multiple collection sessions
        timestamp = int(time.time())
        save_path = os.path.join(gesture_dir, f"data_{timestamp}.csv")
        
        np.savetxt(save_path, data_np, delimiter=",")
        print(f"Data saved to {save_path}")
    else:
        print("No data collected.")

if __name__ == "__main__":
    main()
