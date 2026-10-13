import cv2
import os
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

class HandDetector:
    def __init__(self, mode=False, max_hands=1, detection_con=0.7, track_con=0.7):
        self.max_hands = max_hands
        
        # Download model if not exists
        model_path = os.path.join(os.path.dirname(__file__), 'hand_landmarker.task')
        if not os.path.exists(model_path):
            print("Downloading hand_landmarker.task...")
            url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
            urllib.request.urlretrieve(url, model_path)
            print("Download complete.")
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=self.max_hands,
            running_mode=vision.RunningMode.VIDEO,
            min_hand_detection_confidence=float(detection_con),
            min_hand_presence_confidence=float(track_con),
            min_tracking_confidence=float(track_con)
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.results = None

    def find_hands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
        self.results = self.detector.detect_for_video(mp_image, timestamp_ms)
        
        if draw and self.results.hand_landmarks:
            h, w, _ = img.shape
            for hand_landmarks in self.results.hand_landmarks:
                # Convert normalized landmarks to pixel coordinates
                pixel_landmarks = []
                for lm in hand_landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    pixel_landmarks.append((cx, cy))
                    cv2.circle(img, (cx, cy), 4, (255, 0, 255), cv2.FILLED)
                
                # Draw connections
                for connection in HAND_CONNECTIONS:
                    start_idx = connection[0]
                    end_idx = connection[1]
                    cv2.line(img, pixel_landmarks[start_idx], pixel_landmarks[end_idx], (0, 255, 0), 2)
                    
        return img

    def get_landmarks(self):
        landmarks_list = []
        if self.results and self.results.hand_landmarks:
            # Get up to max_hands
            for my_hand in self.results.hand_landmarks[:self.max_hands]:
                hand_lms = []
                for lm in my_hand:
                    hand_lms.append([lm.x, lm.y, lm.z])
                landmarks_list.append(hand_lms)
        return landmarks_list
