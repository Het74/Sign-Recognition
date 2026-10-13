import numpy as np

def process_landmarks(landmarks_list):
    """
    Normalizes the hand landmarks relative to the wrist (landmark 0) for up to 2 hands.
    Always returns a 1D array of exactly 126 elements (63 per hand).
    """
    if not landmarks_list:
        return None
        
    features = []
    
    for hand_landmarks in landmarks_list[:2]:
        if len(hand_landmarks) != 21:
            continue
            
        hand_landmarks = np.array(hand_landmarks)
        base_x, base_y, base_z = hand_landmarks[0]
        
        normalized = []
        for lm in hand_landmarks:
            nx = lm[0] - base_x
            ny = lm[1] - base_y
            nz = lm[2] - base_z
            normalized.extend([nx, ny, nz])
            
        max_val = np.max(np.abs(normalized))
        if max_val > 0:
            normalized = list(np.array(normalized) / max_val)
            
        features.extend(normalized)
        
    # Pad to 126 if only 1 hand is detected
    while len(features) < 126:
        features.append(0.0)
        
    return features
