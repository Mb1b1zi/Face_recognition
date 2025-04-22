# Configuration settings for the face recognition system

KNOWN_FACES_DIR = "known_faces"
NEW_FACES_DIR = "known_faces/new_detected"
OUTPUT_DIR = "output"

# Minimum confidence threshold for face recognition
MIN_CONFIDENCE_THRESHOLD = 0.6

# Face detection settings
FACE_DETECTION_MODEL = "hog"  # or "cnn" for GPU