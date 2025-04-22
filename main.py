import os
from face_recognition_system import FaceRecognitionSystem
from utils import create_directory_if_not_exists

def main():
    # Directory paths
    KNOWN_FACES_DIR = "known_faces"
    OUTPUT_DIR = "output"
    
    # Create necessary directories
    create_directory_if_not_exists(KNOWN_FACES_DIR)
    create_directory_if_not_exists(OUTPUT_DIR)
    
    # Initialize face recognition system
    system = FaceRecognitionSystem(KNOWN_FACES_DIR)
    
    # Run webcam recognition
    system.run_webcam()

if __name__ == "__main__":
    main()