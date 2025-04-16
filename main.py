import os
import cv2
from face_recognition_system import FaceRecognitionSystem
from utils import create_directory_if_not_exists

def main():
    # Directory paths
    KNOWN_FACES_DIR = "known_faces"
    OUTPUT_DIR = "output"
    
    # Create necessary directories
    if create_directory_if_not_exists(KNOWN_FACES_DIR):
        print(f"Add face images to this directory with filenames like 'person_name.jpg'")
    create_directory_if_not_exists(OUTPUT_DIR)
    
    # Initialize the system
    face_system = FaceRecognitionSystem(KNOWN_FACES_DIR)
    
    # Display menu
    while True:
        print("\nFace Recognition System")
        print("1. Process an image file")
        print("2. Run live webcam recognition")
        print("3. Exit")
        choice = input("Enter your choice (1/2/3): ")
        
        if choice == '1':
            process_image_workflow(face_system, OUTPUT_DIR)
        elif choice == '2':
            face_system.run_webcam()
        elif choice == '3':
            print("Exiting program. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

def process_image_workflow(face_system, output_dir):
    """Handle the image processing workflow."""
    image_path = input("Enter the path to the image file: ")
    if os.path.exists(image_path):
        result = face_system.process_image(image_path, output_dir)
        
        # Show the processed image
        cv2.imshow('Recognition Result', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Image file not found.")

if __name__ == "__main__":
    main()