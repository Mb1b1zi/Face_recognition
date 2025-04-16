
import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime
print("✅ Script started running...")

class FaceRecognitionSystem:
    def __init__(self, known_faces_dir):
        self.known_faces_dir = known_faces_dir
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces()
        
    def load_known_faces(self):
        """Load known faces from directory."""
        print("Loading known faces...")
        for filename in os.listdir(self.known_faces_dir):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                # Get person's name from filename (e.g., john_smith.jpg -> "John Smith")
                name = os.path.splitext(filename)[0].replace('_', ' ').title()
                
                # Load image file
                image_path = os.path.join(self.known_faces_dir, filename)
                image = face_recognition.load_image_file(image_path)
                
                # Get face encoding
                face_locations = face_recognition.face_locations(image)
                if face_locations:
                    face_encoding = face_recognition.face_encodings(image, face_locations)[0]
                    self.known_face_encodings.append(face_encoding)
                    self.known_face_names.append(name)
                    print(f"Loaded: {name}")
        
        print(f"Loaded {len(self.known_face_names)} known faces")
    
    def process_image(self, image_path, output_dir="output"):
        """Process a single image and identify faces."""
        print(f"Processing image: {image_path}")
        
        # Load image
        image = face_recognition.load_image_file(image_path)
        
        # Find all faces in the image
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)
        
        # Convert image to OpenCV format for drawing
        image_cv = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Process each face
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # See if the face matches any known faces
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"
            
            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
                    confidence = 1 - face_distances[best_match_index]
                    name = f"{name} ({confidence:.2f})"
            
            # Draw a box around the face
            cv2.rectangle(image_cv, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Draw a label with a name below the face
            cv2.rectangle(image_cv, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(image_cv, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)
        
        # Make sure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Save the resulting image
        output_path = os.path.join(output_dir, f"recognized_{os.path.basename(image_path)}")
        cv2.imwrite(output_path, image_cv)
        print(f"Output saved to: {output_path}")
        return image_cv
    
    def run_webcam(self):
        """Run face recognition on webcam stream."""
        print("Starting webcam face recognition... Press 'q' to quit")
        
        # Start video capture
        video_capture = cv2.VideoCapture(0)
        
        if not video_capture.isOpened():
            print("Error: Could not open webcam")
            return
        
        while True:
            # Grab a single frame of video
            ret, frame = video_capture.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            # Resize frame for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            
            # Convert the image from BGR color (OpenCV) to RGB color
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Find all faces and face encodings in the current frame
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            # Initialize an array for the names of the detected faces
            face_names = []
            
            for face_encoding in face_encodings:
                # See if the face matches any known faces
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                name = "Unknown"
                
                # Use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = 1 - face_distances[best_match_index]
                        name = f"{name} ({confidence:.2f})"
                
                face_names.append(name)
            
            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since we detected in scaled-down image
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                
                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                
                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)
            
            # Display frame
            cv2.imshow('Face Recognition', frame)
            
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        video_capture.release()
        cv2.destroyAllWindows()
if __name__ == "__main__":
    known_faces_directory = "known_faces"  # Make sure this folder exists and has images
    system = FaceRecognitionSystem(known_faces_directory)
    
    # Option 1: To process a single image
    # system.process_image("test_images/image1.jpg")  # replace with your test image path

    # Option 2: To start webcam recognition
    system.run_webcam()
    print("✅ Script finished running.")
        
        