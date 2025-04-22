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
        print("\nStarting webcam face recognition...")
        print("\nControls:")
        print("- Press 'q' to quit")
        print("- Press 's' to save an unknown face")
        print("\nTips for better face detection:")
        print("1. Ensure good lighting")
        print("2. Face the camera directly")
        print("3. Keep still when saving")
        print("4. Maintain proper distance (2-3 feet from camera)\n")
        
        video_capture = cv2.VideoCapture(0)
        
        # Set higher resolution
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
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
                
                # If face is unknown, show save prompt
                if name.startswith("Unknown"):
                    cv2.putText(frame, "Press 's' to save face", (10, 30), font, 0.7, (0, 0, 255), 1)
            
            # Display frame
            cv2.imshow('Face Recognition', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save unknown faces when 's' is pressed
                for (top, right, bottom, left), name in zip(face_locations, face_names):
                    if name.startswith("Unknown"):
                        face_location = (top*4, right*4, bottom*4, left*4)
                        saved_path = self.save_detected_face(frame, face_location)
                        print(f"Saved unknown face to: {saved_path}")
        
        # Release resources
        video_capture.release()
        cv2.destroyAllWindows()
    
    def save_detected_face(self, frame, face_location, name="Unknown"):
        """Save a detected face for future recognition."""
        try:
            top, right, bottom, left = face_location
            
            # Add padding around the face (20% on each side)
            height = bottom - top
            width = right - left
            padding_y = int(height * 0.2)
            padding_x = int(width * 0.2)
            
            # Ensure padded coordinates don't go outside frame bounds
            frame_height, frame_width = frame.shape[:2]
            top = max(0, top - padding_y)
            bottom = min(frame_height, bottom + padding_y)
            left = max(0, left - padding_x)
            right = min(frame_width, right + padding_x)
            
            # Extract face image with padding
            face_image = frame[top:bottom, left:right]
            
            # Enhance image quality
            face_image = cv2.resize(face_image, (0, 0), fx=1.5, fy=1.5)  # Upscale
            face_image = cv2.detailEnhance(face_image, sigma_s=10, sigma_r=0.15)  # Enhance details
            
            # Create directory for new faces
            new_faces_dir = os.path.join(self.known_faces_dir, "new_detected")
            if not os.path.exists(new_faces_dir):
                os.makedirs(new_faces_dir)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detected_face_{timestamp}.jpg"
            filepath = os.path.join(new_faces_dir, filename)
            
            # Save the enhanced face image
            cv2.imwrite(filepath, face_image)
            print(f"Saved new face to: {filepath}")
            
            # Verify face can be detected in saved image
            saved_image = face_recognition.load_image_file(filepath)
            encodings = face_recognition.face_encodings(saved_image)
            
            if len(encodings) > 0:
                face_encoding = encodings[0]
                self.known_face_encodings.append(face_encoding)
                self.known_face_names.append(f"Person_{timestamp}")
                print(f"Successfully added face to known faces database")
                return filepath
            else:
                print("Warning: Could not detect face in saved image.")
                print("Tips for better face detection:")
                print("1. Ensure good lighting")
                print("2. Face the camera directly")
                print("3. Remove glasses if wearing any")
                print("4. Maintain proper distance from camera")
                os.remove(filepath)
                return None
                
        except Exception as e:
            print(f"Error saving face: {str(e)}")
            return None

if __name__ == "__main__":
    known_faces_directory = "known_faces"  # Make sure this folder exists and has images
    system = FaceRecognitionSystem(known_faces_directory)
    
    # Option 1: To process a single image
    # system.process_image("test_images/image1.jpg")  # replace with your test image path

    # Option 2: To start webcam recognition
    system.run_webcam()
    print("✅ Script finished running.")

