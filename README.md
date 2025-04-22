# Face Recognition System

A modular face recognition system built with Python, OpenCV, and face_recognition library.

## Overview

This project implements a flexible face recognition system that can:
- Detect faces in images or webcam feed
- Recognize faces by matching them against known faces
- Display and save results with highlighted face areas and names

The system is designed to be easily extensible and can be used for various applications like attendance systems, security cameras, or personal photo organization.

## Features

- **Face Detection**: Automatically locates faces in images or video streams
- **Face Recognition**: Matches detected faces against a database of known faces
- **Real-time Processing**: Works with webcam for live face recognition
- **Confidence Scoring**: Shows confidence level for each match
- **Image Processing**: Process individual images and save results
- **Modular Design**: Easily extendable for custom applications

## Project Structure

```
face_recognition_project/
├── main.py                  # Entry point of the application
├── face_recognition_system.py  # Core functionality
├── utils.py                 # Utility functions
├── face_database.py         # Optional database component
├── known_faces/             # Directory to store known face images
└── output/                  # Directory to store processed images
```

## Requirements

- Python 3.6+
- OpenCV
- dlib
- face_recognition
- numpy

## Installation

### For Debian 12 (and similar Linux distributions)

1. Install system dependencies:
```bash
sudo apt update
sudo apt upgrade
sudo apt install -y python3-pip python3-dev build-essential cmake pkg-config
sudo apt install -y libopencv-dev python3-opencv
sudo apt install -y libx11-dev libatlas-base-dev
sudo apt install -y libgtk-3-dev libboost-python-dev
```

2. Install Python packages:
```bash
# Install numpy
pip3 install numpy

# Install dlib
pip3 install dlib

# Install face_recognition
pip3 install face_recognition

# Install opencv-python
pip3 install opencv-python
```

### For other operating systems

Please refer to the installation guides for each dependency:
- [dlib installation guide](https://github.com/davisking/dlib)
- [face_recognition installation guide](https://github.com/ageitgey/face_recognition#installation)
- [OpenCV installation guide](https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html)

## Usage

### Adding Known Faces

1. Create a directory named `known_faces` if it doesn't exist already
2. Add clear photos of people you want to recognize with filenames like `person_name.jpg`
   - Use underscores for spaces (e.g., `john_smith.jpg`)
   - The system will automatically convert the filename to a person's name

### Running the System

1. Run the main application:
```bash
python3 main.py
```

2. Choose from the available options:
   - Process an image file: Enter the path to an image to recognize faces in it
   - Run live webcam recognition: Use your webcam for real-time face recognition

### Using the Optional Database

If you want to store face encodings in a database instead of loading from files each time:

1. Import the FaceDatabase class:
```python
from face_database import FaceDatabase
```

2. Initialize and use it in your application:
```python
# Example usage
db = FaceDatabase()
db.add_face("John Smith", face_encoding)
```

## Extending the System

The modular design makes it easy to extend functionality:

- Add user interface with Tkinter or web frameworks
- Implement an attendance system
- Connect to other databases or cloud services
- Add face registration functionality
- Implement logging and reporting features

## Troubleshooting

- **Webcam not working**: Ensure you have proper permissions and your webcam is correctly connected
- **Face recognition accuracy issues**: Use better lighting and make sure reference photos are clear
- **Performance issues**: Try reducing frame size for webcam processing or optimize image processing

