import sqlite3
import pickle
import numpy as np
import os

class FaceDatabase:
    def __init__(self, db_path="face_database.db"):
        self.db_path = db_path
        self.initialize_database()
    
    def initialize_database(self):
        """Create the database and tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create table for storing face encodings
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            encoding BLOB NOT NULL,
            added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        conn.commit()
        conn.close()
        
    def add_face(self, name, encoding):
        """Add a face encoding to the database."""
        # Convert numpy array to binary for storage
        encoding_binary = pickle.dumps(encoding)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO faces (name, encoding) VALUES (?, ?)",
            (name, encoding_binary)
        )
        
        conn.commit()
        conn.close()
        
        return cursor.lastrowid
    
    def get_all_faces(self):
        """Retrieve all faces from the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, name, encoding FROM faces")
        rows = cursor.fetchall()
        
        faces = []
        for row in rows:
            id, name, encoding_binary = row
            # Convert binary back to numpy array
            encoding = pickle.loads(encoding_binary)
            faces.append({"id": id, "name": name, "encoding": encoding})
        
        conn.close()
        return faces
    
    def get_face_by_id(self, face_id):
        """Retrieve a face by its ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, name, encoding FROM faces WHERE id = ?", (face_id,))
        row = cursor.fetchone()
        
        if row:
            id, name, encoding_binary = row
            encoding = pickle.loads(encoding_binary)
            conn.close()
            return {"id": id, "name": name, "encoding": encoding}
        
        conn.close()
        return None
    
    def delete_face(self, face_id):
        """Delete a face from the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM faces WHERE id = ?", (face_id,))
        
        conn.commit()
        conn.close()
        
        return cursor.rowcount > 0  # Return True if a row was deleted
    
    def get_face_encodings_and_names(self):
        """Get all face encodings and names for recognition."""
        faces = self.get_all_faces()
        encodings = [face["encoding"] for face in faces]
        names = [face["name"] for face in faces]
        
        return encodings, names