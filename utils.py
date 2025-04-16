import os

def create_directory_if_not_exists(directory):
    """Create a directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
        return True
    return False

def is_image_file(filename):
    """Check if a file is an image based on its extension."""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    return any(filename.lower().endswith(ext) for ext in image_extensions)

def format_name_from_filename(filename):
    """Format a person's name from a filename."""
    # Remove extension and replace underscores with spaces
    name = os.path.splitext(filename)[0].replace('_', ' ')
    # Capitalize each word
    return name.title()