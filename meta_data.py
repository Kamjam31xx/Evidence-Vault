from PIL import Image
from app_types import *

def get_image_dimensions(image_path):
    """
    Universal image dimension reader that works with/without EXIF
    Handles: JPEG, PNG, WEBP, HEIC, etc.
    """
    try:
        with Image.open(image_path) as img:
            # Get dimensions directly from image header
            width, height = img.size
            
            # Only check EXIF for JPEG/TIFF files
            if img.format in ('JPEG', 'TIFF'):
                exif = img.getexif()
                orientation = exif.get(274)  # 274 = EXIF orientation tag
                
                # Swap dimensions for rotated images
                if orientation in (5, 6, 7, 8):
                    return height, width
                
            return Dimensions(width, height)
            
    except Exception as e:
        print(f"Error reading {image_path}: {e}")
        return None