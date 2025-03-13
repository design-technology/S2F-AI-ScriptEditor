
#r: opencv-python
#r: numpy

import rhinoscriptsyntax as rs
import os
import tempfile
import cv2

def capture_viewport(filepath=None):
    """Captures active viewport and returns as image"""
    
    # Capture viewport to file
    rs.Command(f'_-ViewCaptureToFile "{filepath}" _Enter', False)
    
    # Load image with OpenCV
    img = cv2.imread(filepath)
            
    return img

def create_negative(img):
    """Create negative image using OpenCV"""
    # Create negative by subtracting from 255
    return 255 - img

def main():
    # Capture and save original to desktop
    save_path = os.path.join(os.path.expanduser("~"), "Desktop", "rhino_capture.png")
    img = capture_viewport(save_path)
    
    # Create negative image
    negative = create_negative(img)
    
    # Save the negative image
    save_path = os.path.join(os.path.expanduser("~"), "Desktop", "rhino_capture_negative.png")
    cv2.imwrite(save_path, negative)
    print(f"Negative image saved to: {save_path}")

if __name__ == "__main__":
    main()
