#r: opencv-python
#r: numpy

import Rhino
import scriptcontext as sc
import rhinoscriptsyntax as rs
import os
import tempfile
import cv2
import System.Threading.Tasks as tasks
import time

def capture_viewport(filepath=None):
    # Grab viewport screenshot
    rs.Redraw()
    
    if not filepath:
        temp_dir = tempfile.gettempdir()
        filepath = os.path.join(temp_dir, f"rhino_capture_{int(time.time())}.png")
    
    rs.Command(f'_-ViewCaptureToFile "{filepath}" _Enter', False)
    
    time.sleep(0.1)
    
    img = cv2.imread(filepath)
    
    if img is None:
        print(f"Couldn't read the image from {filepath}")
        return None
            
    return img

def create_negative(img):
    # Invert colors
    if img is None:
        return None
    return 255 - img

def save_images():
    # Save original and negative images
    temp_filepath = os.path.join(tempfile.gettempdir(), f"rhino_temp_capture_{int(time.time())}.png")
    
    img = capture_viewport(temp_filepath)
    
    if img is None:
        print("Couldn't capture the viewport")
        return
    
    original_save_path = os.path.join(os.path.expanduser("~"), "Desktop", "rhino_capture.png")
    cv2.imwrite(original_save_path, img)
    print(f"Saved the original image to: {original_save_path}")
    
    negative = create_negative(img)
    
    if negative is None:
        print("Failed creating the negative image")
        return
    
    negative_save_path = os.path.join(os.path.expanduser("~"), "Desktop", "rhino_capture_negative.png")
    cv2.imwrite(negative_save_path, negative)
    print(f"Saved the negative image to: {negative_save_path}")
    
    try:
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
    except:
        pass

class ViewportMonitor:
    def __init__(self):
        self.Enabled = True
        self.last_update_time = 0
        self.update_interval = 0.5
        self.is_processing = False
        
        # Hook into display pipeline
        Rhino.Display.DisplayPipeline.CalculateBoundingBox += self.on_viewport_changed
        print("Started monitoring the viewport")
    
    def Disable(self):
        # Unhook event handler
        Rhino.Display.DisplayPipeline.CalculateBoundingBox -= self.on_viewport_changed
        self.Enabled = False
        print("Stopped monitoring the viewport")
    
    def on_viewport_changed(self, sender, e):
        # Skip if already processing
        if self.is_processing:
            return
            
        # Throttle updates
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval:
            return
            
        self.last_update_time = current_time
        self.is_processing = True
        
        try:
            print("The viewport changed - capturing new images...")
            rs.Redraw()
            
            # Process in background
            tasks.Task.Run(lambda: self.process_capture())
        except Exception as ex:
            print(f"Error: {ex}")
            self.is_processing = False
    
    def process_capture(self):
        try:
            rs.Redraw()
            save_images()
            print("Images saved.")
        except Exception as ex:
            print(f"Error saving: {ex}")
        finally:
            self.is_processing = False

def toggle_monitoring():
    # Toggle viewport monitoring
    if 'viewport_monitor' in sc.sticky and sc.sticky['viewport_monitor'].Enabled:
        sc.sticky['viewport_monitor'].Disable()
        print("Stopped monitoring")
    else:
        monitor = ViewportMonitor()
        sc.sticky['viewport_monitor'] = monitor

# Run when executed
if __name__ == '__main__':
    toggle_monitoring()