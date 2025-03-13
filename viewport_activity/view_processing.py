#r: pillow

import rhinoscriptsyntax as rs
import os
import tempfile
import Rhino
import scriptcontext as sc
import System.Drawing as drawing
import System.Windows.Forms as forms
from System.Drawing import Bitmap, Size, Rectangle
from System.IO import MemoryStream
from PIL import Image, ImageOps
import io
import clr
import time

def capture_viewport():
    # this function captures the active viewport and returns it as a PIL Image
    # first create a path to save the image temporarily
    temp_file = os.path.join(os.path.expanduser("~"), "Desktop", "rhino_temp_capture.png")
    
    # use Rhino's command to capture the viewport to a file
    rs.Command(f'_-ViewCaptureToFile "{temp_file}" _Enter', False)
    
    # load the image with PIL
    pil_image = Image.open(temp_file)
    
    # make a copy in memory so we can close the file handle
    pil_image_copy = pil_image.copy()
    pil_image.close()
    
    # try to delete the temp file - might not work if it's locked
    try:
        os.remove(temp_file)
    except:
        pass
        
    return pil_image_copy

def create_negative_image(pil_image):
    # super simple way to create a negative image using PIL
    # just subtract from 255 basically
    return ImageOps.invert(pil_image)

def pil_to_bitmap(pil_image):
    # need to convert PIL image to Bitmap for the Windows form
    # save to a temp file with timestamp to avoid conflicts
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    temp_file = os.path.join(os.path.expanduser("~"), "Desktop", f"temp_negative_{timestamp}.png")
    pil_image.save(temp_file, format='PNG')
    
    # load it as a Bitmap
    bitmap = Bitmap.FromFile(temp_file)
    
    # try to clean up the temp file
    try:
        os.remove(temp_file)
    except:
        pass
    
    return bitmap

def dispose_image(picture_box):
    # this fixes the file locking issue by properly disposing the image
    if picture_box.Image is not None:
        current_image = picture_box.Image
        picture_box.Image = None
        current_image.Dispose()

def show_image_dialog():
    # create a simple window form
    form = forms.Form()
    form.Text = "Rhino Image Viewer"
    form.Size = Size(800, 600)
    form.StartPosition = forms.FormStartPosition.CenterScreen
    
    # add a picture box to display the image
    picture_box = forms.PictureBox()
    picture_box.Dock = forms.DockStyle.Fill
    picture_box.SizeMode = forms.PictureBoxSizeMode.Zoom
    picture_box.BackColor = drawing.Color.LightGray
    
    # add a button to trigger the capture
    button = forms.Button()
    button.Text = "Capture and Create Negative"
    button.Dock = forms.DockStyle.Bottom
    button.Height = 40
    
    # add the controls to the form
    form.Controls.Add(picture_box)
    form.Controls.Add(button)
    
    # what happens when the button is clicked
    def on_button_click(sender, event):
        try:
            # first clean up any existing image to avoid file locks
            dispose_image(picture_box)
            
            # capture the viewport as a PIL image
            pil_image = capture_viewport()
            
            # create the negative version
            negative_pil = create_negative_image(pil_image)
            
            # convert to bitmap for display in the form
            negative_bitmap = pil_to_bitmap(negative_pil)
            
            # show it in the picture box
            picture_box.Image = negative_bitmap
            
            # save to desktop with timestamp so we don't overwrite files
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(os.path.expanduser("~"), "Desktop", f"rhino_capture_negative_{timestamp}.png")
            negative_pil.save(save_path, format='PNG')
            
            print(f"Negative image saved to: {save_path}")
            
        except Exception as e:
            print(f"Error: {str(e)}")
    
    # clean up when the form is closing
    def on_form_closing(sender, event):
        dispose_image(picture_box)
    
    # connect the event handlers
    button.Click += on_button_click
    form.FormClosing += on_form_closing
    
    # show the form and wait for it to close
    form.ShowDialog()

# run the main function
show_image_dialog()