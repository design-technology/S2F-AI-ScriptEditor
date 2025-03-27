import os, sys, threading, tempfile
import os.path as op
cache_path = "Z:\\Development Projects\\huggingface"
os.environ["TRANSFORMERS_CACHE"] = cache_path
os.environ["HF_HUB_CACHE"] = cache_path
os.environ["HUGGINGFACE_HUB_CACHE"] = cache_path 
os.environ["HF_HOME"] = cache_path

import rhinoscriptsyntax as rs
import Rhino
import scriptcontext as sc
import Eto.Forms as forms
import Eto.Drawing as drawing
from System.Drawing import Bitmap, Imaging
from PIL import Image, ImageOps

# Configure environment
CONDA_ENV = r'C:\Users\Hesham.Shawqy\anaconda3\envs\generative_ai'
sys.path.append(op.join(CONDA_ENV, r"Lib\site-packages"))
os.add_dll_directory(op.join(CONDA_ENV, r'Library\bin'))

from image_generation import generate_from_rhino_view, initialize_models, is_loading_complete
# viewport to bitmap to pil image
def capture_viewport():
    view = sc.doc.Views.ActiveView
    bitmap = view.CaptureToBitmap()
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
        temp_path = temp_file.name
    
    bitmap.Save(temp_path, Imaging.ImageFormat.Png)
    pil_image = Image.open(temp_path)
    
    return pil_image

# pil to bitmap to eto
def pil_to_eto_image(pil_image):
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
        temp_path = temp_file.name
    
    pil_image.save(temp_path, format='PNG')
    bitmap = Bitmap(temp_path)
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as eto_temp_file:
        eto_temp_path = eto_temp_file.name
    
    bitmap.Save(eto_temp_path, Imaging.ImageFormat.Png)
    eto_bitmap = drawing.Bitmap(eto_temp_path)

    return eto_bitmap

# Initialize sticky dictionary for form reference
# if 'negative_viewport_form' not in sc.sticky:
#     sc.sticky['negative_viewport_form'] = None

def create_ui_controls():
    # Create image view
    image_view = forms.ImageView()
    image_view.BackgroundColor = drawing.Color.FromArgb(255, 255, 255)
    
    
    ai_button = forms.Button()
    ai_button.Text = "Generate AI Image"
    # only active when the model is loaded
    ai_button.Enabled = is_loading_complete()
    
    # Create status label
    status_label = forms.Label()
    if not is_loading_complete():
        status_label.Text = "Loading AI models...this might take a few minutes"
        status_label.TextColor = drawing.Color.FromArgb(200, 0, 0)
    else:
        status_label.Text = "AI models loaded and ready"
        status_label.TextColor = drawing.Color.FromArgb(0, 128, 0)
    
    # Create prompt controls
    prompt_label = forms.Label()
    prompt_label.Text = "Text Prompt"
    
    prompt_text = forms.TextArea()
    prompt_text.Text = "a stunning architectural visualization of the design, photorealistic render, 4k, high resolution"
    prompt_text.Height = 60
    
    return image_view, ai_button, status_label, prompt_label, prompt_text

def show_image_dialog():
    
    initialize_models()
    
    # Create form
    form = forms.Form()
    form.Title = "Rhino AI Image Generator"
    form.ClientSize = drawing.Size(600, 600)
    form.Padding = drawing.Padding(10)
    # bring the form forward above other applications
    form.Topmost = True
    
    # Get UI controls
    image_view, ai_button, status_label, prompt_label, prompt_text = create_ui_controls()
    
    # Create the main layout
    layout = forms.DynamicLayout()
    layout.Padding = drawing.Padding(10)
    layout.Spacing = drawing.Size(5, 5)
    
    layout.Add(image_view, yscale=True)
    layout.Add(prompt_label)
    layout.Add(prompt_text)
    layout.Add(status_label)
    
    # Create button layout
    button_layout = forms.DynamicLayout()
    button_layout.Spacing = drawing.Size(5, 0)
    button_layout.AddRow(None, ai_button)
    
    layout.Add(button_layout)
    form.Content = layout
    
    # Store image capture
    captured_image = None
    
    # Setup loading timer
    check_loading_timer = forms.UITimer()
    check_loading_timer.Interval = 1.0
    
    #timer event handler
    def update_loading_status(sender, e):
        if is_loading_complete():
            status_label.Text = "AI models loaded and ready"
            status_label.TextColor = drawing.Color.FromArgb(0, 128, 0)
            ai_button.Enabled = True
            check_loading_timer.Stop()
    
    check_loading_timer.Elapsed += update_loading_status
    check_loading_timer.Start()
    
    
    def on_ai_button_click(sender, e):
        try:
            if not is_loading_complete():
                print("AI models are still loading. Please wait...")
                return
            
            nonlocal captured_image
            captured_image = capture_viewport()
            
            prompt = prompt_text.Text
            status_label.Text = "Generating AI image..."
            ai_button.Enabled = False
            
            def generate_in_background():
                try:
                    ai_image = generate_from_rhino_view(captured_image, prompt=prompt)
                    
                    def update_ui():
                        nonlocal ai_image
                        if image_view.Image is not None:
                            image_view.Image = None
                        
                        image_view.Image = pil_to_eto_image(ai_image)
                        
                        ai_button.Enabled = True
                        status_label.Text = "AI models loaded and ready"
                        
                        print("AI image generated and displayed")
                    
                    forms.Application.Instance.Invoke(update_ui)
                    
                except Exception as ex:
                    def show_error():
                        nonlocal ex
                        print(f"Error generating AI image: {str(ex)}")
                        status_label.Text = f"Error: {str(ex)}"
                        ai_button.Enabled = True
                    
                    forms.Application.Instance.Invoke(show_error)
            
            threading.Thread(target=generate_in_background).start()
            
        except Exception as e:
            print(f"Error generating AI image: {str(e)}")
            status_label.Text = f"Error: {str(e)}"
            ai_button.Enabled = True
    
    def on_form_closing(sender, e):
        if image_view.Image is not None:
            image_view.Image = None
        check_loading_timer.Stop()
    
    # Connect events
    ai_button.Click += on_ai_button_click
    form.Closing += on_form_closing
    
    form.Show()
    
    return form

if __name__ == "__main__":
    show_image_dialog()