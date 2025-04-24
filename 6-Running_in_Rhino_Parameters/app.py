import os, sys, threading, tempfile
import os.path as op

# -------------------------------------------
# make sure to change the cache path
# -------------------------------------------
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
import shutil  # For file copying

# -------------------------------------------
# make sure to change the env environment path
# -------------------------------------------

# Configure environment
CONDA_ENV = r'C:\Users\Hesham.Shawqy\anaconda3\envs\generative_ai'
sys.path.append(op.join(CONDA_ENV, r"Lib\site-packages"))
os.add_dll_directory(op.join(CONDA_ENV, r'Library\bin'))

from image_generation.image_generation import generate_from_rhino_view, initialize_models, is_loading_complete

# Track temporary files
temp_files = []

# viewport to bitmap to pil image
def capture_viewport():
    view = sc.doc.Views.ActiveView
    bitmap = view.CaptureToBitmap()
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
        temp_path = temp_file.name
        temp_files.append(temp_path)
    
    bitmap.Save(temp_path, Imaging.ImageFormat.Png)
    pil_image = Image.open(temp_path)
    
    return pil_image

# pil to bitmap to eto
def pil_to_eto_image(pil_image):
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
        temp_path = temp_file.name
        temp_files.append(temp_path)
    
    pil_image.save(temp_path, format='PNG')
    bitmap = Bitmap(temp_path)
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as eto_temp_file:
        eto_temp_path = eto_temp_file.name
        temp_files.append(eto_temp_path)
    
    bitmap.Save(eto_temp_path, Imaging.ImageFormat.Png)
    eto_bitmap = drawing.Bitmap(eto_temp_path)

    return eto_bitmap

# Clean up temporary files
def cleanup_temp_files():
    global temp_files
    for file_path in temp_files:
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error cleaning up temporary file {file_path}: {str(e)}")
    temp_files = []

# creating the UI components
def create_ui_controls():
    # Create image view
    image_view = forms.ImageView()
    image_view.BackgroundColor = drawing.Color.FromArgb(255, 255, 255)
    
    # Create buttons
    ai_button = forms.Button()
    ai_button.Text = "Generate AI Image"
    ai_button.Enabled = is_loading_complete()
    
    save_button = forms.Button()
    save_button.Text = "Save Image"
    save_button.Enabled = False
    
    # Create status label
    status_label = forms.Label()
    if not is_loading_complete():
        status_label.Text = "Loading AI models...this might take a few minutes"
        status_label.TextColor = drawing.Color.FromArgb(200, 0, 0)
    else:
        status_label.Text = "AI models loaded and ready"
        status_label.TextColor = drawing.Color.FromArgb(0, 128, 0)
    
    # Create prompt controls
    prompt_text = forms.TextArea()
    prompt_text.Text = "a stunning architectural visualization of the design, photorealistic render, 4k, high resolution"
    prompt_text.Height = 60
    
    # Create negative prompt controls
    negative_prompt_text = forms.TextArea()
    negative_prompt_text.Text = "ugly, low quality"
    negative_prompt_text.Height = 40
    
    # Create text inputs for parameters
    steps_input = forms.TextBox()
    steps_input.Text = "8"
    
    guidance_input = forms.TextBox()
    guidance_input.Text = "5.0"
    
    control_strength_input = forms.TextBox()
    control_strength_input.Text = "0.5"
    
    # Create seed control
    seed_numeric = forms.NumericUpDown()
    seed_numeric.MinValue = 0
    seed_numeric.MaxValue = 2147483647
    seed_numeric.Value = 0
    seed_numeric.DecimalPlaces = 0
    
    return (image_view, ai_button, save_button, status_label, prompt_text, 
            negative_prompt_text, steps_input, guidance_input, 
            control_strength_input, seed_numeric)


def show_image_dialog():
    initialize_models()
    
    # Create form
    form = forms.Form()
    form.Title = "Rhino AI Image Generator"
    form.ClientSize = drawing.Size(600, 700)
    form.Padding = drawing.Padding(10)
    form.Topmost = True
    
    # Get UI controls
    (image_view, ai_button, save_button, status_label, prompt_text,
     negative_prompt_text, steps_input, guidance_input,
     control_strength_input, seed_numeric) = create_ui_controls()
    
    # Create the main layout
    layout = forms.DynamicLayout()
    layout.Padding = drawing.Padding(10)
    layout.Spacing = drawing.Size(5, 5)
    
    layout.Add(image_view, yscale=True)
    
    # Prompt section
    prompt_label = forms.Label()
    prompt_label.Text = "Text Prompt"
    layout.Add(prompt_label)
    layout.Add(prompt_text)
    
    # Negative prompt section
    negative_prompt_label = forms.Label()
    negative_prompt_label.Text = "Negative Prompt"
    layout.Add(negative_prompt_label)
    layout.Add(negative_prompt_text)
    
    # Parameter inputs section
    params_layout = forms.TableLayout()
    params_layout.Spacing = drawing.Size(5, 5)
    
    # Inference steps row
    steps_label = forms.Label()
    steps_label.Text = "Inference Steps (4-12)"
    steps_cell = forms.TableCell()
    steps_cell.Control = steps_label
    steps_cell.ScaleWidth = True
    
    steps_input_cell = forms.TableCell()
    steps_input_cell.Control = steps_input
    steps_input_cell.ScaleWidth = False
    
    steps_row = forms.TableRow()
    steps_row.Cells.Add(steps_cell)
    steps_row.Cells.Add(steps_input_cell)
    params_layout.Rows.Add(steps_row)
    
    # Guidance scale row
    guidance_label = forms.Label()
    guidance_label.Text = "Guidance Scale (0.0-10.0)"
    guidance_cell = forms.TableCell()
    guidance_cell.Control = guidance_label
    guidance_cell.ScaleWidth = True
    
    guidance_input_cell = forms.TableCell()
    guidance_input_cell.Control = guidance_input
    guidance_input_cell.ScaleWidth = False
    
    guidance_row = forms.TableRow()
    guidance_row.Cells.Add(guidance_cell)
    guidance_row.Cells.Add(guidance_input_cell)
    params_layout.Rows.Add(guidance_row)
    
    # Control strength row
    control_label = forms.Label()
    control_label.Text = "Control Strength (0.0-1.0)"
    control_cell = forms.TableCell()
    control_cell.Control = control_label
    control_cell.ScaleWidth = True
    
    control_input_cell = forms.TableCell()
    control_input_cell.Control = control_strength_input
    control_input_cell.ScaleWidth = False
    
    control_row = forms.TableRow()
    control_row.Cells.Add(control_cell)
    control_row.Cells.Add(control_input_cell)
    params_layout.Rows.Add(control_row)
    
    # Seed row
    seed_label = forms.Label()
    seed_label.Text = "Seed (0 = random)"
    seed_cell = forms.TableCell()
    seed_cell.Control = seed_label
    seed_cell.ScaleWidth = True
    
    seed_input_cell = forms.TableCell()
    seed_input_cell.Control = seed_numeric
    seed_input_cell.ScaleWidth = False
    
    seed_row = forms.TableRow()
    seed_row.Cells.Add(seed_cell)
    seed_row.Cells.Add(seed_input_cell)
    params_layout.Rows.Add(seed_row)
    
    layout.Add(params_layout)
    layout.Add(status_label)
    
    # Create button layout
    button_layout = forms.TableLayout()
    button_layout.Spacing = drawing.Size(5, 0)
    
    empty_cell = forms.TableCell()
    empty_cell.ScaleWidth = True
    
    ai_button_cell = forms.TableCell()
    ai_button_cell.Control = ai_button
    ai_button_cell.ScaleWidth = False
    
    save_button_cell = forms.TableCell()
    save_button_cell.Control = save_button
    save_button_cell.ScaleWidth = False
    
    button_row = forms.TableRow()
    button_row.Cells.Add(empty_cell)
    button_row.Cells.Add(ai_button_cell)
    button_row.Cells.Add(save_button_cell)
    
    button_layout.Rows.Add(button_row)
    layout.Add(button_layout)
    
    form.Content = layout
    
    # Store generated image and temp file path
    generated_image = None
    temp_image_path = None
    
    # Setup loading timer
    check_loading_timer = forms.UITimer()
    check_loading_timer.Interval = 1.0
    
    # Timer event handler
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
            
            # Validate inputs
            try:
                steps = int(steps_input.Text)
                if not (4 <= steps <= 12):
                    raise ValueError("Inference steps must be between 4 and 12")
                
                guidance = float(guidance_input.Text)
                if not (0.0 <= guidance <= 10.0):
                    raise ValueError("Guidance scale must be between 0.0 and 10.0")
                
                control = float(control_strength_input.Text)
                if not (0.0 <= control <= 1.0):
                    raise ValueError("Control strength must be between 0.0 and 1.0")
                
                seed = int(seed_numeric.Value) if seed_numeric.Value > 0 else None
            except ValueError as ve:
                status_label.Text = f"Error: {str(ve)}"
                return
            
            captured_image = capture_viewport()
            prompt = prompt_text.Text
            negative_prompt = negative_prompt_text.Text
            
            status_label.Text = "Generating AI image..."
            ai_button.Enabled = False
            save_button.Enabled = False
            
            def generate_in_background():
                nonlocal generated_image, temp_image_path
                try:
                    ai_image = generate_from_rhino_view(
                        captured_image, 
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=steps,
                        guidance_scale=guidance,
                        control_strength=control,
                        seed=seed
                    )
                    
                    generated_image = ai_image
                    
                    # Save to temporary file right away
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                        temp_image_path = temp_file.name
                        temp_files.append(temp_image_path)
                    
                    ai_image.save(temp_image_path, format='PNG')
                    
                    def update_ui():
                        if image_view.Image is not None:
                            image_view.Image = None
                        
                        image_view.Image = pil_to_eto_image(ai_image)
                        
                        ai_button.Enabled = True
                        save_button.Enabled = True
                        status_label.Text = "AI models loaded and ready"
                        
                        print("AI image generated and displayed")
                    
                    forms.Application.Instance.Invoke(update_ui)
                    
                except Exception as ex:
                    def show_error():
                        print(f"Error generating AI image: {str(ex)}")
                        status_label.Text = f"Error: {str(ex)}"
                        ai_button.Enabled = True
                    
                    forms.Application.Instance.Invoke(show_error)
            
            threading.Thread(target=generate_in_background).start()
            
        except Exception as e:
            print(f"Error generating AI image: {str(e)}")
            status_label.Text = f"Error: {str(e)}"
            ai_button.Enabled = True
    
    def on_save_button_click(sender, e):
        nonlocal temp_image_path
        if temp_image_path is None or not os.path.exists(temp_image_path):
            status_label.Text = "No image to save"
            return
        
        # Disable the save button during the operation
        save_button.Enabled = False
        status_label.Text = "Preparing to save..."
        
        # Use a fixed path in the user's documents folder instead of a dialog
        try:
            # Get user documents folder
            documents_folder = os.path.expanduser("~/Documents")
            rhino_ai_folder = os.path.join(documents_folder, "RhinoAI")
            
            # Create the folder if it doesn't exist
            if not os.path.exists(rhino_ai_folder):
                os.makedirs(rhino_ai_folder)
            
            # Create a filename with timestamp
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"rhino_ai_{timestamp}.png"
            file_path = os.path.join(rhino_ai_folder, file_name)
            
            # Copy the file
            shutil.copy2(temp_image_path, file_path)
            
            status_label.Text = f"Image saved to {file_path}"
            print(f"Image saved to {file_path}")
        except Exception as ex:
            status_label.Text = f"Error saving image: {str(ex)}"
            print(f"Error saving image: {str(ex)}")
        finally:
            save_button.Enabled = True
    
    def on_form_closing(sender, e):
        if image_view.Image is not None:
            image_view.Image = None
        check_loading_timer.Stop()
        cleanup_temp_files()
    
    # Connect events
    ai_button.Click += on_ai_button_click
    save_button.Click += on_save_button_click
    form.Closing += on_form_closing
    
    form.Show()
    
    return form

if __name__ == "__main__":
    show_image_dialog()