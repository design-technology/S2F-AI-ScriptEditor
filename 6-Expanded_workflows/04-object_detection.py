import torch.version
from transformers import pipeline
from PIL import Image, ImageDraw, ImageFont
import torch

# Check if CUDA (GPU) is available and use it for inference if possible
device = 0 if torch.cuda.is_available() else -1

# Load the zero-shot object detection model
checkpoint = "google/owlv2-base-patch16-ensemble"
detector = pipeline(model=checkpoint, task="zero-shot-object-detection", device=device)

# Path to the image
image_path = r"C:/Users/vr_service/Pictures/trukz.jpg"
image = Image.open(image_path).convert("RGB")

# Run zero-shot detection on the image with specific labels
predictions = detector(
    image,
    candidate_labels=["tractor", "crane",],
)

# Draw bounding boxes and labels
draw = ImageDraw.Draw(image)

# Optional: Load a font (if available)
try:
    font = ImageFont.truetype("arial.ttf", size=16)
except IOError:
    font = ImageFont.load_default()

for prediction in predictions:
    score = prediction["score"]
    if score <= 0.2:
        continue  # Skip low-confidence detections

    box = prediction["box"]
    label = prediction["label"]

    xmin, ymin, xmax, ymax = box.values()

    # Draw rectangle
    draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=6)

    # Prepare label text
    text = f"{label}: {round(score, 2)}"
    
    # Get text bounding box
    bbox = font.getbbox(text)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Draw filled background for label
    draw.rectangle(
        (xmin, ymin - text_height, xmin + text_width, ymin),
        fill="white"
    )
    draw.text((xmin, ymin - text_height), text, fill="black", font=font)

# Show the image with annotations
image.show()
