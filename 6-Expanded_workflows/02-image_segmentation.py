from transformers import pipeline
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
import torch  # Make sure torch is imported

# Cache path, adjust as needed
cache_path = r"C:\huggingface"  # Inset your hugging face cache path here

file_path = r"C:/Users/vr_service/Pictures/construction2.jpg"  # Replace this with the actual path to your local image file
image = Image.open(file_path)
image.show()

# Initialize the segmentation pipeline
semantic_segmentation = pipeline(
    "image-segmentation", 
    model="nvidia/segformer-b1-finetuned-cityscapes-1024-1024", # insert model name here
    cache_dir=cache_path
)

# Check if CUDA (GPU) is available
device = 0 if torch.cuda.is_available() else -1  # Set device to GPU if available, else use CPU

# Run the segmentation model on GPU (if available)
semantic_segmentation.model.to(device)

# Run the segmentation model
results = semantic_segmentation(image)

# Output the results to inspect the structure
print("Segmentation results:", results)


combined_mask = np.zeros((image.height, image.width, 3), dtype=np.uint8)  # RGB image

# Loop through the results to extract and colorize each mask
for i, result in enumerate(results):
    mask = result.get('mask')  # Extract the mask (should be a PIL Image)
    
    if mask is not None:
        # Convert mask to a NumPy array if it's a PIL Image
        mask_array = np.array(mask)
        
        # Generate a random color for each mask (for visualization)
        color = [random.randint(0, 255) for _ in range(3)]  # Random color for each mask

        # Colorize the mask: Apply color to the mask where the mask is non-zero
        mask_colorized = np.zeros_like(combined_mask)
        mask_colorized[mask_array > 0] = color 

        # Combine this mask into the final combined mask (overlay the colorized masks)
        combined_mask = np.maximum(combined_mask, mask_colorized)

# Convert combined mask to a PIL Image for display
combined_mask_image = Image.fromarray(combined_mask)

# Display the combined mask
plt.figure(figsize=(6, 6))
plt.imshow(combined_mask_image)
plt.title("Compiled Colorized Segmentation Masks")
plt.axis('off')  # Hide the axis
plt.show()
