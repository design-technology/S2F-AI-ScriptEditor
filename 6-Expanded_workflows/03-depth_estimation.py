from transformers import pipeline
import torch
from PIL import Image
import matplotlib.pyplot as plt

# Set device to GPU if available, else use CPU
device = 0 if torch.cuda.is_available() else -1

# Load the depth estimation model
checkpoint = "depth-anything/Depth-Anything-V2-base-hf"
pipe = pipeline("depth-estimation", model=checkpoint, device=device)

# Load your local image
image_path = "C:/Users/vr_service/Pictures/water.jpg"
image = Image.open(image_path)

# Predict the depth map
predictions = pipe(image)
depth = predictions["depth"]  # This is a PIL Image

# Get the size of the depth image (width, height)
width, height = depth.size

# Set up figure with no borders or padding
fig = plt.figure(frameon=False)
fig.set_size_inches(width / 100, height / 100)  # Match image size in inches
ax = plt.Axes(fig, [0., 0., 1., 1.])  # Full canvas
ax.set_axis_off()
fig.add_axes(ax)

# Show depth map in grayscale and save it
ax.imshow(depth, cmap='gray')
output_path = "C:/Users/vr_service/Pictures/banffdepth_2.png"
fig.savefig(output_path, dpi=100, bbox_inches='tight', pad_inches=0)

print(f" Saved!: {output_path}")


