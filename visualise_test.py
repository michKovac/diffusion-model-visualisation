from gaussian_diffusion import GaussianDiffusion
from torchvision.transforms import Compose, ToTensor, CenterCrop, Resize, Normalize
import torch
from PIL import Image
from utils import show_noisy_images

# Load the image
image = Image.open('retinal_image_2_1.jpg')
image_size = 512

# Define the transformation pipeline
transform = Compose([
    Resize(image_size),  # Resize smaller edge to image_size
    CenterCrop(image_size),  # Make a square image with size image_size
    ToTensor(),  # Convert to tensor with shape CHW and values in the range [0, 1]
    Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # Set the values to the range [-1, 1]
])

# Apply the transformation to the image and add a batch dimension
x0 = transform(image).unsqueeze(0)

# Initialize the GaussianDiffusion model
diffusion = GaussianDiffusion(beta_scheduler='linear')
# diffusion = GaussianDiffusion(beta_scheduler='cosine', schedule_fn_kwargs={'s': 5})
# diffusion = GaussianDiffusion(beta_scheduler='sigmoid', schedule_fn_kwargs={'start': 0.01, 'end': 5, 'tau': 3})

# Define the timesteps to sample
t_sample = [0, 10, 20, 50, 100, 200, 300, 400, 999]

# Generate and display noisy images at different timesteps
show_noisy_images([[diffusion.get_noisy_image(x0, torch.tensor([t])) for t in t_sample]], t_sample)
