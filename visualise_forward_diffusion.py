from gaussian_diffusion import GaussianDiffusion
from PIL import Image
from torchvision.transforms import Compose, ToTensor, CenterCrop, Resize, Normalize
import torch
import matplotlib.pyplot as plt

def show_noisy_images(noisy_images):
        num_of_image_sets = len(noisy_images)
        num_of_images_in_set = len(noisy_images[0])
        image_size = noisy_images[0][0].size[0]

        full_image = Image.new('RGB', (image_size * num_of_images_in_set + (num_of_images_in_set - 1), image_size * num_of_image_sets + (num_of_image_sets - 1)))
        for set_index, image_set in enumerate(noisy_images):
            for image_index, image in enumerate(image_set):
                full_image.paste(image, (image_index * image_size + image_index, set_index * image_size + set_index))
  
        plt.imshow(full_image)
        plt.show()
        plt.axis('off')
        return full_image

image = Image.open('retinal_image_2_1.jpg')
image_size = 512

transform = Compose([
  Resize(image_size),  # resize smaller edge to image_size
  CenterCrop(image_size),  # make a square image with size image_size
  ToTensor(),  # convert to tensor with shape CHW and values in the range [0, 1]
  Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) # set the values to the range [-1, 1]
])
x0 = transform(image).unsqueeze(0)

diffusion = GaussianDiffusion(beta_scheduler='linear')
#diffusion = GaussianDiffusion(beta_scheduler='cosine', schedule_fn_kwargs={'s': 5})
#diffusion = GaussianDiffusion(beta_scheduler='sigmoid', schedule_fn_kwargs={'start': 0.01, 'end': 5, 'tau': 3})
show_noisy_images([[diffusion.get_noisy_image(x0, torch.tensor([t])) for t in [0, 50, 100, 150, 200]]])