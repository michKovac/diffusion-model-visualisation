import torch
from schedulers import linear_beta_schedule, cosine_beta_schedule, sigmoid_beta_schedule
from torchvision.transforms import Compose, ToTensor, CenterCrop, Resize, Normalize
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage

class GaussianDiffusion:
    """
    Forward Process class for denoising diffusion probabilistic models (DDPM).
    """

    SCHEDULER_MAPPING = {
        "linear": linear_beta_schedule,
        "cosine": cosine_beta_schedule,
        "sigmoid": sigmoid_beta_schedule,
    }

    def __init__(self, 
                 num_time_steps: int = 1000,
                 schedule_fn_kwargs: dict | None = None, 
                 beta_start: float = 1e-4, 
                 beta_end: float = 0.02,
                 beta_scheduler: str = 'linear'):
        """
        Initialize the GaussianDiffusion class.

        Args:
        - num_time_steps (int): The number of time steps for the diffusion process.
        - schedule_fn_kwargs (dict | None): Additional keyword arguments for the beta schedule function.
        - beta_start (float): The initial value of beta.
        - beta_end (float): The final value of beta.
        - beta_scheduler (str): The type of beta scheduler to use. Options are 'linear', 'cosine', and 'sigmoid'.

        Raises:
        - ValueError: If an unknown beta scheduler is provided.
        """

        self.beta_scheduler_fn = self.SCHEDULER_MAPPING.get(beta_scheduler)
        if self.beta_scheduler_fn is None:
            raise ValueError(f"Unknown beta schedule: {beta_scheduler}")
        if schedule_fn_kwargs is None:
            schedule_fn_kwargs = {}
        self.betas_t = self.beta_scheduler_fn(num_time_steps, **schedule_fn_kwargs)

        self.alphas_t = 1. - self.betas_t
        self.alphas_bar_t = torch.cumprod(self.alphas_t, dim=0)
        self.alphas_bar_t_minus_1 = torch.cat((torch.tensor([0]), self.alphas_bar_t[:-1]))
        self.one_over_sqrt_alphas_t = 1. / torch.sqrt(self.alphas_t)
        self.sqrt_alphas_bar_t = torch.sqrt(self.alphas_bar_t)
        self.sqrt_1_minus_alphas_bar_t = torch.sqrt(1. - self.alphas_bar_t)
        self.posterior_variance = (1. - self.alphas_bar_t_minus_1) / (1. - self.alphas_bar_t) * self.betas_t
        
        self.reverse_transform_pil = Compose([
            Normalize(mean=(-1, -1, -1), std=(2, 2, 2)),
            ToPILImage()
        ])
        self.reverse_transform_tensor = Compose([
            Normalize(mean=(-1, -1, -1), std=(2, 2, 2)),
        ])

    def sample_by_t(self, tensor_to_sample, timesteps, x_shape):
        """
        Sample tensor values based on the given time steps.

        Args:
        - tensor_to_sample (torch.Tensor): The tensor to sample from.
        - timesteps (torch.Tensor): The time steps to sample at.
        - x_shape (tuple): The shape of the tensor.

        Returns:
        - sampled_tensor (torch.Tensor): The sampled tensor.

        """
        batch_size = timesteps.shape[0]
        sampled_tensor = tensor_to_sample.gather(-1, timesteps.cpu())
        sampled_tensor = torch.reshape(sampled_tensor, (batch_size,) + (1,) * (len(x_shape) - 1))
        return sampled_tensor.to(timesteps.device)

    def sample_q(self, x0, t, noise=None):
        """
        Sample from the q distribution.

        Args:
        - x0 (torch.Tensor): The initial image tensor.
        - t (torch.Tensor): The time step to sample at.
        - noise (torch.Tensor | None): The noise tensor. If None, a new noise tensor will be generated.

        Returns:
        - x_t (torch.Tensor): The sampled tensor from the q distribution.

        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alphas_bar_t_sampled = self.sample_by_t(self.sqrt_alphas_bar_t, t, x0.shape)
        sqrt_1_minus_alphas_bar_t_sampled = self.sample_by_t(self.sqrt_1_minus_alphas_bar_t, t, x0.shape)

        x_t = (sqrt_alphas_bar_t_sampled * x0) + (sqrt_1_minus_alphas_bar_t_sampled * noise)
        return x_t
    
    def get_noisy_image(self, x0, t):
        """
        Generate a noisy image based on the given initial image and time step.

        Args:
        - x0 (torch.Tensor): The initial image tensor.
        - t (int): The time step.

        Returns:
        - noise_image (PIL.Image.Image): The generated noisy image.

        """
        transform = self.reverse_transform_pil
        x_noisy = self.sample_q(x0, t)
        noise_image = transform(x_noisy.squeeze())
        return noise_image

