from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy
import cv2



def show_noisy_images(noisy_images, t_sample, figure_scale_factor=4):
    """
    Displays and saves a composite image and histograms of noisy images at different timesteps.

    Args:
        noisy_images (list of list of PIL.Image): A 2D list where each sublist contains a set of noisy images.
        t_sample (int): The timestep sample to be used for adding images and plotting histograms.
        figure_scale_factor (int, optional): Scale factor for the figure size. Default is 4.

    Returns:
        PIL.Image: The composite image created from the noisy images.
    """
    # Determine the number of image sets and the number of images in each set
    num_of_image_sets = len(noisy_images)
    num_of_images_in_set = len(noisy_images[0])
    image_size = noisy_images[0][0].size[0]

    params = {'mathtext.default': 'regular' }          
    plt.rcParams.update(params)

    # Create a new blank image to hold the composite of all noisy images
    full_image = Image.new('RGB', (image_size * num_of_images_in_set + (num_of_images_in_set - 1), image_size * num_of_image_sets + (num_of_image_sets - 1)))
    
    # Set up a matplotlib figure and axes for plotting histograms
    figure_size = (num_of_images_in_set * figure_scale_factor, figure_scale_factor)
    fig, axes = plt.subplots(num_of_image_sets, num_of_images_in_set, figsize=figure_size)
    fig.tight_layout(pad=3.0)
    
    # Iterate over each set of noisy images and each image within the set
    for set_index, image_set in enumerate(noisy_images):
        for image_index, image in enumerate(image_set):
            # Add the image to the composite image
            add_image(t_sample, image_size, full_image, set_index, image_index, image)
            # Plot the histogram of the image
            plot_hist(t_sample, axes, image_index, image)
    
    # Save the composite image and the histogram figure to files
    full_image.save("forward_diffusion_image.jpg")
    plt.savefig("forward_diffusion_histogram.jpg")
    plt.show()
    
    return full_image

def add_image(t_sample, image_size, full_image, set_index, image_index, image):
    """
    Adds an image to the composite image with annotations.

    Args:
        t_sample (list of int): List of timesteps.
        image_size (int): Size of the image.
        full_image (PIL.Image): The composite image.
        set_index (int): Index of the image set.
        image_index (int): Index of the image within the set.
        image (PIL.Image): The image to be added.
    """
    # Load fonts for annotations
    myFont = ImageFont.truetype('DejaVuSansMono-Bold.ttf', 60)
    Number_font = ImageFont.truetype('DejaVuSansMono-Bold.ttf', 40)
    
    # Draw annotations on the image
    I1 = ImageDraw.Draw(image)
    I1.text((10, 10), f"x", fill="white", font=myFont)
    I1.text((50, 50), f"{t_sample[image_index]+1}", fill="white", font=Number_font)
    
    # Paste the annotated image into the composite image
    full_image.paste(image, (image_index * image_size + image_index, set_index * image_size + set_index))

def plot_hist(t_sample, axes, image_index, image):
    """
    Plots the histogram of an image.

    Args:
        t_sample (list of int): List of timesteps.
        axes (matplotlib.axes.Axes): Axes object for plotting.
        image_index (int): Index of the image within the set.
        image (PIL.Image): The image to be plotted.
    """
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(to_cv2(image), cv2.COLOR_BGR2GRAY)
    
    # Calculate the histogram of the grayscale image
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 255])

    # Plot the histogram
    axes[image_index].plot(hist, color='black')
    axes[image_index].set_xlim([0, 255])
    axes[image_index].set_ylim([0, max(hist) * 1.1])
    axes[image_index].set_title(f'$x_{{{t_sample[image_index]-1}}}$', fontsize=16)
    axes[image_index].set_yticks = ([0, hist.max()/2, hist.max()])
    axes[image_index].set_xtickslabels = ([0, 255/2,255])

def to_cv2(pil_img):
    """
    Converts a PIL image to an OpenCV image.

    Args:
        pil_img (PIL.Image): The PIL image to be converted.

    Returns:
        numpy.ndarray: The converted OpenCV image.
    """
    open_cv_image = numpy.array(pil_img)
    # Convert RGB to BGR
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    return open_cv_image