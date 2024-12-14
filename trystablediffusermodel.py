from diffusers import StableDiffusionPipeline
import os
import torch

# Folder to save generated images
output_folder = "static/images"


# Function to generate high-quality images
def stablediffusion_generate_high_quality_image(prompt, guidance_scale=10, num_inference_steps=150,
                                                seed=None, height=2160, width=3840):
    """
    Generates a high-quality image using Stable Diffusion.

    Args:
        prompt (str): Text prompt for the image generation.
        guidance_scale (float): How much the model should adhere to the prompt.
        num_inference_steps (int): Number of diffusion steps for generation (higher = better quality).
        seed (int, optional): Random seed for reproducibility.
        height (int): Height of the image (pixels).
        width (int): Width of the image (pixels).

    Returns:
        PIL.Image.Image: Generated image.
    """
    # Set random seed for reproducibility if provided
    if seed is not None:
        torch.manual_seed(seed)

    # Load the Stable Diffusion model pipeline
    model = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4').to('cuda')

    # Generate the image
    image = model(
        prompt=prompt,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps
    ).images[0]

    return image



os.makedirs(output_folder, exist_ok=True)

prompt = "a japanese street in tokyo with anime characters and a man in a red shirt and a hat on top of him , cats , cars , street food being made by magic with rain and neon accents"

# Generate 4K image
print(f"Generating 4K image for prompt: \"{prompt}\"")
image = stablediffusion_generate_high_quality_image(
    prompt=prompt,
    height=1024,
    width=1024,
    guidance_scale=12,
    num_inference_steps=500,
    seed=42
)

# Save the image to the output directory
output_path = os.path.join(output_folder, "japan.png")
image.save(output_path)
print(f"4K Image saved at: {output_path}")
