import sys
import os
import torch
import random
import numpy as np
from typing import List
from diffusers import StableDiffusionPipeline, StableDiffusion3Pipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionSafetyChecker,
)

sys.path.append("./third_party/TransformerLens")
try:
    import transformer_lens as lens  # Some python problem causes this to throw on the first import
except:
    import transformer_lens as lens
from general_utils import set_deterministic
from object_counting_utils import OBJECT_TYPES


class NullSafteyChecker(StableDiffusionSafetyChecker):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, clip_input, images):
        has_nsfw_concepts = [False for _ in range(len(images))]
        return images, has_nsfw_concepts

    def forward_onnx(self, clip_input: torch.FloatTensor, images: torch.FloatTensor):
        has_nsfw_concepts = [False for _ in range(len(images))]
        return images, has_nsfw_concepts


def generate(
    pipeline,
    prompts: List[str],
    seed: int,
    output_dir: str = "./",
    save_image: bool = True,
):
    """
    Generate images for a list of prompts, optionally filtering the results.

    Args:
        pipeline: The Text2Image pipeline to use for generation.
        prompts: A list of prompts to generate images for.
        seed: The seed to use for generation.
        output_dir: The directory to save the generated images.
        filter_func: A function to filter the generated images.
        save_image: Whether to save the generated images.
    """
    set_deterministic(seed)

    dataloader = torch.utils.data.DataLoader(prompts[::-1], batch_size=4, shuffle=False)
    for prompt_batch in dataloader:
        print(f"Generating {prompt_batch}")
        paths = [
            os.path.join(output_dir, f"{prompt}_seed_{seed}.png")
            for prompt in prompt_batch
        ]
        images = pipeline(prompt_batch, num_inference_steps=28)["images"]

        images_and_paths = list(zip(images, paths))

        if save_image:
            print("Saving batch")
            for image, image_path in images_and_paths:
                image_path_with_seed = image_path.replace(".png", f"_seed_{seed}.png")
                image.save(image_path_with_seed)

    return images_and_paths


def create_mixed_counting_prompts(prompt_template_idx):
    seq_lengths = list(range(3, 16))

    prompt_template = [
        f"An image of {{seq}}.",
    ][prompt_template_idx]

    prompts = []
    for seq_len in seq_lengths:
        for unique_obj_count in [1, 2, 3, 4]:
            for prompt_idx in range(10):
                seq_objects = random.sample(OBJECT_TYPES, unique_obj_count)

                # Split obj_count_in_seq to different_obj_count parts, summing up to obj_count_in_seq
                obj_counts = np.random.multinomial(
                    seq_len, [1 / unique_obj_count] * unique_obj_count
                )

                # Remove 0 values
                obj_counts = obj_counts[obj_counts != 0].tolist()

                seq = ", ".join(
                    f"{obj_counts[idx]} {seq_objects[idx]}"
                    for idx in range(len(obj_counts))
                )
                prompts.append(prompt_template.format(seq=seq))

    return prompts


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 43)))
    parser.add_argument("--prompt_template_idx", required=True, type=int, default=0)

    return parser.parse_args()


def generate_multi_object_image(
    t2i_pipe, objects, image_size=512, background_color=(255, 255, 255)
):
    """
    Generate a grid of images with specified objects on a white background.

    Args:
    - t2i_pipe: Stable Diffusion 3 Pipeline
    - objects: List of object names to generate
    - image_size: Size of each individual image
    - background_color: RGB color for background (default is white)

    Returns:
    PIL Image with generated objects in a grid layout
    """
    from PIL import Image

    # Validate input
    if not 3 <= len(objects) <= 15:
        raise ValueError("Number of objects must be between 3 and 15")

    # Determine grid layout
    grid_size = int(np.ceil(np.sqrt(len(objects))))

    # Create blank white canvas
    canvas = Image.new(
        "RGB", (image_size * grid_size, image_size * grid_size), color=background_color
    )

    # Generate images for each object
    for idx, obj in enumerate(objects):
        # Calculate grid position
        row = idx // grid_size
        col = idx % grid_size

        # Generate image with prompt
        image = t2i_pipe(
            prompt=f"a {obj} on a white background, centered, full view",
            height=image_size,
            width=image_size,
            num_inference_steps=30,
        ).images[0]

        # Paste image onto canvas
        canvas.paste(image, (col * image_size, row * image_size))

    return canvas


def main():
    args = parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    sd_model_name = "stabilityai/stable-diffusion-3.5-large"

    # Load model
    if "stable" in sd_model_name and "3.5" in sd_model_name:
        t2i_pipe = StableDiffusion3Pipeline.from_pretrained(
            sd_model_name, torch_dtype=torch.bfloat16
        ).to(device)
    else:
        hf_token = os.environ.get("HF_TOKEN")
        t2i_pipe = StableDiffusionPipeline.from_pretrained(
            sd_model_name, use_auth_token=hf_token
        ).to(device)

    output_dir = f"./data/counting/images/{args.prompt_template_idx + 1}"
    os.makedirs(output_dir, exist_ok=True)

    prompts = create_mixed_counting_prompts(args.prompt_template_idx)

    # Generate images
    images = []
    for seed in args.seeds:
        images += generate(
            t2i_pipe,
            prompts,
            seed,
            output_dir=output_dir,
            save_image=True,
        )


if __name__ == "__main__":
    main()
