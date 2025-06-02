import gc
import os
import torch
import random
import json
import shutil
from tqdm import tqdm
import numpy as np
from PIL import Image

from datasets import load_dataset
from transformers import MllamaForConditionalGeneration, AutoProcessor

from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionSafetyChecker
from diffusers import StableDiffusion3Pipeline

from utils import get_color_tokens

HF_TOKEN = ""


class SafteyChecker(StableDiffusionSafetyChecker):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, clip_input, images):
        has_nsfw_concepts = [False for _ in range(len(images))]
        return images, has_nsfw_concepts

    def forward_onnx(self, clip_input: torch.FloatTensor, images: torch.FloatTensor):
        has_nsfw_concepts = [False for _ in range(len(images))]
        return images, has_nsfw_concepts


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def generate_with_seed(sd_pipeline, prompts, seed, output_path="./", image_path=None, image_params="", save_image=True):
    set_seed(seed)
    outputs = []
    for prompt in prompts:
        print(prompt)
        # image = sd_pipeline(prompt, callback=save_diffusion_steps_images)['images'][0]
        image = sd_pipeline(prompt)['images'][0]

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        if image_params != "":
            image_params = "_" + image_params

        image_name = image_path or f"{output_path}/{prompt}.jpeg"
        if save_image:
            print(image_name)
            image.save(image_name)
        outputs.append((image, image_name))

    if len(outputs) == 1:
        return outputs[0]
    return outputs


def main():
    seed = 0
    output_path = "./mcqa/"
    verified_output_path = "./mcqa/verified_images/"

    pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large-turbo",
                                                    torch_dtype=torch.bfloat16,
                                                    device_map="balanced",
                                                    )

    mcqa = load_dataset("mech-interp-bench/copycolors_mcqa", "2_answer_choices", split="train")
    generation_prompt_template = "an image of {color} {objects} on a white background"

    color_tokens = dict()

    images_to_verify = []
    for sample in tqdm(mcqa):
        object1 = sample["elements"][0]
        colors = sample["choices"]["text"]

        # if not validate_single_token(object1, object2, colors):
        #     continue

        answerkey = sample["answerKey"]
        correct_color = colors[answerkey]
        generation_prompt = generation_prompt_template.format(color=correct_color, objects=object1)

        image_path = f"{output_path}/{generation_prompt}.jpeg"
        if not os.path.exists(image_path):
            generate_with_seed(
                pipe, [generation_prompt], seed,
                # output_path=output_path, save_image=True
                image_path=image_path, save_image=True
            )

        images_to_verify.append([generation_prompt])

    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="balanced",
    )
    processor = AutoProcessor.from_pretrained(model_id)

    verification_prompt_template = "<|image|>. What color are the {objects}? Answer in a single token:"
    verified_data = []

    for index, (object1, correct_color, image_path) in enumerate(images_to_verify):
        print("*" * 100)
        image = Image.open(f"./{output_path}/{image_path}.jpeg")
        prompt = verification_prompt_template.format(objects=object1)
        color_tokens = get_color_tokens([correct_color], color_tokens, processor)

        inputs = processor(
            image,
            prompt,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(model.device)

        outputs = model(
            **inputs,
        )
        logits = outputs.logits[:, -1]
        scores = torch.softmax(logits, dim=-1).detach().cpu()
        scores = scores.squeeze()
        top_token = torch.argmax(scores).item()
        verified = top_token in color_tokens[correct_color]

        if verified:
            verified_data.append(image_path)
            shutil.copyfile(f"./{output_path}/{image_path}.jpeg", f"./{verified_output_path}/{image_path}.jpeg")

        for k, v in inputs.items():
            inputs[k] = v.cpu()

    with open(f"./verified_data.json", "w") as f:
        json.dump(verified_data, f)
        print(f"saved {len(verified_data)} verified data")


if __name__ == "__main__":
    main()
