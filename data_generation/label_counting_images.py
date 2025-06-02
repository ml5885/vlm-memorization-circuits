import sys
import os
import re
import glob
import time
import base64
import openai
import argparse
from tqdm import tqdm

sys.path.append("./third_party/TransformerLens")
try:
    import transformer_lens as lens
except ImportError:
    import transformer_lens as lens
from object_counting_utils import get_vl_counting_prompt, get_counts_and_objects


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="gpt-4o")
    args = parser.parse_args()
    return args


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def filter_image_list(image_list):
    """
    Implement various filters to prioritize labelng specific images over others.
    """
    new_list = []
    for img_path in image_list:
        img_name = os.path.basename(img_path)
        counts_and_objs = get_counts_and_objects(img_name)
        unique_obj_count = len(counts_and_objs)
        total_obj_count = sum([int(count) for count, _ in counts_and_objs])
        # if total_obj_count >= 6:
        #   new_list.append(img_path)
        if total_obj_count >= 10 and unique_obj_count >= 2:
            new_list.append(img_path)

    return new_list


def make_api_request(image_prompts, image_path):
    request = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt} for prompt in image_prompts]
            + [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encode_image(image_path)}",
                        "detail": "high",
                    },
                }
            ],
        }
    ]
    return request


def process_response(response_text):
    response_text = response_text.lower()
    split_responses = re.split("\ |;|,|\n", response_text)
    string_numbers = [
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
        "twenty",
    ]

    response_counts = []
    for i, word in enumerate(split_responses):
        if word in string_numbers:
            response_counts.append(string_numbers.index(word))
        elif word.isdigit():
            response_counts.append(int(word))

    return response_counts


def main():
    """
    Use OpenAI API to count the number of objects in an image and generate a new file name.
    This is used as a automatic verification that the amount of objects in each image generated
    by flux/SD/any other t2i model is correct.

    This isn't 100% correct. However, this is correct most of the times, and manually going over
    the results of this script and verifying them is much faster than manually labeling each image.
    """
    args = parse_args()

    failed_dir = args.output_dir + "_failed"
    os.makedirs(failed_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Requires pre-definition of OPENAI_API_KEY environment variable
    client = openai.OpenAI()

    # For each image in the input directory, use ChatGPT to count the number of objects
    # in the image and use the labels to generate a new file name with the correct counts.
    image_list = glob.glob(os.path.join(args.input_dir, "*.png"))
    image_list = filter_image_list(image_list)

    for idx, image_path in enumerate(tqdm(image_list)):
        # Temporary limit to process only 100 images
        if idx > 50:
            break

        # To avoid rate limiting, sleep for a second between each request
        time.sleep(1)

        # Image names look like 'An image of 9 orange, 3 book._seed_46_seed_46.png',
        # extract the count of each object and the object name
        counts_and_objects = get_counts_and_objects(image_path)

        # Generate a prompt for each object in the image
        prompts = []
        for count, object_name in counts_and_objects:
            prompts.append(get_vl_counting_prompt(object_name)["content"][-1]["text"])

        response = client.chat.completions.create(
            model=args.model,
            messages=make_api_request(prompts, image_path),
            max_tokens=32,
        )

        response_text = response.choices[0].message.content
        response_counts = process_response(response_text)

        # In case of unexpected number of outputs (which doesn't match the amount of prompts),
        # move the image to a separate directory for manual labeling and save the response for
        # debugging purposes
        if len(response_counts) != len(counts_and_objects):
            os.rename(
                image_path, os.path.join(failed_dir, os.path.basename(image_path))
            )
            with open(
                os.path.join(failed_dir, os.path.basename(image_path) + ".txt"), "w"
            ) as f:
                f.write(response.to_json())

                continue

        # Convert the name to something that includes the true labels,
        # like "9_orange_3_book.png"

        new_image_name = "_".join(
            f"{count}_{object_name}"
            for (count, object_name) in zip(
                response_counts,
                [object_name for _, object_name in counts_and_objects],
            )
            if count != 0
        )

        # Copy the image to the output directory with the new name
        new_image_path = os.path.join(args.output_dir, new_image_name + ".png")

        if os.path.exists(new_image_path):
            new_image_path = os.path.join(
                os.path.dirname(new_image_path), f"{new_image_name}_x{idx}.png"
            )

        os.rename(image_path, new_image_path)


if __name__ == "__main__":
    main()
