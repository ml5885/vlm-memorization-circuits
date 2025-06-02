import csv
import glob
import os
import random
import re
import sys
import transformers
from tqdm import tqdm
from typing import List, Optional
from PIL import Image
from vision_language_prompts import VLPrompt

sys.path.append("./third_party/TransformerLens")
import transformer_lens as lens
from general_utils import (
    get_content_key_for_prompt_dict,
    get_single_token_tokens,
    load_image_for_model,
    load_prompts_from_csv,
    set_deterministic,
)

# Objects that should be tokenized to a single token across models
OBJECT_TYPES = [
    "people",
    "girl",
    "boy",
    "apple",
    "banana",
    "orange",
    "dog",
    "cat",
    "cow",
    "bird",
    "horse",
    "truck",
    "boat",
    "car",
    "bike",
    "ball",
    "tree",
    "flower",
    "book",
    "chair",
    "cup",
    "fork",
    "knife",
    "toy",
    "key",
    "bag",
    "hat",
    "card",
    "coin",
]

COUNTING_SEQ_LEN = 7


def load_counting_vl_prompts_list(
    data_csv_path: str,
    images_dir: str = "./data/counting/images",
    model: Optional[lens.HookedVLTransformer] = None,
    processor: Optional[transformers.processing_utils.ProcessorMixin] = None,
    correct_preds_only: bool = True,
    measure_acc_across_limited_labels: bool = False,
    object_count: int = COUNTING_SEQ_LEN,
    image_size=(504, 504),
) -> List[VLPrompt]:
    """
    Load a list of Vision-Language prompts for the Vision-based counting task.

    Args:
        data_csv_path: The path to the CSV file containing the dataset.
        images_dir: The directory containing the images.
        model: The model to use for generating predictions. Must be supplied in case the CSV doesn't exist.
        processor: The processor to use for generating prompts. Must be supplied in case the CSV doesn't exist.
        correct_preds_only: Whether to include only correct predictions in the dataset.
        image_size: The resize dimensions for the images.
    """
    single_token_object_types = get_single_token_tokens(processor, OBJECT_TYPES)
    assert (
        len(single_token_object_types) > 15
    ), f"There are too few objects left in {single_token_object_types}."

    data_csv_dir = os.path.dirname(data_csv_path)
    if not os.path.exists(data_csv_path):
        assert (
            model is not None and processor is not None
        ), "Model and processor must be supplied if the CSV file doesn't exist."

        # Create the dataset file from scratch.
        # For each image that exists in the image directory, for each object appearing in it
        # we generate a counting prompt and evaluate it to predict the model's correctness.
        data_tuples = []
        content_key = get_content_key_for_prompt_dict(model.cfg.model_name)
        for file_path in tqdm(glob.glob(os.path.join(images_dir, "*.png"))):
            relative_path_from_csv = os.path.relpath(file_path, data_csv_dir)
            image = load_image_for_model(
                file_path, model.model_name, target_size=image_size
            )
            image_name = os.path.basename(file_path)
            counts_and_objects = get_counts_and_objects(image_name)
            objects_in_image = sum([int(count) for count, _ in counts_and_objects])

            # Use only images with the correct number of objects
            if objects_in_image != object_count:
                continue

            for obj_count, obj in counts_and_objects:
                # Verify the object is in the single_token_object_types list (which differs per model)
                if obj not in single_token_object_types:
                    continue

                prompt = processor.apply_chat_template(
                    [get_vl_counting_prompt(obj, content_key)],
                    add_generation_prompt=True,
                )
                gt_answer = str(obj_count)
                pred_logits = model(prompt, [image])[:, -1]

                if measure_acc_across_limited_labels:
                    limited_labels = get_counting_limited_labels()
                    limited_tokens = (
                        model.to_tokens(limited_labels, prepend_bos=False)
                        .view(-1)
                        .to(pred_logits.device)
                    )
                    preds_ll_logits = pred_logits[:, limited_tokens].argmax(-1)
                    pred_answer = limited_labels[preds_ll_logits][0]
                else:
                    pred_answer = model.to_str_tokens(pred_logits.argmax(dim=-1))[0]

                data_tuples.append(
                    (relative_path_from_csv, prompt, gt_answer, pred_answer)
                )

        with open(data_csv_path, "w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                ["relative_image_path", "prompt", "gt_answer", "pred_answer"]
            )
            writer.writerows(data_tuples)

    vl_prompts = load_prompts_from_csv(
        data_csv_path, model.model_name, image_size, correct_preds_only
    )
    return vl_prompts


def load_counting_l_prompts_list(
    data_csv_path: str,
    model: Optional[lens.HookedVLTransformer] = None,
    processor: Optional[transformers.processing_utils.ProcessorMixin] = None,
    random_seed: int = 42,
    sequence_length: int = COUNTING_SEQ_LEN,
    correct_preds_only: bool = True,
    measure_acc_across_limited_labels: bool = False,
) -> List[VLPrompt]:
    """
    Load a list of Language-only prompts for the text-based counting task.

    Args:
        data_csv_path: The path to the CSV file containing the dataset.
        model: The model to use for generating predictions. Must be supplied in case the CSV doesn't exist.
        processor: The processor to use for generating prompts. Must be supplied in case the CSV doesn't exist.
        correct_preds_only: Whether to include only correct predictions in the dataset.
    """
    if not os.path.exists(data_csv_path):
        assert (
            model is not None and processor is not None
        ), "Model and processor must be supplied if the CSV file doesn't exist."

        # Create the dataset from scratch.
        set_deterministic(random_seed)
        print("Generating prompts")
        l_counting_prompts = generate_l_counting_prompts(
            processor,
            # The sequench length aross all language prompts must be of the same size
            # for position-aware analysis (such as mean cache calculation, patching, etc.)
            # The number 7 is picked to be high enough to allow high unique_obj_counts and
            # low enough to allow OK-level performance in the visual prompting setting.
            seq_len_range=(sequence_length, sequence_length),
            unique_obj_count_range=(1, 4),
            prompts_per_setting=500,
        )
        print(f"Generated {len(l_counting_prompts)} prompts")
        l_counting_prompts = list(set(l_counting_prompts))
        print(f"Unique prompts: {len(l_counting_prompts)}")

        # Save the generated dataset to a csv
        data_tuples = []
        print("Evaluating generated prompts")
        for l_prompt in tqdm(l_counting_prompts):
            pred_logits = model(l_prompt.prompt, [])[:, -1]
            if measure_acc_across_limited_labels:
                limited_labels = get_counting_limited_labels()
                limited_tokens = (
                    model.to_tokens(limited_labels, prepend_bos=False)
                    .view(-1)
                    .to(pred_logits.device)
                )
                preds_ll_logits = pred_logits[:, limited_tokens].argmax(-1)
                pred_answer = limited_labels[preds_ll_logits][0]
            else:
                pred_answer = model.to_str_tokens(pred_logits.argmax(dim=-1))[0]
            data_tuples.append((l_prompt.prompt, l_prompt.answer, pred_answer))

        with open(data_csv_path, "w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["prompt", "gt_answer", "pred_answer"])
            writer.writerows(data_tuples)

    return load_prompts_from_csv(
        data_csv_path, model.model_name, correct_preds_only=correct_preds_only
    )


def load_counting_parallel_l_prompts(
    vl_prompts: List[VLPrompt],
    processor: transformers.processing_utils.ProcessorMixin,
    seed: int = 42,
):
    """
    Loads a list of perfectly-aligned Language-only prompts from a given list of Vision-Language prompts.
    A perfectly aligned counting prompt is a prompt that has the same content as a Vision-Language counting prompt.
    For example, an image containing 3 dogs and 2 cats can get "translated" to a texutal sequence
    like "dog dog cat dog cat".
    """
    set_deterministic(seed)
    parallel_l_prompts = []
    for vl_prompt in vl_prompts:
        # Extract the counted object from the VL prompt
        obj_in_prompt = re.findall('"(.*)"', vl_prompt.prompt)[
            0
        ]  # Object appears in quotes

        # Create a textual sequence of the objects in the image
        image_filename = os.path.basename(vl_prompt.metadata["image_path"])
        seq_counts_and_objects = get_counts_and_objects(image_filename)
        parallel_textual_seq = sum(
            [[co[1]] * int(co[0]) for co in seq_counts_and_objects], []
        )
        random.shuffle(parallel_textual_seq)

        # Process and create the textual prompt
        content_key = get_content_key_for_prompt_dict(processor.__class__.__name__)
        prompt = processor.apply_chat_template(
            [
                get_l_counting_prompt(
                    obj_in_prompt, " ".join(parallel_textual_seq), content_key
                )
            ],
            add_generation_prompt=True,
        )
        parallel_l_prompts.append(VLPrompt(prompt, [], str(vl_prompt.answer)))

    return parallel_l_prompts


def generate_l_counting_prompts(
    processor,
    seq_len_range=(3, 15),
    unique_obj_count_range=(1, 4),
    prompts_per_setting=100,
):
    l_counting_prompts = []
    single_token_object_types = get_single_token_tokens(processor, OBJECT_TYPES)
    assert (
        len(single_token_object_types) > 15
    ), f"There are too few objects left in {single_token_object_types}."

    for seq_len in range(seq_len_range[0], seq_len_range[1] + 1):
        for unique_obj_count in range(
            unique_obj_count_range[0], unique_obj_count_range[1] + 1
        ):
            for _ in range(prompts_per_setting):
                obj = random.choice(single_token_object_types)
                if unique_obj_count == 1:
                    obj_count = seq_len
                else:
                    obj_count = random.randint(
                        1, seq_len - 1
                    )  # "Oversample" the target obj

                other_objects = random.sample(
                    list(set(single_token_object_types) - {obj}), unique_obj_count - 1
                )
                other_object_counts = random.choices(
                    range(len(other_objects)), k=seq_len - obj_count
                )

                seq = [obj] * obj_count + [
                    other_objects[i] for i in other_object_counts
                ]
                random.shuffle(seq)
                content_key = get_content_key_for_prompt_dict(
                    processor.__class__.__name__
                )
                prompt = processor.apply_chat_template(
                    [get_l_counting_prompt(obj, " ".join(seq), content_key)],
                    add_generation_prompt=True,
                )
                l_counting_prompts.append(VLPrompt(prompt, [], str(obj_count)))
    return l_counting_prompts


def get_counts_and_objects(image_name):
    """
    Given a counting-task image name such as "3_apple_2_boy_1_coin.png",
    return a list of tuples of the form (count, object) for each object in the image.
    """
    return re.findall(r"(\d+)_([a-z]+)(?:_|\.png$)", image_name)


def get_vl_counting_prompt(obj: str, content_key: str = "text") -> dict:
    """
    Generates the input to `processor.apply_chat_template` in order to generate the final text input for the Vision-Language forward pass.
    """
    return {
        "role": "user",
        "content": [
            {"type": "image"},
            {
                "type": "text",
                content_key: f'How many "{obj}" are in the image? Answer in a single number. ',
            },
        ],
    }


def get_l_counting_prompt(obj: str, seq: str, content_key: str = "text") -> dict:
    """
    Generates the input to `processor.apply_chat_template` in order to generate the final
    text input for the Language only forward pass.

    Args:
        obj: The object type to count in the sequence.
        seq: The sequence of objects to count.
    """
    return {
        "role": "user",
        "content": [
            {
                "type": "text",
                content_key: f'Sequence: {seq}. How many "{obj}" are in the sequence? Answer in a single number. ',
            },
        ],
    }


def get_counting_limited_labels():
    """
    Get the limited labels for the counting task.
    """
    return [str(i) for i in range(10)]
