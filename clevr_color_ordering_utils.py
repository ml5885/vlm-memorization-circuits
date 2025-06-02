import csv
import json
import os
import random
import re
import sys
from tqdm import tqdm
import transformers

sys.path.append("./third_party/TransformerLens")
from general_utils import (
    get_content_key_for_prompt_dict,
    get_single_token_tokens,
    load_image_for_model,
    load_prompts_from_csv,
    set_deterministic,
)
import transformer_lens as lens
from typing import List, Optional
from PIL import Image
from vision_language_prompts import VLPrompt

CLEVR_HOME = "./data/color_ordering/CLEVR"
CLEVR_COLORS = [
    "purple",
    "gray",
    "cyan",
    "blue",
    "brown",
    "yellow",
    "green",
    "red",
]
SIDES = ["left", "right"]
ORDER_INDICES = [
    "first",
    "second",
    "third",
    "fourth",
    "fifth",
    "sixth",
    "seventh",
    "eighth",
    "ninth",
]
OBJECTS_PER_SCENE_TEXTUAL = "four"
OBJECTS_PER_SCENE = 4


def get_color_ordering_limited_labels(processor):
    return get_single_token_tokens(processor, CLEVR_COLORS)


def load_color_ordering_vl_prompts_list(
    data_csv_path: str,
    model: Optional[lens.HookedVLTransformer] = None,
    processor: Optional[transformers.processing_utils.ProcessorMixin] = None,
    correct_preds_only: bool = True,
    measure_acc_across_limited_labels: bool = True,  # NOTE DIFFERENCE FROM OTHER TASKS, SHOULD CONVERT ALL TASKS TO DEFAULT TRUE
    prompts_per_scene: int = 1,
    image_size=(504, 504),
    random_seed: int = 42,
) -> List[VLPrompt]:
    """
    Load a list of Vision-Language prompts for the Vision-based object ordering task using CLEVR images.

    Args:
        data_csv_path: The path to the CSV file containing the dataset.
        images_dir: The directory containing the images.
        model: The model to use for generating predictions. Must be supplied in case the CSV doesn't exist.
        processor: The processor to use for generating prompts. Must be supplied in case the CSV doesn't exist.
        correct_preds_only: Whether to include only correct predictions in the dataset.
        target_size: The resize dimensions for the images.
    """
    single_token_colors = get_color_ordering_limited_labels(model.processor)

    assert (
        len(single_token_colors) >= 5
    ), f"There are too few colors left in {single_token_colors}."

    data_csv_dir = os.path.dirname(data_csv_path)
    if not os.path.exists(data_csv_path):
        # Create the dataset file from scratch.
        set_deterministic(random_seed)
        assert (
            model is not None and processor is not None
        ), "Model and processor must be supplied if the CSV file doesn't exist."

        data_tuples = []
        content_key = get_content_key_for_prompt_dict(model.cfg.model_name)

        scenes = get_clevr_scenes(
            CLEVR_HOME,
            OBJECTS_PER_SCENE,
            all_different_color=False,
            limit_colors=single_token_colors,
        )

        for scene in tqdm(scenes):
            image_path = os.path.join(CLEVR_HOME, "images", scene["image_filename"])
            relative_path_from_csv = os.path.relpath(image_path, data_csv_dir)
            image = load_image_for_model(image_path, model.model_name, image_size)

            for _ in range(prompts_per_scene):
                side_idx = random.choice([0, 1])
                side = SIDES[side_idx]

                order_idx = random.choice(list(range(1, len(scene["objects"]) + 1)))
                order = ORDER_INDICES[order_idx - 1]

                objects_sorted_by_x_coord = sorted(
                    scene["objects"],
                    key=lambda x: x["pixel_coords"][0],
                    reverse=(side == "right"),
                )
                gt_answer = objects_sorted_by_x_coord[order_idx - 1]["color"]
                prompt = processor.apply_chat_template(
                    [get_vl_color_ordering_prompt(order, side, content_key)],
                    add_generation_prompt=True,
                )

                pred_logits = model(prompt, [image])[:, -1]
                if measure_acc_across_limited_labels:
                    limited_labels = single_token_colors
                    limited_tokens = (
                        model.to_tokens(limited_labels, prepend_bos=False)
                        .view(-1)
                        .to(pred_logits.device)
                    )
                    preds_ll_logits = pred_logits[:, limited_tokens].argmax(-1)
                    pred_answer = limited_labels[preds_ll_logits]
                else:
                    pred_answer = model.to_str_tokens(pred_logits.argmax(dim=-1))[0]
                data_tuples.append(
                    (relative_path_from_csv, prompt, gt_answer, pred_answer)
                )

            data_tuples = list(set(data_tuples))

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


def load_color_ordering_l_prompts_list(
    data_csv_path: str,
    model: Optional[lens.HookedVLTransformer] = None,
    processor: Optional[transformers.processing_utils.ProcessorMixin] = None,
    correct_preds_only: bool = True,
    measure_acc_across_limited_labels: bool = False,
    prompts_per_scene: int = 1,
    random_seed: int = 42,
) -> List[VLPrompt]:
    """
    Load a list of Language-only prompts for the text-based color ordering task.

    Args:
        data_csv_path: The path to the CSV file containing the dataset.
        model: The model to use for generating predictions. Must be supplied in case the CSV doesn't exist.
        processor: The processor to use for generating prompts. Must be supplied in case the CSV doesn't exist.
        correct_preds_only: Whether to include only correct predictions in the dataset.
        measure_acc_across_limited_labels: Whether to measure accuracy across a limited set of labels (The allowed CLEVR colors).
    """
    single_token_colors = get_color_ordering_limited_labels(model.processor)
    assert (
        len(single_token_colors) >= 5
    ), f"There are too few colors left in {single_token_colors}."

    if not os.path.exists(data_csv_path):
        # Create the dataset file from scratch.
        set_deterministic(random_seed)

        assert (
            model is not None and processor is not None
        ), "Model and processor must be supplied if the CSV file doesn't exist."
        data_tuples = []
        content_key = get_content_key_for_prompt_dict(model.cfg.model_name)

        scenes = get_clevr_scenes(
            CLEVR_HOME,
            OBJECTS_PER_SCENE,
            all_different_color=False,  # NOTE
            limit_colors=single_token_colors,
        )

        for scene in tqdm(scenes):
            # Create textual scene decrpition
            scene_objects = sorted(
                scene["objects"], key=lambda x: x["pixel_coords"][0]
            )  # sorted by x value
            scene_desc = get_scene_description(scene_objects)

            for _ in range(prompts_per_scene):
                side_idx = random.choice([0, 1])
                side = SIDES[side_idx]

                order_idx = random.choice(list(range(1, len(scene["objects"]) + 1)))
                order = ORDER_INDICES[order_idx - 1]

                objects_sorted_by_side = sorted(
                    scene_objects,
                    key=lambda x: x["pixel_coords"][0],
                    reverse=(side == "right"),
                )
                gt_answer = objects_sorted_by_side[order_idx - 1]["color"]

                prompt = processor.apply_chat_template(
                    [get_l_color_ordering_prompt(order, side, scene_desc, content_key)],
                    add_generation_prompt=True,
                )

                pred_logits = model(prompt, [])[:, -1]
                if measure_acc_across_limited_labels:
                    limited_labels = single_token_colors
                    limited_tokens = (
                        model.to_tokens(limited_labels, prepend_bos=False)
                        .view(-1)
                        .to(pred_logits.device)
                    )
                    preds_ll_logits = pred_logits[:, limited_tokens].argmax(-1)
                    pred_answer = limited_labels[preds_ll_logits]
                else:
                    pred_answer = model.to_str_tokens(pred_logits.argmax(dim=-1))[0]
                data_tuples.append((prompt, gt_answer, pred_answer))

            data_tuples = list(set(data_tuples))

        with open(data_csv_path, "w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["prompt", "gt_answer", "pred_answer"])
            writer.writerows(data_tuples)

    return load_prompts_from_csv(
        data_csv_path, model.model_name, correct_preds_only=correct_preds_only
    )


def load_color_ordering_parallel_l_prompts(
    vl_prompts: List[VLPrompt],
    processor: transformers.processing_utils.ProcessorMixin,
):
    """
    Loads a list of perfectly-aligned Language-only prompts from a given list of Vision-Language prompts.
    """
    parallel_l_prompts = []
    single_token_colors = get_color_ordering_limited_labels(processor)
    scenes = get_clevr_scenes(
        CLEVR_HOME,
        OBJECTS_PER_SCENE,
        all_different_color=False,
        limit_colors=single_token_colors,
    )

    for vl_prompt in vl_prompts:
        scene = [
            s
            for s in scenes
            if s["image_filename"] == os.path.basename(vl_prompt.metadata["image_path"])
        ][0]
        scene_objects = sorted(scene["objects"], key=lambda x: x["pixel_coords"][0])
        scene_desc = get_scene_description(scene_objects)
        order, side = re.search(
            r".*What is the color of the (\w+) object from the (\w+)\? Answer in a single word\..*",
            vl_prompt.prompt,
        ).groups()

        # Process and create the textual prompt
        content_key = get_content_key_for_prompt_dict(processor.__class__.__name__)
        prompt = processor.apply_chat_template(
            [get_l_color_ordering_prompt(order, side, scene_desc, content_key)],
            add_generation_prompt=True,
        )
        parallel_l_prompts.append(VLPrompt(prompt, [], str(vl_prompt.answer)))

    return parallel_l_prompts


def assert_colors_sides_and_orders_are_tokenized_well(model):
    """
    Check that all investigated objects are tokenized to a single token (so they are all interchangeable in interventions).

    Args:
        model: The model to use for tokenization.
    """
    side_len_in_tokens = model.to_tokens(SIDES[0], prepend_bos=False).numel()
    for side in SIDES:
        len_in_tokens = model.to_tokens(side, prepend_bos=False).numel()
        assert (
            len_in_tokens == side_len_in_tokens
        ), f"{side} is tokenized to {len_in_tokens} tokens."

    order_len_in_tokens = model.to_tokens(ORDER_INDICES[0], prepend_bos=False).numel()
    for order in ORDER_INDICES:
        len_in_tokens = model.to_tokens(order, prepend_bos=False).numel()
        assert (
            len_in_tokens == order_len_in_tokens
        ), f"{order} is tokenized to {len_in_tokens} tokens."

    for color in CLEVR_COLORS:
        len_in_tokens = model.to_tokens(color, prepend_bos=False).numel()
        assert len_in_tokens == 1, f"{color} is tokenized to {len_in_tokens} tokens."


def get_clevr_scenes(
    CLEVR_HOME, objects_per_image, all_different_color=True, limit_colors=None
):
    scenes_file = os.path.join(CLEVR_HOME, rf"./scenes/CLEVR_val_scenes.json")
    scenes = json.load(open(scenes_file, "rb"))["scenes"]
    scenes = [
        scenes[i]
        for i in range(len(scenes))
        if len(scenes[i]["objects"]) == objects_per_image
    ]
    if all_different_color:
        scenes_all_different_color = [
            scenes[i]
            for i in range(len(scenes))
            if len(set([obj["color"] for obj in scenes[i]["objects"]]))
            == len(scenes[i]["objects"])
        ]
        scenes = scenes_all_different_color

    if limit_colors is not None:
        scenes_limited_colors = [
            scene
            for scene in scenes
            if all([obj["color"] in limit_colors for obj in scene["objects"]])
        ]
        scenes = scenes_limited_colors

    return scenes


def get_scene_description(scene_objects):
    scene_desc = f"In a scene with {OBJECTS_PER_SCENE_TEXTUAL} objects arranged horizontally, there is "
    for i, obj in enumerate(scene_objects):
        scene_desc += f"a {obj['color']} object"
        if i < len(scene_objects) - 2:
            scene_desc += ", "
        elif i == len(scene_objects) - 2:
            scene_desc += " and "
    return scene_desc


def get_vl_color_ordering_prompt(
    order: str, side: str, content_key: str = "text"
) -> dict:
    """
    Generates the input to `processor.apply_chat_template` in order to generate the final text input for the Vision-Language forward pass.
    """
    return {
        "role": "user",
        "content": [
            {"type": "image"},
            {
                "type": "text",
                content_key: f"What is the color of the {order} object from the {side}? Answer in a single word.",
            },
        ],
    }


def get_l_color_ordering_prompt(
    order: str, side: str, scene_desc: str, content_key: str = "text"
) -> dict:
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
                content_key: f"{scene_desc}. What is the color of the {order} object from the {side}? Answer in a single word.",
            },
        ],
    }


#
# For the complex multi-hop task
#
def get_vl_complex_color_oredring_prompt(
    order: str, side: str, reference_obj_color: str, content_key: str = "text"
) -> dict:
    return {
        "role": "user",
        "content": [
            {"type": "image"},
            {
                "type": "text",
                content_key: f"What is the color of the object which is {order} {'object' if order == 'one' else 'objects'} {side} from the {reference_obj_color} object? Answer in a single word.",
            },
        ],
    }


def get_l_complex_color_ordering_prompt(
    order: str,
    side: str,
    scene_desc: str,
    reference_obj_color: str,
    content_key: str = "text",
):
    return {
        "role": "user",
        "content": [
            {
                "type": "text",
                content_key: f"{scene_desc}. What is the color of the object which is {order} {'object' if order == 'one' else 'objects'} {side} from the {reference_obj_color} object? Answer in a single word.",
            },
        ],
    }
