import csv
import glob
import os
import random
import re
import sys
import torch
from tqdm import tqdm
import transformers

sys.path.append("./third_party/TransformerLens")
from general_utils import (
    get_content_key_for_prompt_dict,
    load_image_for_model,
    load_prompts_from_csv,
    safe_eval,
    set_deterministic,
)
import transformer_lens as lens

from data_generation.generate_arithmetic_data import get_operand_range
from typing import List, Optional
from vision_language_prompts import VLPrompt

OPSTR_TO_OP = {"plus": "+", "minus": "-", "mult": "*", "divide": "/"}


def load_arithmetic_vl_prompts_list(
    data_csv_path: str,
    images_dir: str = "./data/arithmetic/images",
    model: Optional[lens.HookedVLTransformer] = None,
    processor: Optional[transformers.processing_utils.ProcessorMixin] = None,
    correct_preds_only: bool = True,
    measure_acc_across_limited_labels: bool = False,
    target_size=(338, 75),
) -> List[VLPrompt]:
    """
    Load a list of Vision-Language prompts for the Vision-based counting task.

    Args:
        data_csv_path: The path to the CSV file containing the dataset.
        images_dir: The directory containing the images.
        model: The model to use for generating predictions. Must be supplied in case the CSV doesn't exist.
        processor: The processor to use for generating prompts. Must be supplied in case the CSV doesn't exist.
        correct_preds_only: Whether to include only correct predictions in the dataset.
        target_size: The resize dimensions for the images.
    """
    data_csv_dir = os.path.dirname(data_csv_path)
    if not os.path.exists(data_csv_path):
        assert (
            model is not None and processor is not None
        ), "Model and processor must be supplied if the CSV file doesn't exist."

        # Create the dataset file from scratch.
        data_tuples = []
        content_key = get_content_key_for_prompt_dict(model.cfg.model_name)

        for file_path in tqdm(glob.glob(os.path.join(images_dir, "*.png"))):
            relative_path_from_csv = os.path.relpath(file_path, data_csv_dir)
            image = load_image_for_model(
                file_path, model.model_name, target_size=target_size
            )
            image_name = os.path.basename(file_path)
            op1, operator, op2 = re.match(
                r"(\d+)([a-zA-Z]+)(\d+).png", image_name
            ).groups()
            operator = OPSTR_TO_OP[operator]
            full_prompt = processor.apply_chat_template(
                [get_vl_arithmetic_prompt(content_key)],
                add_generation_prompt=True,
            )
            gt_answer = get_first_answer_token(
                str(safe_eval(f"{op1}{operator}{op2}")), model
            )

            pred_logits = model(full_prompt, [image])[:, -1]
            if measure_acc_across_limited_labels:
                limited_labels = get_arithmetic_limited_labels()
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
                (relative_path_from_csv, full_prompt, gt_answer, pred_answer)
            )

        with open(data_csv_path, "w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                ["relative_image_path", "prompt", "gt_answer", "pred_answer"]
            )
            writer.writerows(data_tuples)

    vl_prompts = load_prompts_from_csv(
        data_csv_path, model.model_name, target_size, correct_preds_only
    )
    return vl_prompts


def load_arithmetic_l_prompts_list(
    data_csv_path: str,
    model: Optional[lens.HookedVLTransformer] = None,
    processor: Optional[transformers.processing_utils.ProcessorMixin] = None,
    random_seed: int = 42,
    correct_preds_only: bool = True,
    measure_acc_across_limited_labels: bool = False,
) -> List[VLPrompt]:
    """
    Load a list of Language-only prompts for the text-based arithmetic task.

    Args:
        data_csv_path: The path to the CSV file containing the dataset.
        model: The model to use for generating predictions. Must be supplied in case the CSV doesn't exist.
        processor: The processor to use for generating prompts. Must be supplied in case the CSV doesn't exist.
        correct_preds_only: Whether to include only correct predictions in the dataset.
        measure_acc_across_limited_labels: Whether to measure accuracy across numerical labels only.
    """
    if not os.path.exists(data_csv_path):
        assert (
            model is not None and processor is not None
        ), "Model and processor must be supplied if the CSV file doesn't exist."

        # Create the dataset from scratch.
        set_deterministic(random_seed)
        print("Generating prompts")

        l_arithmetic_prompts = []
        for i in range(1000):
            op1 = random.randint(10**1, 10**2 - 1)
            operator = random.choice(list(OPSTR_TO_OP.values()))
            op2_range = list(
                get_operand_range(
                    operator, op1, operand_min=10**1, operand_max=10**2 - 1
                )
            )
            if len(op2_range) == 0:
                continue
            op2 = random.choice(op2_range)
            prompt = f"{op1}{operator}{op2}"
            full_prompt = processor.apply_chat_template(
                [
                    get_l_arithmetic_prompt(
                        prompt, get_content_key_for_prompt_dict(model.model_name)
                    )
                ],
                add_generation_prompt=True,
            )
            gt_answer = get_first_answer_token(str(safe_eval(prompt)), model)
            l_arithmetic_prompts.append((full_prompt, gt_answer))

        print(f"Generated {len(l_arithmetic_prompts)} prompts")
        l_arithmetic_prompts = list(set(l_arithmetic_prompts))
        print(f"Unique prompts: {len(l_arithmetic_prompts)}")

        # Save the generated dataset to a csv
        data_tuples = []
        print("Evaluating generated prompts")
        for prompt, gt_answer in tqdm(l_arithmetic_prompts):
            pred_logits = model(prompt, [])[:, -1]
            if measure_acc_across_limited_labels:
                limited_labels = get_arithmetic_limited_labels()
                limited_tokens = (
                    model.to_tokens(limited_labels, prepend_bos=False)
                    .view(-1)
                    .to(pred_logits.device)
                )
                preds_ll_logits = pred_logits[:, limited_tokens].argmax(-1)
                pred_answer = limited_labels[preds_ll_logits][0]
            else:
                pred_answer = model.to_str_tokens(pred_logits.argmax(dim=-1))[0]
            data_tuples.append((prompt, gt_answer, pred_answer))

        with open(data_csv_path, "w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["prompt", "gt_answer", "pred_answer"])
            writer.writerows(data_tuples)

    return load_prompts_from_csv(
        data_csv_path, model.model_name, correct_preds_only=correct_preds_only
    )


def load_arithmetic_parallel_l_prompts(
    vl_prompts: List[VLPrompt],
    processor: transformers.processing_utils.ProcessorMixin,
):
    """
    Loads a list of perfectly-aligned Language-only prompts from a given list of Vision-Language prompts.
    A perfectly aligned counting prompt is a prompt that has the same content as a Vision-Language counting prompt.
    For example, an image containing 3+4= will get "translated" to the textual prompt "3+4".
    """
    parallel_l_prompts = []
    for vl_prompt in vl_prompts:
        # Create a textual sequence of the objects in the image
        image_filename = os.path.basename(vl_prompt.metadata["image_path"])
        op1, operator, op2 = re.match(
            r"(\d+)([a-zA-Z]+)(\d+).png", image_filename
        ).groups()
        operator = OPSTR_TO_OP[operator]
        full_prompt = processor.apply_chat_template(
            [
                get_l_arithmetic_prompt(
                    f"{op1}{operator}{op2}",
                    get_content_key_for_prompt_dict(processor.__class__.__name__),
                )
            ],
            add_generation_prompt=True,
        )
        parallel_l_prompts.append(VLPrompt(full_prompt, [], str(vl_prompt.answer)))

    return parallel_l_prompts


def get_vl_arithmetic_prompt(content_key: str = "text") -> dict:
    """
    Generates the input to `processor.apply_chat_template` in order to generate the final text input for the Vision-Language forward pass.
    """
    return {
        "role": "user",
        "content": [
            {"type": "image"},
            {
                "type": "text",
                content_key: "What is the result of the given arithmetic calculation? Answer in a single number.",
            },
        ],
    }


def get_l_arithmetic_prompt(
    arithmetic_question: str, content_key: str = "text"
) -> dict:
    """
    Generates the input to `processor.apply_chat_template` in order to generate the final
    text input for the Language only forward pass.

    Args:
        arithmetic_question: The arithmetic calculation to perform. e.g. 34-12.
    """
    return {
        "role": "user",
        "content": [
            {
                "type": "text",
                content_key: f"Question: {arithmetic_question}. What is the result of the given arithmetic calculation? Answer in a single number.",
            },
        ],
    }


def get_arithmetic_limited_labels():
    """
    Get the limited labels for the arithmetic task.
    """
    return [str(i) for i in range(10)]


def get_first_answer_token(answer, model):
    """
    Get the first token of the answer (usually will be a single digit,
    but might just be an entire number in case the model doesn't tokenize per-digit).
    """
    return model.to_string(model.to_tokens([answer], prepend_bos=False).view(-1)[0])
