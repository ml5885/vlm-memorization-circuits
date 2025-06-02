import csv
import glob
import json
import os
import sys
import transformers
from tqdm import tqdm
from typing import List, Optional
from vision_language_prompts import VLPrompt

sys.path.append("./third_party/TransformerLens")
from general_utils import (
    get_content_key_for_prompt_dict,
    load_image_for_model,
    load_prompts_from_csv,
    set_deterministic,
    to_single_token,
    truncate_to_n_tokens,
)
import transformer_lens as lens


POSSIBLE_SENTIMENTS = ["happy", "sad", "neutral"]
MAX_SCENE_TOKEN_LENGTH = 20  # Truncate after this amount of tokens, chosen because 99% of scenes are longer than this, voiding the need to pad short scenes (shorter scenes are just dropped)


def load_sentiment_analysis_vl_prompts_list(
    data_csv_path: str,
    images_dir: str = "./data/sentiment_analysis/images",
    model: Optional[lens.HookedVLTransformer] = None,
    processor: Optional[transformers.processing_utils.ProcessorMixin] = None,
    correct_preds_only: bool = True,
    measure_acc_across_limited_labels: bool = False,
    image_size=(256, 256),
) -> List[VLPrompt]:
    """
    Load a list of Vision-Language prompts for the vision-based sentiment analysis task.
    """
    data_csv_dir = os.path.dirname(data_csv_path)
    if not os.path.exists(data_csv_path):
        assert (
            model is not None and processor is not None
        ), "Model and processor must be supplied if the CSV file doesn't exist."

        # Create the dataset file from scratch.
        data_tuples = []
        content_key = get_content_key_for_prompt_dict(model.cfg.model_name)
        for sentiment in POSSIBLE_SENTIMENTS:
            for file_path in tqdm(
                glob.glob(os.path.join(images_dir, sentiment, "*.jpeg"))
            ):
                relative_path_from_csv = os.path.relpath(file_path, data_csv_dir)
                image = load_image_for_model(
                    file_path, model.model_name, target_size=image_size
                )
                prompt = processor.apply_chat_template(
                    [get_vl_sentiment_analysis_prompt(content_key)],
                    add_generation_prompt=True,
                )
                gt_answer = to_single_token(model, sentiment)

                pred_logits = model(prompt, [image])[:, -1]
                if measure_acc_across_limited_labels:
                    limited_labels = get_sentiment_analysis_limited_labels(processor)
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


def load_sentiment_analysis_l_prompts_list(
    data_csv_path: str,
    model: Optional[lens.HookedVLTransformer] = None,
    processor: Optional[transformers.processing_utils.ProcessorMixin] = None,
    random_seed: int = 42,
    correct_preds_only: bool = True,
    measure_acc_across_limited_labels: bool = False,
) -> List[VLPrompt]:
    """
    Load a list of Language-only prompts for the text-based sentiment analysis task.

    Args:
        data_csv_path: The path to the CSV file containing the dataset.
        model: The model to use for generating predictions. Must be supplied in case the CSV doesn't exist.
        processor: The processor to use for generating prompts. Must be supplied in case the CSV doesn't exist.
        random_seed: The random seed to use for reproducibility.
        correct_preds_only: Whether to include only correct predictions in the dataset.
        measure_acc_across_limited_labels: Whether to measure accuracy across limited labels only (or consider all possible tokens).
    """
    if not os.path.exists(data_csv_path):
        assert (
            model is not None and processor is not None
        ), "Model and processor must be supplied if the CSV file doesn't exist."

        # Create the dataset from scratch.
        set_deterministic(random_seed)

        sentiment_json = json.load(open("./data/sentiment_analysis/sentiment_vl.json"))
        data_tuples = []
        for entry in tqdm(sentiment_json):
            text_description = entry["text_data"]
            gt_answer = to_single_token(model, entry["sentiment"])

            if (
                model.to_tokens(text_description, prepend_bos=False).numel()
                < MAX_SCENE_TOKEN_LENGTH
            ):
                continue

            content_key = get_content_key_for_prompt_dict(model.model_name)
            prompt = processor.apply_chat_template(
                [
                    get_l_sentiment_analysis_prompt(
                        text_description, processor, content_key
                    )
                ],
                add_generation_prompt=True,
            )

            pred_logits = model(prompt, [])[:, -1]
            if measure_acc_across_limited_labels:
                limited_labels = get_sentiment_analysis_limited_labels(processor)
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

        with open(data_csv_path, "w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["prompt", "gt_answer", "pred_answer"])
            writer.writerows(data_tuples)

    return load_prompts_from_csv(
        data_csv_path, model.model_name, correct_preds_only=correct_preds_only
    )


def load_sentiment_analysis_parallel_l_prompts(vl_prompts: List[VLPrompt], processor):
    """
    Loads a list of perfectly-aligned language-only prompts from a given list of Vision-Language prompts.
    A perfectly-aligned textual prompt for a given VLPrompt is the textual prompt that contains the description
    used to generate the image.

    NOTE NOTE NOTE: The entity isn't tested to be tokenized properly, so it might not be aligned between prompts!
    NOTE NOTE NOTE: Thus, don't use it for anything that requires positional alignment.
    """
    parallel_l_prompts = []

    sentiment_json = json.load(open("./data/sentiment_analysis/sentiment_vl.json"))
    for vl_prompt in vl_prompts:
        image_filename = os.path.basename(vl_prompt.metadata["image_path"])
        entry = [
            entry for entry in sentiment_json if entry["image_path"] == image_filename
        ][0]

        text_description = entry["text_data"]
        content_key = get_content_key_for_prompt_dict(processor.__class__.__name__)
        prompt = processor.apply_chat_template(
            [get_l_sentiment_analysis_prompt(text_description, processor, content_key)],
            add_generation_prompt=True,
        )
        parallel_l_prompts.append(VLPrompt(prompt, [], str(vl_prompt.answer)))

    return parallel_l_prompts


def get_vl_sentiment_analysis_prompt(content_key: str = "text") -> dict:
    return {
        "role": "user",
        "content": [
            {"type": "image"},
            {
                "type": "text",
                content_key: f"\nIs this scene happy, sad, or neutral? Answer in a single word.",
            },
        ],
    }


def get_l_sentiment_analysis_prompt(
    text_description, processor, content_key: str = "text"
) -> dict:
    return {
        "role": "user",
        "content": [
            {
                "type": "text",
                content_key: f'"""{truncate_to_n_tokens(text_description, processor, MAX_SCENE_TOKEN_LENGTH, " ...")}"""'
                + "\nIs this scene happy, sad, or neutral? Answer in a single word.",
            },
        ],
    }


def get_sentiment_analysis_limited_labels(processor):
    """
    Get the limited labels for the sentiment_analysis task.
    This returns the first token of the words "happy", "sad", and "neutral", depending on the model used.
    """
    first_sentiment_tokens = [
        processor(text=s, return_tensors="pt", add_special_tokens=False)[
            "input_ids"
        ].view(-1)[0]
        for s in POSSIBLE_SENTIMENTS
    ]
    return [processor.decode(t) for t in first_sentiment_tokens]
