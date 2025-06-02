import csv
import json
import os
import re
import sys
import transformers
from functools import lru_cache
from typing import List, Optional

sys.path.append("./third_party/TransformerLens")
import transformer_lens as lens
from general_utils import (
    get_content_key_for_prompt_dict,
    load_image_for_model,
    load_prompts_from_csv,
    set_deterministic,
    to_single_token,
)
from vision_language_prompts import VLPrompt

FACTUAL_RECALL_QUESTION_REGEX = r"((?:What|Which).*single word\.)"
RAW_DATA_JSON_PATH = "./data/factual_recall/qa_raw.json"
TOKENS_PER_ENTITY_AND_PREFIX = 5  # A constant number of tokens per entity, must be constant for positional alignment reasons


def load_factual_recall_vl_prompts_list(
    data_csv_path: str,
    images_dir: str = "./data/factual_recall/images",
    model: Optional[lens.HookedVLTransformer] = None,
    processor: Optional[transformers.processing_utils.ProcessorMixin] = None,
    correct_preds_only: bool = True,
    measure_acc_across_limited_labels: bool = True,
    image_size=(256, 256),
) -> List[VLPrompt]:
    data_csv_dir = os.path.dirname(data_csv_path)
    if not os.path.exists(data_csv_path):
        assert (
            model is not None and processor is not None
        ), "Model and processor must be supplied if the CSV file doesn't exist."

        # Create the dataset file from scratch.
        data_tuples = []
        content_key = get_content_key_for_prompt_dict(model.cfg.model_name)

        entity_json = json.load(open(RAW_DATA_JSON_PATH, "r"))
        for question_template, entities_answers in entity_json.items():
            for entity, answer in entities_answers:
                img_path = os.path.join(images_dir, entity.replace(" ", "_") + ".png")
                if not os.path.exists(img_path):
                    print(f"Image not found: {img_path}")
                    continue
                relative_path_from_csv = os.path.relpath(img_path, data_csv_dir)
                image = load_image_for_model(
                    img_path, model.model_name, target_size=image_size
                )
                prompt = processor.apply_chat_template(
                    [get_vl_factual_recall_prompt(question_template, content_key)],
                    add_generation_prompt=True,
                )
                gt_answer = to_single_token(model, answer)
                pred_logits = model(prompt, [image])[:, -1]
                if measure_acc_across_limited_labels:
                    limited_labels = get_factual_recall_limited_labels(
                        question_template, processor
                    )
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


def load_factual_recall_l_prompts_list(
    data_csv_path: str,
    model: Optional[lens.HookedVLTransformer] = None,
    processor: Optional[transformers.processing_utils.ProcessorMixin] = None,
    random_seed: int = 42,
    correct_preds_only: bool = True,
    measure_acc_across_limited_labels: bool = True,
) -> List[VLPrompt]:
    if not os.path.exists(data_csv_path):
        assert (
            model is not None and processor is not None
        ), "Model and processor must be supplied if the CSV file doesn't exist."

        # Create the dataset from scratch.
        set_deterministic(random_seed)

        content_key = get_content_key_for_prompt_dict(model.cfg.model_name)
        entity_json = json.load(open(RAW_DATA_JSON_PATH, "r"))
        data_tuples = []
        for question_template, entities_answers in entity_json.items():
            for entity, answer in entities_answers:
                if not entity_tokenized_correctly(entity, model):
                    continue

                prompt = processor.apply_chat_template(
                    [
                        get_l_factual_recall_prompt(
                            question_template, entity, content_key
                        )
                    ],
                    add_generation_prompt=True,
                )

                gt_answer = to_single_token(model, answer)
                pred_logits = model(prompt, [])[:, -1]
                if measure_acc_across_limited_labels:
                    limited_labels = get_factual_recall_limited_labels(
                        question_template, processor
                    )
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

        # The previous token length check might sometimes not be enough because of
        # the combination of the entity into the prompt. Thus we need to check again.
        # possible_prompt_lengths = {model.to_tokens(t[0]).numel() for t in data_tuples}
        # len_to_tuples = {
        #     l: [t for t in data_tuples if model.to_tokens(t[0]).numel() == l]
        #     for l in possible_prompt_lengths
        # }
        # data_tuples = max(len_to_tuples.values(), key=lambda tups: len(tups))

        with open(data_csv_path, "w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["prompt", "gt_answer", "pred_answer"])
            writer.writerows(data_tuples)

    return load_prompts_from_csv(
        data_csv_path, model.model_name, correct_preds_only=correct_preds_only
    )


def load_factual_recall_parallel_l_prompts(
    vl_prompts: List[VLPrompt],
    processor: transformers.processing_utils.ProcessorMixin,
    return_indices: bool = False,
):
    """
    Loads a list of perfectly-aligned language-only prompts from a given list of Vision-Language prompts.
    A perfectly-aligned Language-only prompt is a prompt that has the same entity as an image in the VL prompt.

    NOTE NOTE NOTE: The entity isn't tested to be tokenized properly, so it might not be aligned between prompts!
    NOTE NOTE NOTE: Thus, don't use this for anything that requires positional alignment.
    """
    parallel_l_prompts = []
    entity_json = json.load(open(os.path.join(RAW_DATA_JSON_PATH), "r"))
    good_prompt_indices = []
    for i, vl_prompt in enumerate(vl_prompts):
        image_filename = os.path.basename(vl_prompt.metadata["image_path"])

        entity = os.path.splitext(image_filename)[0].replace("_", " ")

        correct_question_template = ""
        for question_template, entities_answers in entity_json.items():
            if entity in [e[0] for e in entities_answers]:
                correct_question_template = question_template
                break

        if correct_question_template == "":
            raise Exception(
                f"Entity {entity} ({image_filename}) not found in the question templates."
            )

        content_key = get_content_key_for_prompt_dict(processor.__class__.__name__)
        prompt = processor.apply_chat_template(
            [
                get_l_factual_recall_prompt(
                    correct_question_template, entity, content_key
                )
            ],
            add_generation_prompt=True,
        )
        good_prompt_indices.append(i)
        parallel_l_prompts.append(VLPrompt(prompt, [], str(vl_prompt.answer)))

    if return_indices:
        return parallel_l_prompts, good_prompt_indices
    else:
        return parallel_l_prompts


def entity_tokenized_correctly(entity, model):
    """
    Check if the entity is tokenized well by the model, allowing a good positional alignment between entities.
    A good alignment should start with a first token of the "Consider" prefix, 3 entity tokens, and a last token
    for the '.' token.
    The "Consider" and "." and spaces must also be included in the test because they can change the amounts of
    tokens the entity is tokenized into.
    """
    return (
        model.to_tokens(f"Consider {entity}.", prepend_bos=False).numel()
        == TOKENS_PER_ENTITY_AND_PREFIX
    )


@lru_cache(maxsize=None)
def get_factual_recall_limited_labels(question_template, processor):
    """
    Get the possible answer tokens for a given question template.
    This function is cached using @lru_cache to avoid re-identifying
    the possible answers (a process that requires loading up the entity json).
    """
    if question_template is None:
        entity_json = json.load(open(os.path.join(RAW_DATA_JSON_PATH), "r"))
        possible_answers = list(
            {
                ea[1]
                for entities_answers in entity_json.values()
                for ea in entities_answers
            }
        )
    else:
        entity_json = json.load(open(os.path.join(RAW_DATA_JSON_PATH), "r"))
        possible_answers = list({ea[1] for ea in entity_json[question_template]})
    possible_answers_first_token = []
    for answer in possible_answers:
        label = (
            processor(text=answer, return_tensors="pt", add_special_tokens=False)[
                "input_ids"
            ]
            .view(-1)[0]
            .item()
        )
        possible_answers_first_token.append(processor.decode(label))
    return possible_answers_first_token


def get_factual_recall_question_template(question):
    """
    Get the question template from the question.
    The question template is the part of the question that is presented after the entity.
    """
    # Split the question into parts and get the first part
    template_from_question = re.findall(FACTUAL_RECALL_QUESTION_REGEX, question)[0]
    entity_json = json.load(open(os.path.join(RAW_DATA_JSON_PATH), "r"))
    for ref_question_template in entity_json.keys():
        if template_from_question in ref_question_template:
            return ref_question_template

    raise Exception(f"Question template not found in the question: {question}")


def get_vl_factual_recall_prompt(question_template, content_key):
    no_prefix_template = question_template.split(".\n")[1]
    return {
        "role": "user",
        "content": [
            {"type": "image"},
            {
                "type": "text",
                content_key: no_prefix_template,
            },
        ],
    }


def get_l_factual_recall_prompt(question_template, entity, content_key):
    return {
        "role": "user",
        "content": [
            {
                "type": "text",
                content_key: question_template.replace("[X]", entity),
            },
        ],
    }
