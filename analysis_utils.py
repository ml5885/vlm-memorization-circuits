#
# Contains various functions used across analysis scripts (such as model loading, dataset loading, etc)
#

import logging
import sys
from typing import Union

from arithmetic_utils import (
    get_arithmetic_limited_labels,
    load_arithmetic_l_prompts_list,
    load_arithmetic_parallel_l_prompts,
    load_arithmetic_vl_prompts_list,
)
from clevr_color_ordering_utils import (
    CLEVR_COLORS,
    get_color_ordering_limited_labels,
    load_color_ordering_l_prompts_list,
    load_color_ordering_parallel_l_prompts,
    load_color_ordering_vl_prompts_list,
)
from component import Component
from factual_recall_utils import (
    get_factual_recall_limited_labels,
    load_factual_recall_l_prompts_list,
    load_factual_recall_parallel_l_prompts,
    load_factual_recall_vl_prompts_list,
)
from general_utils import (
    balanced_answers_train_test_split,
    get_gpu_count,
    get_image_size_for_model,
    setup_random_counterfactual_prompts,
)
from object_counting_utils import (
    COUNTING_SEQ_LEN,
    get_counting_limited_labels,
    load_counting_l_prompts_list,
    load_counting_parallel_l_prompts,
    load_counting_vl_prompts_list,
)
from sentiment_analysis_utils import (
    get_sentiment_analysis_limited_labels,
    load_sentiment_analysis_l_prompts_list,
    load_sentiment_analysis_parallel_l_prompts,
    load_sentiment_analysis_vl_prompts_list,
)

sys.path.append("./third_party/TransformerLens")

import torch
import transformer_lens as lens
from transformers import (
    Qwen2VLForConditionalGeneration,
    Gemma3ForConditionalGeneration,
    MllamaForConditionalGeneration,
    LlavaForConditionalGeneration,
    AutoProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
)


SUPPORTED_TASKS = [
    "counting",
    "arithmetic",
    "color_ordering",
    "factual_recall",
    "sentiment_analysis",
]


def load_model(
    model_name: str,
    model_path: str,
    device: Union[str, torch.device],
    use_tlens_wrapper: bool = True,
    extra_hooks: bool = True,
    torch_dtype: torch.dtype = torch.float32,
):
    if "llama3.2" in model_name.lower():
        inner_model = MllamaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="cpu",
        )
        processor = AutoProcessor.from_pretrained(model_path, device=device)
        if use_tlens_wrapper:
            model = lens.HookedVLTransformer.from_pretrained(
                model_name=model_name,
                hf_model=inner_model,
                processor=processor,
                fold_ln=True,
                center_unembed=True,
                center_writing_weights=True,
                fold_value_biases=True,
                n_devices=get_gpu_count(),
                device=device,
            )
            model.set_use_split_qkv_input(extra_hooks)
            model.set_use_attn_result(extra_hooks)
            model.set_use_hook_mlp_in(extra_hooks)
            model.eval()
        else:
            model = inner_model
        return model, processor

    elif "qwen" in model_name.lower():
        inner_model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="cpu",
        )
        processor = AutoProcessor.from_pretrained(model_path)
        if use_tlens_wrapper:
            inner_model.vision_model = inner_model.visual
            model = lens.HookedVLTransformer.from_pretrained(
                model_name=model_name,
                hf_model=inner_model,
                processor=processor,
                fold_ln=True,
                center_unembed=True,
                center_writing_weights=True,  # False,
                fold_value_biases=True,
                n_devices=get_gpu_count(),
                device=device,
            )
            model.cfg.default_prepend_bos = False  # To match HF Qwen model forward pass
            model.set_use_split_qkv_input(extra_hooks)
            model.set_use_attn_result(extra_hooks)
            model.set_use_hook_mlp_in(extra_hooks)
            model.eval()
        else:
            model = inner_model
        model.model_name = model_name
        return model, processor

    elif "pixtral" in model_name.lower() or "llava" in model_name.lower():
        inner_model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="cpu",
        )
        processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
        if use_tlens_wrapper:
            inner_model.vision_model = inner_model.vision_tower
            model = lens.HookedVLTransformer.from_pretrained(
                model_name=model_name,
                hf_model=inner_model,
                processor=processor,
                fold_ln=True,
                center_unembed=True,
                center_writing_weights=True,
                fold_value_biases=True,
                n_devices=get_gpu_count(),
                device=device,
            )
            model.cfg.default_prepend_bos = (
                False if "pixtral" in model_name.lower() else True
            )
            # model.cfg.default_prepend_bos = False
            model.set_use_split_qkv_input(extra_hooks)
            model.set_use_attn_result(extra_hooks)
            model.set_use_hook_mlp_in(extra_hooks)
            model.eval()
        else:
            model = inner_model
        model.model_name = model_name
        return model, processor

    elif "gemma-3" in model_name.lower():
        inner_model = Gemma3ForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="cpu",
        )
        processor = AutoProcessor.from_pretrained(model_path)
        # processor.chat_template = processor.chat_template.replace("{{ bos_token }}", "")
        if use_tlens_wrapper:
            inner_model.vision_model = inner_model.vision_tower
            model = lens.HookedVLTransformer.from_pretrained(
                model_name=model_name,
                hf_model=inner_model,
                processor=processor,
                fold_ln=False,  # ,
                center_unembed=True,  # True,
                center_writing_weights=True,  # True,
                fold_value_biases=False,  # True,
                n_devices=get_gpu_count(),
                device=device,
            )
            model.cfg.default_prepend_bos = False
            model.set_use_split_qkv_input(extra_hooks)
            model.set_use_attn_result(extra_hooks)
            model.set_use_hook_mlp_in(extra_hooks)
            model.eval()
        else:
            model = inner_model
        model.model_name = model_name
        return model, processor

    else:
        print("WARNING: Using model not officially supported in load_model")
        inner_model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if use_tlens_wrapper:
            model = lens.HookedTransformer.from_pretrained(
                model_name=model_name,
                hf_model=inner_model,
                tokenizer=tokenizer,
                fold_ln=True,
                center_unembed=True,
                center_writing_weights=True,
                fold_value_biases=True,
                n_devices=get_gpu_count(),
                device=device,
            )
            model.set_use_split_qkv_input(extra_hooks)
            model.set_use_attn_result(extra_hooks)
            model.set_use_hook_mlp_in(extra_hooks)
        else:
            model = inner_model
        model.model_name = model_name
        return model, tokenizer


def load_dataset(
    model: lens.HookedVLTransformer,
    processor: AutoProcessor,
    task_name: str,
    model_name: str,
    language_only: bool,
    seed: int,
    train_test_split_ratio: float,
    correct_preds_only: bool = True,
):
    """
    Loads a dataset of correctly-completed prompts for a given model and task.
    Each prompt is paired with a counterfactual prompt, which is a prompt with a different answer.

    Args:
        model (lens.HookedVLTransformer): The model to load the dataset for.
        processor (AutoProcessor): The processor to use for the model.
        task_name (str): The name of the task to load the dataset for.
        model_name (str): The name of the model to load the dataset for.
        language_only (bool): Whether to load only language prompts (True) or vision-languageprompts (False).
        seed (int): A random seed to be used in generation, CF pairing, and train-test split.
        train_test_split_ratio (float): The ratio of prompts to use for training vs. evaluation.

    Returns:
        Tuple[List[VLPrompt], List[VLPrompt], List[VLPrompt]]: A tuple containing the list of all generated prompts,
            a list of discovery prompts, and a list of evaluation prompts.
    """
    # Create / Load the base dataset
    data_path = f"./data/{task_name}/{model_name}_{'textual' if language_only else 'visual'}_data.csv"

    if task_name.lower() == "counting":
        is_model_acc_low = any([n in model_name.lower() for n in ["llava", "pixtral"]])
        total_prompt_count = 215 if is_model_acc_low else 250
        possible_answers = [str(x) for x in range(1, COUNTING_SEQ_LEN + 1)]
        if language_only:
            vl_prompts = load_counting_l_prompts_list(
                data_path,
                model=model,
                processor=processor,
                random_seed=seed,
                correct_preds_only=correct_preds_only,
                measure_acc_across_limited_labels=is_model_acc_low,
            )
        else:
            vl_prompts = load_counting_vl_prompts_list(
                data_path,
                images_dir=f"./data/counting/images",
                model=model,
                processor=processor,
                correct_preds_only=correct_preds_only,
                measure_acc_across_limited_labels=is_model_acc_low,
                image_size=get_image_size_for_model(model_name),
            )

    elif task_name.lower() == "color_ordering":
        total_prompt_count = 250
        possible_answers = get_color_ordering_limited_labels(model.processor)
        if language_only:
            vl_prompts = load_color_ordering_l_prompts_list(
                data_path,
                model,
                processor,
                correct_preds_only=correct_preds_only,
                measure_acc_across_limited_labels=True,
                prompts_per_scene=2 if "pixtral" in model_name else 1,
                random_seed=seed,
            )
        else:
            vl_prompts = load_color_ordering_vl_prompts_list(
                data_path,
                model,
                processor,
                correct_preds_only=correct_preds_only,
                measure_acc_across_limited_labels=True,
                prompts_per_scene=2 if "pixtral" in model_name else 1,
                image_size=get_image_size_for_model(model_name),
                random_seed=seed,
            )

    elif task_name.lower() == "arithmetic":
        total_prompt_count = 210 if "pixtral" in model_name else 250
        possible_answers = get_arithmetic_limited_labels()
        if language_only:
            vl_prompts = load_arithmetic_l_prompts_list(
                data_path,
                model=model,
                processor=processor,
                random_seed=seed,
                correct_preds_only=correct_preds_only,
                measure_acc_across_limited_labels=True,
            )
        else:
            vl_prompts = load_arithmetic_vl_prompts_list(
                data_path,
                images_dir=f"./data/arithmetic/images",
                model=model,
                processor=processor,
                correct_preds_only=correct_preds_only,
                measure_acc_across_limited_labels=True,
            )

    elif task_name.lower() == "sentiment_analysis":
        if "qwen" in model_name:
            total_prompt_count = 220
        elif "pixtral" in model_name:
            total_prompt_count = 170
        else:
            total_prompt_count = 215
        possible_answers = get_sentiment_analysis_limited_labels(processor)
        if language_only:
            vl_prompts = load_sentiment_analysis_l_prompts_list(
                data_path,
                model=model,
                processor=processor,
                random_seed=seed,
                correct_preds_only=correct_preds_only,
                measure_acc_across_limited_labels=True,
            )
        else:
            vl_prompts = load_sentiment_analysis_vl_prompts_list(
                data_path,
                images_dir=f"./data/sentiment_analysis/images",
                model=model,
                processor=processor,
                correct_preds_only=correct_preds_only,
                measure_acc_across_limited_labels=True,
                image_size=get_image_size_for_model(model_name),
            )
    elif task_name.lower() == "factual_recall":
        total_prompt_count = 250
        if language_only:
            vl_prompts = load_factual_recall_l_prompts_list(
                data_path,
                model=model,
                processor=processor,
                random_seed=seed,
                correct_preds_only=correct_preds_only,
                measure_acc_across_limited_labels=True,
            )
        else:
            vl_prompts = load_factual_recall_vl_prompts_list(
                data_path,
                images_dir=f"./data/factual_recall/images",
                model=model,
                processor=processor,
                correct_preds_only=correct_preds_only,
                measure_acc_across_limited_labels=True,
                image_size=get_image_size_for_model(model_name),
            )
        possible_answers = list({p.answer for p in vl_prompts})
    else:
        raise ValueError(f"Unknown task {task_name}")

    assert (
        len(vl_prompts) >= total_prompt_count
    ), f"Too few data points ({len(vl_prompts)} / {total_prompt_count})"

    # Split to train and eval
    original_vl_prompts = vl_prompts  # Used in mean cache calculation
    vl_prompts, eval_vl_prompts = balanced_answers_train_test_split(
        vl_prompts,
        possible_answers=possible_answers,
        target_total_prompt_count=total_prompt_count,
        train_test_split_ratio=train_test_split_ratio,
        seed=seed,
    )
    logging.info(
        f"Split to {len(vl_prompts)} discovery prompts and {len(eval_vl_prompts)} eval prompts"
    )

    # Setup counterfactual prompts for each prompt
    logging.info("Setting up counterfactuals for each prompt")
    vl_prompts = setup_random_counterfactual_prompts(
        vl_prompts, seed=seed, task_name=task_name
    )
    eval_vl_prompts = setup_random_counterfactual_prompts(
        eval_vl_prompts, seed=seed, task_name=task_name
    )
    for prompt in vl_prompts + eval_vl_prompts:
        split = "Train" if prompt in vl_prompts else "Eval"
        logging.debug(
            f"{split} - Prompt: {prompt.prompt}; Counterfactual: {prompt.cf_prompt}; Answer: {prompt.answer}; Counterfactual answer: {prompt.cf_answer}"
        )

    return original_vl_prompts, vl_prompts, eval_vl_prompts


def load_l_vl_scores(task_name, model_name, metric="LD"):
    """
    Load the L and VL node attribution scores scores for a given model, task and metric.
    """
    logging.info(f"Loading L and VL scores")
    l_scores = torch.load(
        f"./data/{task_name}/results/{model_name}/node_scores/nap_ig_l_ig=5_metric={metric}.pt",
        weights_only=True,
    )
    l_scores = {k: v.abs() for k, v in l_scores.items()}
    vl_scores = torch.load(
        f"./data/{task_name}/results/{model_name}/node_scores/nap_ig_vl_ig=5_metric={metric}.pt",
        weights_only=True,
    )
    vl_scores = {k: v.abs() for k, v in vl_scores.items()}
    return l_scores, vl_scores


def get_limited_labels_for_task(task, model):
    if task == "counting":
        labels = get_counting_limited_labels()
    elif task == "color_ordering":
        labels = get_color_ordering_limited_labels(model.processor)
    elif task == "arithmetic":
        labels = get_arithmetic_limited_labels()
    elif task == "sentiment_analysis":
        labels = get_sentiment_analysis_limited_labels(model.processor)
    elif task == "factual_recall":
        labels = get_factual_recall_limited_labels(None, model.processor)
        pass
    else:
        raise ValueError(f"Task {task} not recognized")
    return model.to_tokens(labels, prepend_bos=False).view(-1)


def get_parallel_l_prompts(vl_prompts, processor, task_name, seed):
    if task_name == "counting":
        return load_counting_parallel_l_prompts(vl_prompts, processor, seed)
    elif task_name == "arithmetic":
        return load_arithmetic_parallel_l_prompts(vl_prompts, processor)
    elif task_name == "color_ordering":
        return load_color_ordering_parallel_l_prompts(vl_prompts, processor)
    elif task_name == "sentiment_analysis":
        return load_sentiment_analysis_parallel_l_prompts(vl_prompts, processor)
    elif task_name == "factual_recall":
        return load_factual_recall_parallel_l_prompts(vl_prompts, processor)
    else:
        raise ValueError(f"Task {task_name} not recognized")
