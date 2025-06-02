from functools import partial
import torch
import argparse
import logging
import os
import sys
import json

try:
    sys.path.append("third_party/TransformerLens")
    import transformer_lens as lens  # Some python problem causes this to throw on the first import
except:
    import transformer_lens as lens

from component import Component
from modality_alignment_utils import get_image_positions
from vision_language_prompts import VLPrompt

# from script_backpatching_experiment import backpatching
from general_utils import (
    generate_activations,
    get_content_key_for_prompt_dict,
    get_text_seq_len_and_image_seq_len,
    load_image_for_model,
    set_deterministic,
    to_single_token,
)
from analysis_utils import load_model
from evaluation_utils import model_accuracy


VQA_ANNOTATIONS_PATH = r"./vqa/v2_mscoco_val2014_annotations.json"
VQA_QUESTIONS_PATH = r"./vqa/v2_OpenEnded_mscoco_val2014_questions.json"
VQA_IMAGES_PATH = r"./vqa/val2014"
IMAGE_NAME_PREFIX = f"COCO_val2014"

torch.set_grad_enabled(False)
device = "cuda"


def load_vqa_dataset(model, processor, args):
    # Load annotations and questions
    with open(VQA_ANNOTATIONS_PATH, "r") as f:
        annotations_data = json.load(f)
    with open(VQA_QUESTIONS_PATH, "r") as f:
        questions_data = json.load(f)

    # Create a dictionary for quick question lookup by question_id
    question_id_to_question_text = {
        q["question_id"]: q["question"] for q in questions_data["questions"]
    }
    used_images = set()

    dataset = []
    for ann in annotations_data["annotations"]:
        if len(dataset) >= 3000:
            break

        question_id = ann["question_id"]
        image_id = ann["image_id"]
        if image_id in used_images:
            # Skip if the image has already been used
            continue

        used_images.add(image_id)
        answer = ann["multiple_choice_answer"]
        first_tok_answer = to_single_token(model, answer)

        # Get the original question text
        if question_id not in question_id_to_question_text:
            print(
                f"Warning: Question ID {question_id} found in annotations but not in questions. Skipping."
            )
            continue
        original_question = question_id_to_question_text[question_id]
        modified_question = f"{original_question} Answer in a single word."
        content_key = get_content_key_for_prompt_dict(args.model_name)
        prompt = processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", content_key: modified_question},
                    ],
                }
            ],
            add_generation_prompt=True,
        )
        image_path = os.path.join(
            VQA_IMAGES_PATH, f"{IMAGE_NAME_PREFIX}_{str(image_id).zfill(12)}.jpg"
        )

        try:
            img = load_image_for_model(
                image_path, args.model_name, target_size=(252, 252)
            )
            dataset.append(
                VLPrompt(prompt=prompt, images=[img], answer=first_tok_answer)
            )
        except FileNotFoundError:
            print(
                f"Warning: Image file not found at {image_path}. Skipping entry for question ID {question_id}."
            )
        except Exception as e:
            print(
                f"Warning: Could not load image {image_path} due to {e}. Skipping entry for question ID {question_id}."
            )

    return dataset


def vqa_backpatching(
    model,
    args,
    eval_vl_prompts,
    src_layer,
    dst_layer,
    repeat_processing_in_data_positions=True,
    layer_window_size=1,
    cached_activations=None,
):
    """
    Run a single back-patching experiment, with a given l_src -> l_dst.
    This is a version of script_backpatching_experiment.backpatching function, but changed to match the limitations of the VQAv2 dataset
    (i.e., each prompt is of different length, etc).

    Args:
        model (lens.HookedTransformer): The model to run the experiment on.
        args (argparse.Namespace): The arguments for the experiment.
        eval_vl_prompts (List[VLPrompt]): The visual prompts to use for the experiment.
        src_layer (int): The layer to back-patch from.
        dst_layer (int): The layer to back-patch to.
        repeat_processing_in_data_positions (bool): Whether to repeat the processing in the data positions.
        layer_window_size (int): The size of the window to use for back-patching.
        cached_activations (List[torch.Tensor]): Clean activations on all VL prompts. Used for efficiency. If None, they will be generated.
            Each tensor in the list is of shape (n_layers, seq_len, d_model).

    Return:
        acc (float): The accuracy of the model after back-patching.
        cached_activations (List[torch.Tensor]): The cached activations on all VL prompts. To be used in further calls to the function.
            Each tensor in the list is of shape (n_layers, seq_len, d_model).
    """
    data_positions = get_image_positions(args.model_name, "vqa")

    logging.info(f"Backpatching {src_layer} -> {dst_layer}")
    if cached_activations is None:
        components = [
            Component("resid_post", layer=l) for l in range(model.cfg.n_layers)
        ]
        cached_activations = []
        for i, prompt in enumerate(eval_vl_prompts):
            seq_len = get_text_seq_len_and_image_seq_len(
                model, [prompt.prompt], [prompt.images]
            )[0]
            cached_activations.append(
                torch.stack(
                    generate_activations(
                        model,
                        [prompt],
                        components,
                        batch_size=1,
                        text_seq_len=seq_len,
                        verbose=False,
                    )
                )[
                    :, 0, :, :
                ]  # Shape: (n_layers, seq_len, d_model)
            )

    def hook_dst_layer(value, hook, prompt_index, src_layer):
        """
        Patch the activation of a later src_layer to an earlier dst_layer, in the image positions only.
        """
        value[:, data_positions[0] : data_positions[1], :] = cached_activations[
            prompt_index
        ][
            src_layer,
            data_positions[0] : data_positions[1],
            :,
        ]
        return value

    def hook_override_repeat_data_positions(value, hook, prompt_index):
        value[:, data_positions[0] : data_positions[1], :] = cached_activations[
            prompt_index
        ][
            hook.layer(),
            data_positions[0] : data_positions[1],
            :,
        ]
        return value

    src_dst_layers = list(
        zip(
            range(
                src_layer - layer_window_size // 2,
                src_layer + 1 + layer_window_size // 2,
            ),
            range(
                dst_layer - layer_window_size // 2,
                dst_layer + 1 + layer_window_size // 2,
            ),
        )
    )
    src_dst_layers = [
        (src, dst)
        for src, dst in src_dst_layers
        if src >= 0
        and dst >= 0
        and src < model.cfg.n_layers
        and dst < model.cfg.n_layers
    ]
    src_layers = [src for src, _ in src_dst_layers]
    dst_layers = [dst for _, dst in src_dst_layers]

    all_answer_tokens = torch.stack(
        [model.to_tokens(a) for a in {p.answer for p in eval_vl_prompts}]
    ).view(-1)
    correct = torch.zeros(len(eval_vl_prompts))
    for prompt_index, prompt in enumerate(eval_vl_prompts):
        hooks = [
            (
                f"blocks.{dl}.hook_resid_post",
                partial(hook_dst_layer, prompt_index=prompt_index, src_layer=sl),
            )
            for sl, dl in zip(src_layers, dst_layers)
        ]

        if not repeat_processing_in_data_positions:
            # In this configuration, we want to make sure the back-patched activations don't go
            # through further processing in the image positions. Thus we override the activations
            # of the image positions in layers later than dst_layer to be the same as the
            # activations in dst_layer.
            hooks += [
                (
                    f"blocks.{l}.hook_resid_post",
                    partial(
                        hook_override_repeat_data_positions, prompt_index=prompt_index
                    ),
                )
                for l in range(
                    dst_layer + 1 + layer_window_size // 2, model.cfg.n_layers
                )
            ]

        # Measure accuracy
        answer = model.to_tokens(prompt.answer, prepend_bos=False).view(-1)
        logits = model.run_with_hooks(prompt.prompt, prompt.images, fwd_hooks=hooks)
        pred_label_index = logits[:, -1, all_answer_tokens.to(logits.device)].argmax(-1)
        pred_label = all_answer_tokens[pred_label_index.to(all_answer_tokens.device)]
        correct[prompt_index] = pred_label.cpu() == answer.cpu()

    acc = correct.float().mean().item()
    logging.info(f"Accuracy: {acc :.3f}")
    return acc, cached_activations


def parse_args():
    parser = argparse.ArgumentParser()

    # General arguments
    parser.add_argument("--model_name", type=str, help="Name of the model to be loaded")
    parser.add_argument("--model_path", type=str, help="Path to the model to be loaded")
    parser.add_argument("--seed", type=int, help="Random seed", default=42)

    # Backpatching specific arguments
    parser.add_argument(
        "--src_layer_range", type=int, nargs=2, help="The layer range to backpatch from"
    )
    parser.add_argument(
        "--dst_layer_range", type=int, nargs=2, help="The layer range to backpatch to"
    )

    args = parser.parse_args()
    return args


def main():
    logging.info("Running script_backpatch_vqav2.py")
    args = parse_args()

    set_deterministic(args.seed)

    # Load model
    logging.info(f"Loading model {args.model_name} from {args.model_path}")
    model, processor = load_model(
        args.model_name,
        args.model_path,
        device,
        use_tlens_wrapper=True,
        extra_hooks=True,
        torch_dtype=torch.float32,
    )
    logging.info("Model loaded")

    # Load datasets
    vl_prompts = load_vqa_dataset(model, processor, args)

    # Backpatching experiments
    results_path = (
        f"./data/vqa/results/{args.model_name}/backpatching_results_seed={args.seed}.pt"
    )
    if os.path.exists(results_path):
        logging.info(f"Results already exist at {results_path}. Loading them")
        results_dict, src_layer_range, dst_layer_range = torch.load(results_path)
    else:
        results_dict = {}
        src_layer_range = range(args.src_layer_range[0], args.src_layer_range[1])
        dst_layer_range = range(args.dst_layer_range[0], args.dst_layer_range[1])

    # Clean accs to beat
    logging.info("Running baseline")
    if "clean_accs" not in results_dict:
        all_answer_tokens = torch.stack(
            [model.to_tokens(a) for a in {p.answer for p in vl_prompts}]
        ).view(-1)
        unpatched_acc = model_accuracy(
            model,
            vl_prompts,
            batch_size=1,
            limited_labels=all_answer_tokens,
            verbose=False,
        )
        results_dict["clean_accs"] = (unpatched_acc,)
    logging.info(f"Unpatched acc: {results_dict['clean_accs'][0] :.3f}")

    # Perform back-patching experiments across all settings
    cached_activations = None
    for repeat_processing_in_data_positions in [True]:
        for layer_window_size in [5, 3, 1]:

            logging.info(
                f"Running backpatching for {layer_window_size=}, {repeat_processing_in_data_positions=}"
            )

            current_cfg = (repeat_processing_in_data_positions, layer_window_size)
            current_cfg_accs = results_dict.get(
                current_cfg,
                -1 * torch.ones((len(src_layer_range), len(dst_layer_range))),
            )
            for i, src_layer in enumerate(src_layer_range):
                for j, dst_layer in enumerate(dst_layer_range):
                    if current_cfg_accs[i, j] != -1:
                        continue
                    if dst_layer >= src_layer:
                        # We don't want to backpatch to the same layer or a later layer
                        current_cfg_accs[i, j] = 0
                    else:
                        if current_cfg_accs[i, j] == -1:
                            current_cfg_accs[i, j], cached_activations = (
                                vqa_backpatching(
                                    model,
                                    args,
                                    vl_prompts,
                                    src_layer,
                                    dst_layer,
                                    repeat_processing_in_data_positions=repeat_processing_in_data_positions,
                                    layer_window_size=layer_window_size,
                                    cached_activations=cached_activations,
                                )
                            )

                    results_dict[current_cfg] = current_cfg_accs
                    torch.save(
                        (results_dict, src_layer_range, dst_layer_range), results_path
                    )

    logging.info("Analysis complete")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
