# Hack to avoid some import problem due to the library being a subfolder
import torch
import argparse
import logging
import os
import sys

try:
    sys.path.append("third_party/TransformerLens")
    import transformer_lens as lens  # Some python problem causes this to throw on the first import
except:
    import transformer_lens as lens

from factual_recall_utils import (
    get_factual_recall_limited_labels,
    get_factual_recall_question_template,
)
from functools import partial
from modality_alignment_utils import get_image_positions, get_text_sequence_positions
from component import Component
from general_utils import (
    generate_activations,
    get_text_seq_len_and_image_seq_len,
    set_deterministic,
)
from analysis_utils import (
    SUPPORTED_TASKS,
    get_limited_labels_for_task,
    get_parallel_l_prompts,
    load_model,
    load_dataset,
)
from evaluation_utils import model_accuracy, model_accuracy_for_factual_recall


torch.set_grad_enabled(False)
device = "cuda"


def backpatching(
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

    Args:
        model (lens.HookedTransformer): The model to run the experiment on.
        args (argparse.Namespace): The arguments for the experiment.
        eval_vl_prompts (List[VLPrompt]): The visual prompts to use for the experiment.
        src_layer (int): The layer to back-patch from.
        dst_layer (int): The layer to back-patch to.
        repeat_processing_in_data_positions (bool): Whether to repeat the processing in the data positions.
        layer_window_size (int): The size of the window to use for back-patching.
        cached_activations (torch.Tensor): Clean activations on all VL prompts. Used for efficiency. If None, they will be generated.

    Return:
        acc (float): The accuracy of the model after back-patching.
        cached_activations (torch.Tensor): The cached activations on all VL prompts. To be used in further calls to the function.
    """
    is_control = len(eval_vl_prompts[0].images) == 0
    if is_control:
        data_positions = get_text_sequence_positions(args.model_name, args.task_name)
    else:
        data_positions = get_image_positions(args.model_name, args.task_name)

    logging.info(f"Backpatching {src_layer} -> {dst_layer}")
    if cached_activations is None:
            components = [
                Component("resid_post", layer=l) for l in range(model.cfg.n_layers)
            ]
            seq_len = get_text_seq_len_and_image_seq_len(
                model, [eval_vl_prompts[0].prompt], [eval_vl_prompts[0].images]
            )[0]
            cached_activations = torch.stack(
                generate_activations(
                    model,
                    eval_vl_prompts,
                    components,
                    batch_size=1,
                    text_seq_len=seq_len,
                )
            )  # Shape: (n_layers, n_prompts, seq_len, d_model)

    def hook_dst_layer(value, hook, prompt_index, src_layer):
        """
        Patch the activation of a later src_layer to an earlier dst_layer, in the image positions only.
        """
        value[:, data_positions[0] : data_positions[1], :] = cached_activations[
            src_layer,
            prompt_index,
            data_positions[0] : data_positions[1],
            :,
        ]
        return value

    def hook_override_repeat_data_positions(value, hook, prompt_index):
        value[:, data_positions[0] : data_positions[1], :] = cached_activations[
            hook.layer(),
            prompt_index,
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
        if args.task_name == "factual_recall":
            # HACK find factual recall limited labels in a special way (because each task has its own limited labels)
            template = get_factual_recall_question_template(prompt.prompt)
            limited_tokens = get_factual_recall_limited_labels(
                template, model.processor
            )
            limited_labels = (
                model.to_tokens(limited_tokens, prepend_bos=False)
                .to(logits.device)
                .view(-1)
            )
        else:
            limited_labels = (
                get_limited_labels_for_task(args.task_name, model)
                .to(logits.device)
                .view(-1)
            )
        pred_label_index = logits[:, -1, limited_labels].argmax(-1)
        pred_label = limited_labels[pred_label_index]
        correct[prompt_index] = pred_label.cpu() == answer.cpu()

    acc = correct.float().mean().item()
    logging.info(f"{'Control ' if is_control else ''}Accuracy: {acc :.3f}")
    return acc, cached_activations


def looped_backpatching(
    model,
    args,
    eval_vl_prompts,
    max_loop_count,
    src_layer,
    dst_layer,
    repeat_processing_in_data_positions=False,
    layer_window_size=1,
):
    """
    Runs an iterative back-patching experiment. Largely same args as backpatching.
    """
    assert repeat_processing_in_data_positions, "Unsupported otherwise"

    img_positions = get_image_positions(
        args.model_name, args.task_name, return_range=False
    )
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

    src_layer_activations = [None for i in src_layers]
    src_activations_full = False  # Flag to mark that the src activations have been filled and should be used to fill dst activations

    def hook_dst_layer(value, hook, prompt_index, src_layer):
        """
        Patch the activation of a later src_layer to an earlier dst_layer, in the image positions only.
        """
        # if src_layer_activations[src_layers.index(src_layer)] is not None:
        if src_activations_full:
            value[:, img_positions[0] : img_positions[1], :] = src_layer_activations[
                src_layers.index(src_layer)
            ][
                prompt_index,
                img_positions[0] : img_positions[1],
                :,
            ]
        return value

    def hook_src_layer(value, hook, prompt_index, src_layer_activations):
        """
        Save the activation of the src_layer to a global cache, in the image positions only.
        """
        if src_layer_activations[src_layers.index(hook.layer())] is None:
            # Initialize for the first time
            src_layer_activations[src_layers.index(hook.layer())] = torch.zeros(
                len(eval_vl_prompts), *value.shape[1:]
            )

        src_layer_activations[src_layers.index(hook.layer())][prompt_index] = (
            value.squeeze(0)
        )

    looped_accs = []
    for loop_iter in range(max_loop_count):
        correct = torch.zeros(len(eval_vl_prompts))
        for prompt_index, prompt in enumerate(eval_vl_prompts):
            hooks = [
                (
                    f"blocks.{dl}.hook_resid_post",
                    partial(hook_dst_layer, prompt_index=prompt_index, src_layer=sl),
                )
                for sl, dl in zip(src_layers, dst_layers)
            ] + [
                (
                    f"blocks.{sl}.hook_resid_post",
                    partial(
                        hook_src_layer,
                        prompt_index=prompt_index,
                        src_layer_activations=src_layer_activations,
                    ),
                )
                for sl in src_layers
            ]

            # Measure accuracy
            answer = model.to_tokens(prompt.answer, prepend_bos=False).view(-1)
            logits = model.run_with_hooks(prompt.prompt, prompt.images, fwd_hooks=hooks)
            limited_labels = (
                get_limited_labels_for_task(args.task_name, model)
                .to(logits.device)
                .view(-1)
            )
            pred_label_index = logits[:, -1, limited_labels].argmax(-1)
            pred_label = limited_labels[pred_label_index]
            correct[prompt_index] = pred_label.cpu() == answer.cpu()

        src_activations_full = (
            True  # Change to True after first full iteration over prompts
        )
        looped_accs.append(correct.float().mean().item())

    return looped_accs


def filter_bad_sequence_lengths(l_prompts, vl_prompts, model):
    """
    Filter out any prompts that aren't aligned in positions.
    This is done by finding the most common textual prompt sequence length and filtering out any prompts
    that don't match it (as well as their visual analogs).
    """
    logging.info("Filtering bad sequence lengths")
    most_common_seq_len = max(
        set([model.to_tokens(l.prompt).numel() for l in l_prompts]),
        key=lambda x: len(
            [l for l in l_prompts if model.to_tokens(l.prompt).numel() == x]
        ),
    )
    logging.info(f"Most common sequence length: {most_common_seq_len}")
    filtered_indices = [
        i
        for i in range(len(l_prompts))
        if model.to_tokens(l_prompts[i].prompt).numel() == most_common_seq_len
    ]
    filtered_l = [l_prompts[i] for i in filtered_indices]
    filtered_vl = [vl_prompts[i] for i in filtered_indices]

    logging.info(
        f"Post filter: {len(filtered_l)} L prompts and {len(filtered_vl)} VL prompts"
    )
    return filtered_l, filtered_vl


def parse_args():
    parser = argparse.ArgumentParser()

    # General arguments
    parser.add_argument("--model_name", type=str, help="Name of the model to be loaded")
    parser.add_argument("--model_path", type=str, help="Path to the model to be loaded")
    parser.add_argument("--seed", type=int, help="Random seed", default=42)
    parser.add_argument(
        "--task_name",
        type=str,
        choices=SUPPORTED_TASKS,
        help="Name of the task to be localized",
    )

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
    logging.info("Running script_backpatching_experiment.py")
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
    logging.info(f"Loading dataset for task {args.task_name}")
    vl_prompts = load_dataset(
        model=model,
        processor=processor,
        task_name=args.task_name,
        model_name=args.model_name,
        language_only=False,
        seed=args.seed,
        correct_preds_only=False,  # Important because we don't want to have 100% accuracy, but instead want to be able to improve it
        train_test_split_ratio=0.5,  # Doesn't matter, we don't split to discovery and test here
    )[0]
    l_prompts = get_parallel_l_prompts(vl_prompts, processor, args.task_name, args.seed)
    logging.info(f"Loaded {len(vl_prompts)} VL prompts and {len(l_prompts)} L prompts")

    l_prompts, vl_prompts = filter_bad_sequence_lengths(l_prompts, vl_prompts, model)

    # Backpatching experiments
    results_path = (
        f"./data/{args.task_name}/results/{args.model_name}/backpatching_results.pt"
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
        if args.task_name == "factual_recall":
            unpatched_acc = model_accuracy_for_factual_recall(
                model, vl_prompts, batch_size=1, verbose=False
            )
            unpatched_control_acc = model_accuracy_for_factual_recall(
                model,
                l_prompts,
                batch_size=1,
                verbose=False,
            )
        else:
            unpatched_acc = model_accuracy(
                model,
                vl_prompts,
                batch_size=1,
                limited_labels=get_limited_labels_for_task(args.task_name, model),
                verbose=False,
            )
            unpatched_control_acc = model_accuracy(
                model,
                l_prompts,
                batch_size=1,
                limited_labels=get_limited_labels_for_task(args.task_name, model),
                verbose=False,
            )
        results_dict["clean_accs"] = (unpatched_acc, unpatched_control_acc)
    logging.info(
        f"Unpatched acc: {results_dict['clean_accs'][0] :.3f}; Unpatched Control acc: {results_dict['clean_accs'][1] :.3f})"
    )

    # Perform back-patching experiments across all settings
    cached_activations, cached_activations_l_baseline = None, None
    for repeat_processing_in_data_positions in [True]:  # , False]:
        for layer_window_size in [5, 3, 1]:

            logging.info(
                f"Running backpatching for {layer_window_size=}, {repeat_processing_in_data_positions=}"
            )

            current_cfg = (repeat_processing_in_data_positions, layer_window_size)
            current_cfg_accs, current_cfg_baseline_accs = results_dict.get(
                current_cfg,
                (
                    -1 * torch.ones((len(src_layer_range), len(dst_layer_range))),
                    -1 * torch.ones((len(src_layer_range), len(dst_layer_range))),
                ),
            )
            for i, src_layer in enumerate(src_layer_range):
                for j, dst_layer in enumerate(dst_layer_range):
                    if (
                        current_cfg_accs[i, j] != -1
                        and current_cfg_baseline_accs[i, j] != -1
                    ):
                        continue
                    if dst_layer >= src_layer:
                        # We don't want to backpatch to the same layer or a later layer
                        current_cfg_accs[i, j] = 0
                        current_cfg_baseline_accs[i, j] = 0
                    else:
                        if current_cfg_accs[i, j] == -1:
                            current_cfg_accs[i, j], cached_activations = backpatching(
                                model,
                                args,
                                vl_prompts,
                                src_layer,
                                dst_layer,
                                repeat_processing_in_data_positions=repeat_processing_in_data_positions,
                                layer_window_size=layer_window_size,
                                cached_activations=cached_activations,
                            )

                        if current_cfg_baseline_accs[i, j] == -1:
                            (
                                current_cfg_baseline_accs[i, j],
                                cached_activations_l_baseline,
                            ) = backpatching(
                                model,
                                args,
                                l_prompts,
                                src_layer,
                                dst_layer,
                                repeat_processing_in_data_positions=repeat_processing_in_data_positions,
                                layer_window_size=layer_window_size,
                                cached_activations=cached_activations_l_baseline,
                            )

                    results_dict[current_cfg] = (
                        current_cfg_accs,
                        current_cfg_baseline_accs,
                    )
                    torch.save(
                        (results_dict, src_layer_range, dst_layer_range), results_path
                    )

    # Choosing the best-found setting and doing a looped backpatching experiment
    best_cfg = max(
        [cfg for cfg in results_dict.keys() if len(cfg) == 2],
        key=lambda cfg: results_dict[cfg][0].max().item(),
    )
    best_repeat_processing_in_data_positions, best_layer_window_size = best_cfg
    print(best_cfg, results_dict[best_cfg][0].max())
    i, j = torch.where(results_dict[best_cfg][0] == results_dict[best_cfg][0].max())
    if len(i) > 1:
        # Multiple identical values exist
        i, j = i[0], j[0]
    best_src_layer, best_dst_layer = src_layer_range[i], dst_layer_range[j]
    logging.info(
        f"Best configuration: {best_cfg}, best src layer: {best_src_layer}, best dst layer: {best_dst_layer}: {results_dict[best_cfg][0][i, j].item()}"
    )
    logging.info("Running looped backpatching experiment")
    max_loop_count = 10
    if (
        best_repeat_processing_in_data_positions,
        best_layer_window_size,
        max_loop_count - 1,
    ) not in results_dict:
        looped_accs = looped_backpatching(
            model,
            args,
            vl_prompts,
            max_loop_count,
            best_src_layer,
            best_dst_layer,
            repeat_processing_in_data_positions=best_repeat_processing_in_data_positions,
            layer_window_size=best_layer_window_size,
        )
        logging.info(
            f"Looped backpatching accuracy for loops: {list(zip(range(max_loop_count), looped_accs))}"
        )

        for loop_count, looped_acc in zip(range(max_loop_count), looped_accs):
            results_dict[
                (
                    best_repeat_processing_in_data_positions,
                    best_layer_window_size,
                    loop_count,
                )
            ] = looped_acc
        torch.save((results_dict, src_layer_range, dst_layer_range), results_path)

    logging.info("Analysis complete")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
