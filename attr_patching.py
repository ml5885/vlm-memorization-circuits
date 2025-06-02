from functools import partial
import random
import torch
from general_utils import get_text_seq_len_and_image_seq_len
import transformer_lens as lens
from tqdm import tqdm
from typing import List, Tuple, Dict
from metrics import indirect_effect, logit_diff

from vision_language_prompts import VLPrompt


def node_attribution_patching_ig(
    model: lens.HookedVLTransformer,
    vl_prompts: List[VLPrompt],
    metric: str = "LD",
    ig_steps: int = 5,
    attributed_hook_names: List[str] = [
        "mlp.hook_post",
        "hook_z",
        "hook_cross_attn_states",
    ],
    verbose: bool = True,
    reset_hooks_pre_fwd: bool = True,
):
    model.requires_grad_(True)

    # Node filter function
    should_measure_hook_filter = partial(
        should_measure_hook,
        measurable_hooks=attributed_hook_names
        + ["hook_embed", "hook_cross_attn_states"],
    )

    attr_patching_scores = {}

    vl_prompts_iter = enumerate(tqdm(vl_prompts)) if verbose else enumerate(vl_prompts)
    for idx, vl_prompt in vl_prompts_iter:
        prompt_specific_attr_scores = {
            k: torch.zeros_like(v) for k, v in attr_patching_scores.items()
        }

        label = model.to_tokens(vl_prompt.answer, prepend_bos=False).view(-1, 1)
        cf_label = model.to_tokens(vl_prompt.cf_answer, prepend_bos=False).view(-1, 1)

        # Forward pass to get counterfactual cache
        _, cf_cache = model.run_with_cache(vl_prompt.cf_prompt, vl_prompt.cf_images)
        cf_cache = {
            k: v
            for (k, v) in cf_cache.cache_dict.items()
            if should_measure_hook_filter(k)
        }

        # Forward pass to get clean cache
        clean_logits_orig, clean_cache = model.run_with_cache(
            vl_prompt.prompt, vl_prompt.images
        )
        clean_cache = {
            k: v
            for (k, v) in clean_cache.cache_dict.items()
            if should_measure_hook_filter(k)
        }

        # Calculate the difference between every two parallel activation cache elements
        diff_cache = {}
        for k in clean_cache:
            diff_cache[k] = cf_cache[k] - clean_cache[k]

        def backward_hook_fn(grad, hook):
            # Gradient is multiplied with the activation difference (between counterfactual and clean prompts).
            if hook.name not in prompt_specific_attr_scores:
                prompt_specific_attr_scores[hook.name] = torch.zeros(
                    clean_cache[hook.name].shape[1:], device="cpu"
                )
            prompt_specific_attr_scores[hook.name] += (
                (diff_cache[hook.name] * grad).cpu().squeeze(0)
            )

        if reset_hooks_pre_fwd:
            model.reset_hooks()
        bwd_hooks = []
        for hook_point_name in model.hook_dict.keys():
            if should_measure_hook_filter(hook_point_name):
                bwd_hooks.append((hook_point_name, backward_hook_fn))

        cf_text_embeds, clean_text_embeds = (
            cf_cache["hook_embed"],
            clean_cache["hook_embed"],
        )
        if hasattr(model, "vl_strategy") and model.vl_strategy == "cross":
            cf_cross_attention_states, clean_cross_attention_states = (
                cf_cache["hook_cross_attn_states"],
                clean_cache["hook_cross_attn_states"],
            )

        def input_interpolation_hook(k: int):
            def hook_fn(activations, hook):
                if "cross_attn_states" in hook.name:
                    new_input = cf_cross_attention_states + (k / ig_steps) * (
                        clean_cross_attention_states - cf_cross_attention_states
                    )
                else:
                    new_input = cf_text_embeds + (k / ig_steps) * (
                        clean_text_embeds - cf_text_embeds
                    )
                new_input.requires_grad = True
                return new_input

            return hook_fn

        with torch.set_grad_enabled(True):
            for step in range(1, ig_steps + 1):
                with model.hooks(
                    fwd_hooks=[
                        ("hook_embed", input_interpolation_hook(step)),
                        (
                            "hook_cross_attn_states",
                            input_interpolation_hook(step),
                        ),
                    ],
                    bwd_hooks=bwd_hooks,
                ):
                    clean_logits = model(
                        vl_prompt.prompt, vl_prompt.images, return_type="logits"
                    )

                if metric == "IE":
                    metric_value = indirect_effect(
                        clean_logits_orig[:, -1]
                        .softmax(dim=-1)
                        .to(device=model.cfg.device),
                        clean_logits[:, -1].softmax(dim=-1).to(device=model.cfg.device),
                        label,
                        cf_label,
                    ).mean(dim=0)
                elif metric == "LD":
                    metric_value = logit_diff(
                        clean_logits[:, -1],
                        label.to(clean_logits.device),
                        cf_label.to(clean_logits.device),
                    ).mean(dim=0)
                elif metric == "L" or metric == "logit":
                    metric_value = (
                        clean_logits[:, -1]
                        .gather(1, label.to(clean_logits.device))
                        .mean(dim=0)
                    )
                else:
                    raise ValueError(f"Unknown metric {metric}")

                metric_value.backward()
                model.zero_grad()

        del diff_cache

        # Accumulate the attribution scores
        for k in prompt_specific_attr_scores:
            if k not in attr_patching_scores:
                attr_patching_scores[k] = torch.zeros_like(
                    prompt_specific_attr_scores[k]
                )
            attr_patching_scores[k] += prompt_specific_attr_scores[k].abs() / ig_steps

    model.reset_hooks()
    model.requires_grad_(False)
    torch.cuda.empty_cache()

    attr_patching_scores = {
        k: v / len(vl_prompts) for (k, v) in attr_patching_scores.items()
    }

    return attr_patching_scores


def node_attribution_patching(
    model: lens.HookedVLTransformer,
    vl_prompts: List[VLPrompt],
    metric: str = "LD",
    attributed_hook_names: List[str] = [
        "mlp.hook_post",
        "hook_z",
        "hook_cross_attn_states",
    ],
    verbose: bool = True,
):
    model.requires_grad_(True)

    # Node filter function
    should_measure_hook_filter = partial(
        should_measure_hook, measurable_hooks=attributed_hook_names
    )

    # Choose a random counterfactual prompt for each prompt, if not given
    attr_patching_scores = {}

    vl_prompts_iter = enumerate(tqdm(vl_prompts)) if verbose else enumerate(vl_prompts)
    for idx, vl_prompt in vl_prompts_iter:
        label = model.to_tokens(vl_prompt.answer, prepend_bos=False).view(-1, 1)
        cf_label = model.to_tokens(vl_prompt.cf_answer, prepend_bos=False).view(-1, 1)

        # Forward pass to get counterfactual cache
        _, cf_cache = model.run_with_cache(vl_prompt.cf_prompt, vl_prompt.cf_images)
        cf_cache = {
            k: v
            for (k, v) in cf_cache.cache_dict.items()
            if should_measure_hook_filter(k)
        }

        # Forward pass to get clean cache
        clean_logits_orig, clean_cache = model.run_with_cache(
            vl_prompt.prompt, vl_prompt.images
        )
        clean_cache = {
            k: v
            for (k, v) in clean_cache.cache_dict.items()
            if should_measure_hook_filter(k)
        }

        # Calculate the difference between every two parallel activation cache elements
        diff_cache = {}
        for k in clean_cache:
            diff_cache[k] = cf_cache[k] - clean_cache[k]

        def backward_hook_fn(grad, hook):
            # Gradient is multiplied with the activation difference (between counterfactual and clean prompts).
            if hook.name not in attr_patching_scores:
                attr_patching_scores[hook.name] = torch.zeros(
                    clean_cache[hook.name].shape[1:], device="cpu"
                )
            attr_patching_scores[hook.name] += (diff_cache[hook.name] * grad).cpu()

        model.reset_hooks()
        model.add_hook(
            name=should_measure_hook_filter,
            hook=partial(backward_hook_fn),
            dir="bwd",
        )

        with torch.set_grad_enabled(True):
            clean_logits = model(
                vl_prompt.prompt, vl_prompt.images, return_type="logits"
            )
            if metric == "IE":
                metric_value = indirect_effect(
                    clean_logits_orig[:, -1]
                    .softmax(dim=-1)
                    .to(device=model.cfg.device),
                    clean_logits[:, -1].softmax(dim=-1).to(device=model.cfg.device),
                    label,
                    cf_label,
                ).mean(dim=0)
            elif metric == "LD":
                metric_value = logit_diff(
                    clean_logits[:, -1],
                    label.to(clean_logits.device),
                    cf_label.to(clean_logits.device),
                ).mean(dim=0)
            elif metric == "L" or metric == "logit":
                metric_value = (
                    clean_logits[:, -1]
                    .gather(1, label.to(clean_logits.device))
                    .mean(dim=0)
                )
            else:
                raise ValueError(f"Unknown metric {metric}")

            metric_value.backward()
            model.zero_grad()

        del diff_cache

    model.reset_hooks()
    model.requires_grad_(False)
    torch.cuda.empty_cache()

    attr_patching_scores = {
        k: v / len(vl_prompts) for (k, v) in attr_patching_scores.items()
    }

    return attr_patching_scores


def should_measure_hook(hook_name, measurable_hooks):
    if any([h in hook_name for h in measurable_hooks]):
        return True
