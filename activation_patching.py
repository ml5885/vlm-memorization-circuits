import sys

sys.path.append("third_party/TransformerLens")
import torch
import random
import transformer_lens as lens
from functools import partial
from metrics import indirect_effect, logit_diff
from typing import Callable, List, Optional, Tuple
from vision_language_prompts import VLPrompt
from component import Component


def activation_patching_experiment(
    model: lens.HookedVLTransformer,
    vl_prompts: List[VLPrompt],
    metric: str = "LD",
    token_pos: Optional[int] = -1,
    batch_size: int = 32,
):
    """
    Performs an activation patching experiment.
    Each prompt is passed through the model, and at each layer, the activations of each model component
    (MLP and attention head) are patched with the activations from another prompt.

    Args:
        model (lens.HookedTransformer): The model to patch.
        vl_prompts (List[VLPrompt]): A list of Vision-Language prompts to use for the experiment.
        metric (str): The metric to use for measuring the effect of the patching. Defaults to logit difference (LD).
        token_pos (int): The token position to patch. Defaults to -1 (last token). If None, all positions are patched.
        batch_size (int): The batch size to load vl_prompts in.
    Returns:
        torch.Tensor (n_prompts, n_layers, n_components): The metric results for each prompt, layer and component.
    """
    # Initialize the results tensor
    n_prompts = len(vl_prompts)
    n_layers = model.cfg.n_layers
    n_components = model.cfg.n_heads + 1  # Attention heads + MLP output
    patching_results = torch.zeros(
        n_prompts, n_layers, n_components, dtype=torch.float32
    )

    # Perform the patching experiment
    for layer in range(n_layers):
        components = [
            Component("z", layer=layer, head=head) for head in range(model.cfg.n_heads)
        ] + [Component("mlp_out", layer=layer)]
        for comp_idx, component in enumerate(components):
            patching_results[:, layer, comp_idx] = _single_activation_patching(
                model,
                vl_prompts,
                component,
                metric=metric,
                token_pos=token_pos,
                batch_size=batch_size,
            )

    return patching_results


def _single_activation_patching(
    model: lens.HookedVLTransformer,
    vl_prompts: List[VLPrompt],
    hook_component: Component,
    metric: str = "LD",
    token_pos: int = -1,
    hook_func_overload: Callable = None,
    batch_size: int = 32,
):
    """
    Perform an activation patching experiment on a single component.
    """
    patching_results = torch.zeros(len(vl_prompts), dtype=torch.float32)

    clean_prompts = [p.prompt for p in vl_prompts]
    clean_images = [p.images for p in vl_prompts]
    cf_prompts = [p.cf_prompt for p in vl_prompts]
    cf_images = [p.cf_images for p in vl_prompts]
    clean_labels = model.to_tokens([p.answer for p in vl_prompts], prepend_bos=False)
    cf_labels = model.to_tokens([p.cf_answer for p in vl_prompts], prepend_bos=False)

    for batch_start in range(0, len(vl_prompts), batch_size):
        batch_end = min(batch_start + batch_size, len(vl_prompts))

        batch_prompts, batch_images, batch_labels = (
            clean_prompts[batch_start:batch_end],
            clean_images[batch_start:batch_end],
            clean_labels[batch_start:batch_end],
        )
        batch_cf_prompts, batch_cf_images, batch_cf_labels = (
            cf_prompts[batch_start:batch_end],
            cf_images[batch_start:batch_end],
            cf_labels[batch_start:batch_end],
        )

        # Run both prompt batches to get the logits and activation cache
        clean_logits = model(batch_prompts, batch_images, return_type="logits")
        cf_logits, cf_cache = model.run_with_cache(
            batch_cf_prompts,
            batch_cf_images,
            return_type="logits",
        )

        # Define a default hooking function, which works for patching MLP / full attention output activations
        def default_patching_hook(value, hook, component, token_pos):
            """
            A hook that works for some of the more common modules (MLP outputs, Attention outputs, Attention head outputs).
            """
            if token_pos is None:
                token_pos = slice(None)

            if len(value.shape) == 3:
                # Entire block output (MLP_out or Attn_out), shape (batch pos d_model)
                value[:, token_pos, :] = cf_cache[hook.name][:, token_pos, :]
            else:
                # Specific head output, shape (batch pos head_idx d_head)
                assert (
                    component.is_attn
                ), "If you want to hook a non-head component this way, pass your own hook function"
                value[:, token_pos, component.head_idx, :] = cf_cache[hook.name][
                    :, token_pos, component.head_idx, :
                ]

            return value

        hook_func = (
            default_patching_hook if hook_func_overload is None else hook_func_overload
        )

        # Patch the component to measure the effect metric
        hook_fn_with_cache = partial(
            hook_func, component=hook_component, token_pos=token_pos
        )
        patched_logits = model.run_with_hooks(
            batch_prompts,
            batch_images,
            fwd_hooks=[
                (
                    hook_component.valid_hook_name(),
                    hook_fn_with_cache,
                )
            ],
            return_type="logits",
        )
        if metric == "IE":
            # Indirect effect
            patching_results[batch_start:batch_end] = indirect_effect(
                clean_logits[:, -1].softmax(dim=-1).to(model.cfg.device),
                patched_logits[:, -1].softmax(dim=-1).to(model.cfg.device),
                batch_labels.to(model.cfg.device),
                batch_cf_labels.to(model.cfg.device),
            )
        elif metric == "IE-Logits":
            # Indirect effect on logits instead of probs
            patching_results[batch_start:batch_end] = indirect_effect(
                clean_logits[:, -1].to(model.cfg.device),
                patched_logits[:, -1].to(model.cfg.device),
                batch_labels.to(model.cfg.device),
                batch_cf_labels.to(model.cfg.device),
            )
        elif metric == "LD":
            # Logit diff
            patching_results[batch_start:batch_end] = logit_diff(
                patched_logits[:, -1].to(model.cfg.device),
                batch_labels.to(model.cfg.device),
                batch_cf_labels.to(model.cfg.device),
            )
        else:
            raise ValueError(f"Unknown metric {metric}")

    print(hook_component, patching_results)
    return patching_results
