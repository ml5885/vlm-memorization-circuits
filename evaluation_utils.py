import transformer_lens as lens
import torch
from functools import partial
from component import Component
from typing import List, Optional, Tuple, Dict
from tqdm import tqdm
from PIL.Image import Image
from transformer_lens.hook_points import HookPoint
from metrics import logit_diff
from general_utils import get_hook_dim
from vision_language_prompts import VLPrompt, vlp_collate_fn


#
# Node-based faithfulness analysis functions, per position
#
def circuit_faithfulness(
    model: lens.HookedVLTransformer,
    circuit_components: List[Component],
    vl_prompts: List[VLPrompt],
    metric: str = "LD",
    limited_labels: Optional[torch.Tensor] = None,
    batch_size: int = 1,
    verbose: bool = True,
):
    """
    Calculate the faithfulness of the circuit w.r.t to the entire model.
    Non-circuit components are ablated using their activations for counterfactual (corrupt) prompts.
    The faithfulness is normalized using two baselines - a "good" one where no components are mean ablated, and a "bad" one where all components are mean ablated.
    """
    # NOTE - This function doesn't support mean ablation yet because I didn't need it. It can be added rather simply.

    model.cfg.ungroup_grouped_query_attention = True

    scores = torch.zeros(len(vl_prompts))
    dataloader = torch.utils.data.DataLoader(
        vl_prompts, batch_size=batch_size, shuffle=False, collate_fn=vlp_collate_fn
    )
    dataloader = tqdm(dataloader) if verbose else dataloader

    circuit_ablation_hooks = (
        None  # Used for caching of hook list to avoid re-calculating each prompt
    )
    for idx, batch in enumerate(dataloader):
        # Later each prompt batch is wrapped with a list() call. This is used to clone the list,
        # because some processors modify the prompt input list in place.
        prompts_batch = batch["prompt"]
        images_batch = batch["images"]
        cf_prompts_batch = batch["cf_prompt"]
        cf_images_batch = batch["cf_images"]
        answers_batch = model.to_tokens(batch["answer"], prepend_bos=False).view(-1, 1)
        cf_answers_batch = model.to_tokens(batch["cf_answer"], prepend_bos=False).view(
            -1, 1
        )
        assert (
            model.to_tokens(vl_prompts[idx].answer).item() == answers_batch.item()
            and model.to_tokens(vl_prompts[idx].cf_answer).item()
            == cf_answers_batch.item()
        ), "SANITY, TO REMOVE"

        # For the "good" baseline, no components are mean ablated, so its a simple forward pass
        good_baseline_logits = model(list(prompts_batch), images_batch)[:, -1]

        # Create the ablation cache
        _, cf_cache = model.run_with_cache(
            list(cf_prompts_batch),
            cf_images_batch,
            return_type="logits",
        )
        # Convert the clean_cache to a dict with components as keys
        ablation_cache = {}
        for layer in range(model.cfg.n_layers):
            for head_idx in [None] + list(range(model.cfg.n_heads)):
                hook_name = "mlp_post" if head_idx is None else "z"
                if head_idx is None:
                    ablation_cache[Component(hook_name, layer)] = cf_cache[
                        lens.utils.get_act_name(hook_name, layer)
                    ]
                else:
                    ablation_cache[Component(hook_name, layer, head=head_idx)] = (
                        cf_cache[lens.utils.get_act_name(hook_name, layer)][
                            :, :, head_idx
                        ]
                    )
        del cf_cache

        # For the "bad" baseline, all components are ablated
        bad_baseline_logits = run_and_ablate(
            model, [], list(prompts_batch), images_batch, ablation_cache=ablation_cache
        )[0][:, -1].to(good_baseline_logits.device)

        # For the ablated model, all non-circuit components are ablated
        ablated_logits, circuit_ablation_hooks = run_and_ablate(
            model,
            circuit_components,
            list(prompts_batch),
            images_batch,
            ablation_cache=ablation_cache,
            ablation_hooks=circuit_ablation_hooks,
        )
        ablated_logits = ablated_logits[:, -1].to(good_baseline_logits.device)

        answers_batch = answers_batch.to(good_baseline_logits.device)
        cf_answers_batch = cf_answers_batch.to(good_baseline_logits.device)

        if metric == "logit":
            good_baseline_correct_logits = good_baseline_logits.gather(1, answers_batch)
            ablated_correct_logits = ablated_logits.gather(1, answers_batch)
            bad_baseline_correct_logits = bad_baseline_logits.gather(1, answers_batch)
            batch_scores = (ablated_correct_logits - bad_baseline_correct_logits) / (
                good_baseline_correct_logits - bad_baseline_correct_logits
            )
        elif metric == "prob":
            good_baseline_correct_probs = good_baseline_logits.softmax(dim=-1).gather(
                1, answers_batch
            )
            ablated_correct_probs = ablated_logits.softmax(dim=-1).gather(
                1, answers_batch
            )
            bad_baseline_correct_probs = bad_baseline_logits.softmax(dim=-1).gather(
                1, answers_batch
            )
            batch_scores = (ablated_correct_probs - bad_baseline_correct_probs) / (
                good_baseline_correct_probs - bad_baseline_correct_probs
            )
        elif metric == "LD":
            good_ld = logit_diff(
                good_baseline_logits,
                answers_batch,
                cf_answers_batch,
            )
            ablated_ld = logit_diff(
                ablated_logits,
                answers_batch,
                cf_answers_batch,
            )
            bad_baseline_ld = logit_diff(
                bad_baseline_logits,
                answers_batch,
                cf_answers_batch,
            )
            batch_scores = (ablated_ld - bad_baseline_ld) / (good_ld - bad_baseline_ld)
        elif metric == "acc":
            if limited_labels is not None:
                limited_labels = limited_labels.to(good_baseline_logits.device).view(-1)
                good_baseline_pred = limited_labels[
                    good_baseline_logits[:, limited_labels].argmax(dim=-1)
                ]
                bad_baseline_pred = limited_labels[
                    bad_baseline_logits[:, limited_labels].argmax(dim=-1)
                ]
                ablated_pred = limited_labels[
                    ablated_logits[:, limited_labels].argmax(dim=-1)
                ]
            else:
                good_baseline_pred = good_baseline_logits.argmax(dim=-1)
                bad_baseline_pred = bad_baseline_logits.argmax(dim=-1)
                ablated_pred = ablated_logits.argmax(dim=-1)
            good_baseline_acc = (good_baseline_pred == answers_batch).float()
            ablated_acc = (ablated_pred == answers_batch).float()
            bad_baseline_acc = (bad_baseline_pred == answers_batch).float()
            batch_scores = (ablated_acc - bad_baseline_acc) / (
                good_baseline_acc - bad_baseline_acc
            )
        elif metric == "ce":
            ce = torch.nn.CrossEntropyLoss()
            good_baseline_ce = -ce(good_baseline_logits, answers_batch)
            ablated_ce = -ce(ablated_logits, answers_batch)
            bad_baseline_ce = -ce(bad_baseline_logits, answers_batch)
            batch_scores = (ablated_ce - bad_baseline_ce) / (
                good_baseline_ce - bad_baseline_ce
            )
        else:
            raise ValueError(f"Unknown metric {metric}")

        scores[idx * batch_size : (idx + 1) * batch_size] = batch_scores

    return scores.mean()


def run_and_ablate(
    model: lens.HookedVLTransformer,
    components: List[Component],
    prompts_batch: List[str],
    images_batch: List[Image],
    ablation_cache: Dict[Component, torch.Tensor],
    ablation_hooks: Optional[List[HookPoint]] = None,
    reverse_ablation: bool = True,
):
    def hook_ablate_from_ablation_cache(value, hook, hooked_comps=None):
        # Each pos might have different neurons to ablate
        for hooked_comp in hooked_comps:
            assert (
                hooked_comp.pos is not None or len(hooked_comps) == 1
            ), f"If None is passed as a position, there must be a single component and not {len(hooked_comps)}"

            if hooked_comp.pos is None:
                pos = slice(None)
            elif isinstance(hooked_comp.pos, list) and len(hooked_comp.pos) == 2:
                pos = slice(*hooked_comp.pos)
            else:
                pos = hooked_comp.pos

            pos_and_neurons_agnostic_comp = Component(
                hooked_comp.hook_name,
                layer=hooked_comp.layer,
                head=hooked_comp.head_idx,
            )
            if hooked_comp.head_idx is None and hooked_comp.neuron_indices is None:
                # An entire MLP at the current position
                value[:, pos] = ablation_cache[pos_and_neurons_agnostic_comp][
                    :, pos, :
                ].to(value.device, value.dtype)
            elif (
                hooked_comp.head_idx is None and hooked_comp.neuron_indices is not None
            ):
                # MLP with specific neurons at the current position
                value[:, pos, hooked_comp.neuron_indices] = ablation_cache[
                    pos_and_neurons_agnostic_comp
                ][:, pos, hooked_comp.neuron_indices].to(value.device, value.dtype)
            elif (
                hooked_comp.head_idx is not None and hooked_comp.neuron_indices is None
            ):
                # Attention head with all neurons (maybe in a specific position)
                value[:, pos, hooked_comp.head_idx, :] = ablation_cache[
                    pos_and_neurons_agnostic_comp
                ][:, pos].to(value.device, value.dtype)
            else:
                # Attention head with specific neurons
                raise NotImplementedError()
        return value

    # Build the hooks for the ablation
    assert reverse_ablation  # All components EXCEPT the list of given components are ablated

    # For each layer, we check for each of its components (the full MLP and each attention head) if its in the circuit.
    # If the full version is in the circuit, we don't do anything (it should be mean ablated). Otherwise, we check if
    # specific neurons are in the circuit, in which case we mean ablate the rest of the neurons. If the component is not
    # in the circuit at all, we mean ablate the entire component.
    if ablation_hooks is None:
        ablation_hooks = []
        for layer in range(model.cfg.n_layers):
            full_mlp_comp = Component("mlp_post", layer=layer)
            if full_mlp_comp not in components:
                # The entire MLP isnt part of the circuit; Some specific neurons / some specific positions in it might still be
                mlp_layer_comps = [
                    c
                    for c in components
                    if c.layer == layer
                    and c.head_idx is None
                    and c.valid_hook_name()
                    == lens.utils.get_act_name("mlp_post", layer=c.layer)
                ]
                mlp_hooked_comps = []
                if len(mlp_layer_comps) == 0:
                    # No component with specific MLP neurons / specific position is a part of the circuit -
                    # The entire MLP should be mean ablated
                    mlp_hooked_comps.append(full_mlp_comp)
                else:
                    circuit_positions = set([c.pos for c in mlp_layer_comps])
                    seq_len = ablation_cache[list(ablation_cache.keys())[0]].shape[1]
                    for pos in range(seq_len):
                        if pos in circuit_positions:
                            # There are specific neurons in the circuit in this position.
                            # All other neurons in this position are ablated.
                            mlp_layer_pos_comps = [
                                c for c in mlp_layer_comps if c.pos == pos
                            ]
                            layer_neuron_indices = [
                                c.neuron_indices for c in mlp_layer_pos_comps
                            ]
                            circuit_neuron_indices = set(
                                sum(layer_neuron_indices, start=())
                            )
                            mean_ablated_neuron_indices = list(
                                set(range(get_hook_dim(model, full_mlp_comp.hook_name)))
                                - circuit_neuron_indices
                            )
                            mlp_hooked_comps.append(
                                Component(
                                    full_mlp_comp.hook_name,
                                    layer=layer,
                                    neurons=mean_ablated_neuron_indices,
                                    position=pos,
                                )
                            )
                        else:
                            # There aren't any circuit MLP comps in this position
                            # The entire position is ablated
                            mlp_hooked_comps.append(
                                Component(
                                    full_mlp_comp.hook_name, layer=layer, position=pos
                                )
                            )

                ablation_hooks.append(
                    (
                        mlp_hooked_comps[0].valid_hook_name(),
                        partial(
                            hook_ablate_from_ablation_cache,
                            hooked_comps=mlp_hooked_comps,
                        ),
                    )
                )

            for head in range(model.cfg.n_heads):
                full_head_comp = Component("z", layer=layer, head=head)
                if full_head_comp not in components:
                    head_layer_comps = [
                        c
                        for c in components
                        if c.layer == layer
                        and c.head_idx == head
                        and c.valid_hook_name()
                        == lens.utils.get_act_name("z", layer=c.layer)
                    ]
                    seq_len = ablation_cache[list(ablation_cache.keys())[0]].shape[1]
                    if len(head_layer_comps) == 0:
                        # No head with specific neurons / specific positions is part of the circuit;
                        # The entire head should be mean ablated
                        head_hooked_comps = [full_head_comp]
                    else:
                        head_hooked_comps = []
                        circuit_positions = set([c.pos for c in head_layer_comps])
                        positions_to_ablate = list(
                            set(range(seq_len)) - circuit_positions
                        )
                        for pos in positions_to_ablate:
                            # The attention head in this position in not part of the circuit; Mean ablate it
                            head_hooked_comps.append(
                                Component(
                                    full_head_comp.hook_name,
                                    head=head,
                                    layer=layer,
                                    position=pos,
                                )
                            )

                    if len(head_hooked_comps) > 0:
                        # If there are any positions in the head that are not part of the circuit, add the hook
                        ablation_hooks.append(
                            (
                                head_hooked_comps[0].valid_hook_name(),
                                partial(
                                    hook_ablate_from_ablation_cache,
                                    hooked_comps=head_hooked_comps,
                                ),
                            )
                        )
    else:
        ablation_hooks = [
            (h_name, partial(hook_ablate_from_ablation_cache, **h_func.keywords))
            for (h_name, h_func) in ablation_hooks
        ]

    # Get the logits after ablating the components
    ablated_logits = model.run_with_hooks(
        prompts_batch,
        images_batch,
        fwd_hooks=ablation_hooks,
    )
    return ablated_logits, ablation_hooks


#
# General evaluation functions
#
def model_accuracy(
    model: lens.HookedVLTransformer,
    vl_prompts: List[VLPrompt],
    batch_size: int = 8,
    limited_labels: Optional[torch.Tensor] = None,
    hooks: List[Tuple[str, HookPoint]] = None,
    report_both_acc_and_llacc=False,
    report_correct_prompts=False,
    verbose: bool = True,
) -> float:
    """
    Measure the model's accuracy on a set of prompts.

    Args:
        model (lens.HookedVLTransformer): The model to measure the accuracy of.
        vl_prompts (List[VLPrompt]): The prompts to measure the accuracy on.
        batch_size (int): The batch size to use while calculating the accuracy. (Default is 8)
        limited_labels (torch.Tensor): If not None, only measure the accuracy based on the logits of the labels in this tensor. (Default is None)
        hooks (list[(str, Callable)]): A list of hooks to use while calculating the accuracy on the prompts. If None, no hooks are used. (Default is None)
        report_both_acc_and_llacc (bool): If True, report both the accuracy and the log-likelihood accuracy. Else, only one is reported. (Default is False)
        verbose (bool): If True, print the wrong prompts as well as a progress bar. (Default is True)
    """
    if report_both_acc_and_llacc:
        assert limited_labels is not None

    correct = torch.zeros(len(vl_prompts), dtype=torch.bool)
    correct_ll = torch.zeros(len(vl_prompts), dtype=torch.bool)
    try:
        if hooks is not None:
            for h in hooks:
                model.add_hook(h[0], h[1])

        dataloader = torch.utils.data.DataLoader(
            vl_prompts, batch_size=batch_size, shuffle=False, collate_fn=vlp_collate_fn
        )
        dataloader = tqdm(dataloader) if verbose else dataloader
        for idx, batch in enumerate(dataloader):
            prompts_batch = batch["prompt"]
            images_batch = batch["images"]
            answers_batch = model.to_tokens(batch["answer"], prepend_bos=False).view(-1)

            logits = model(prompts_batch, images_batch)
            if limited_labels is not None:
                limited_labels = limited_labels.to(logits.device).view(-1)
                preds_ll = logits[:, -1, limited_labels].argmax(-1)
                pred_labels = limited_labels[preds_ll]
                correct_ll[idx * batch_size : (idx + 1) * batch_size] = (
                    pred_labels.cpu() == answers_batch.cpu()
                )
            preds = logits[:, -1, :].argmax(-1)
            correct[idx * batch_size : (idx + 1) * batch_size] = (
                preds.cpu() == answers_batch.cpu()
            )

    finally:
        model.remove_all_hook_fns()

    if report_both_acc_and_llacc and report_correct_prompts:
        return correct, correct_ll
    elif report_both_acc_and_llacc:
        return correct.float().mean().item(), correct_ll.float().mean().item()
    elif report_correct_prompts:
        return correct if limited_labels is None else correct_ll
    else:
        return (
            correct.float().mean().item()
            if limited_labels is None
            else correct_ll.float().mean().item()
        )


# HACK UGLY FUNCTION FOR QUICK TESTING
# Used to handle the requirement of splitting factual-recall prompts per template
def model_accuracy_for_factual_recall(
    model: lens.HookedVLTransformer,
    vl_prompts: List[VLPrompt],
    batch_size: int = 8,
    hooks: List[Tuple[str, HookPoint]] = None,
    report_both_acc_and_llacc=False,
    report_correct_prompts=False,
    verbose: bool = True,
) -> float:
    import json
    import re

    qa_json = json.load(open("./data/factual_recall/qa_raw.json", "r"))
    prompts_split_by_templates = {
        question_template: [] for question_template in qa_json.keys()
    }
    limited_labels_by_template = {
        question_template: set() for question_template in qa_json.keys()
    }
    for vl_prompt in vl_prompts:
        prompt_question = re.search(
            r"((?:What|Which).*single word\.)", vl_prompt.prompt
        ).groups()[0]
        for question_template in qa_json.keys():
            if prompt_question in question_template:
                prompts_split_by_templates[question_template].append(vl_prompt)
                limited_labels_by_template[question_template].add(vl_prompt.answer)
                break

    results = []
    for template in prompts_split_by_templates.keys():
        prompts = prompts_split_by_templates[template]
        ll = model.to_tokens(
            list(limited_labels_by_template[template]), prepend_bos=False
        )
        results += model_accuracy(
            model,
            prompts,
            batch_size,
            ll,
            hooks,
            report_correct_prompts=True,
            verbose=verbose,
        )
    return torch.stack(results).float().mean().item()
