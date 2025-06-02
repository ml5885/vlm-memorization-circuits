import torch
from collections import defaultdict
from component import Component


class PositionMapping:
    """
    A class to store the mapping between positions in prompts for the same task across two modalities.
    """

    def __init__(self, mapping=None):
        self.l_to_vl_positions = defaultdict(set)
        self.vl_to_l_positions = defaultdict(set)
        if mapping is not None:
            for l_pos, vl_pos in mapping:
                self.add(l_pos, vl_pos)

    def add(self, l_pos, vl_pos):
        self.l_to_vl_positions[l_pos].add(vl_pos)
        self.vl_to_l_positions[vl_pos].add(l_pos)

    def get_vl_positions(self, l_pos):
        """Get all vl positions associated with a l position."""
        return self.l_to_vl_positions.get(l_pos, set())

    def get_l_positions(self, vl_pos):
        """Get all l positions associated with a vl position."""
        return self.vl_to_l_positions.get(vl_pos, set())

    def is_equivalent(self, l_pos, vl_pos):
        first = vl_pos in self.l_to_vl_positions[l_pos]
        second = (
            l_pos in self.vl_to_l_positions[vl_pos]
        )  # Only for sanity, the mapping should always be bi-directional
        assert (first and second) or (not first and not second), "Sanity check failed"
        return first

    def map_attn_pattern_rows(
        self, attn_pattern_rows: torch.Tensor, l_to_vl: bool = True
    ):
        """
        Convert the attention pattern row from one modality to another using the position mapping.
        The attention given a position is split equally between all mapped positions.

        For example, when the mapping maps l positions 0->0; 1,2->1,2,3,4; 3->5; and the attn pattern
        shows attention to positions 0,1,2 with scores [0.52, 0.2, 0.28, 0], the mapped attn pattern
        will be [0.52, 0.12, 0.12, 0.12, 0.12, 0].

        Args:
            attn_pattern_row (torch.tensor): Attention pattern row to convert, should be of shape (bs, seq_len).
            l_to_vl: If True, convert from l to vl. Otherwise, convert from vl to l.
        Returns:
            A torch.tensor shaped (bs, other_seq_len).
        """
        if l_to_vl:
            other_seq_len = len(self.vl_to_l_positions)
            mapping_func = self.get_vl_positions
        else:
            other_seq_len = len(self.l_to_vl_positions)
            mapping_func = self.get_l_positions

        new_attn_pattern_row = torch.zeros(
            (attn_pattern_rows.shape[0], other_seq_len), device=attn_pattern_rows.device
        )

        for pos in range(attn_pattern_rows.shape[1]):
            mapped_positions = list(mapping_func(pos))
            if len(mapped_positions) > 0:
                new_attn_pattern_row[:, mapped_positions] += attn_pattern_rows[
                    :, pos
                ].unsqueeze(1) / len(mapped_positions)

        return new_attn_pattern_row

    def print_mapping(self, l_prompt, vl_prompt, model, src="l"):
        src_prompt, tgt_prompt = (
            (l_prompt, vl_prompt) if src == "l" else (vl_prompt, l_prompt)
        )
        src_to_tgt = self.l_to_vl_positions if src == "l" else self.vl_to_l_positions
        get_tgt_positions = (
            self.get_vl_positions if src == "l" else self.get_l_positions
        )

        src_prompt_tokens = [
            model.to_single_str_token(t.item())
            for t in (model.to_tokens(src_prompt.prompt, src_prompt.images)).view(-1)
        ]
        tgt_prompt_tokens = [
            model.to_single_str_token(t.item())
            for t in (model.to_tokens(tgt_prompt.prompt, tgt_prompt.images)).view(-1)
        ]

        for src_pos in src_to_tgt.keys():
            src_token = src_prompt_tokens[src_pos]
            tgt_positions = get_tgt_positions(src_pos)
            if len(tgt_positions) > 0:
                for tgt_pos in tgt_positions:
                    tgt_token = tgt_prompt_tokens[tgt_pos]
                    print(
                        f"({src_pos:2d}, {tgt_pos:2d}): {src_token!r} -> {tgt_token!r}"
                    )
            else:
                print(f"({src_pos:2d}, ): {src_token!r} -> UNMAPPED!")

    def assert_full_mapping(self, l_prompt, vl_prompt, model):
        l_tokens = [
            model.to_single_str_token(t.item())
            for t in (model.to_tokens(l_prompt.prompt, l_prompt.images)).view(-1)
        ]
        vl_tokens = [
            model.to_single_str_token(t.item())
            for t in (model.to_tokens(vl_prompt.prompt, vl_prompt.images)).view(-1)
        ]

        l_seq_len, vl_seq_len = len(l_tokens), len(vl_tokens)
        assert l_seq_len == len(
            self.l_to_vl_positions
        ), f"Length mismatch: {l_seq_len} != {len(self.l_to_vl_positions)}"
        assert vl_seq_len == len(
            self.vl_to_l_positions
        ), f"Length mismatch: {vl_seq_len} != {len(self.vl_to_l_positions)}"

    def remove_l_pos(self, l_pos):
        if l_pos in self.l_to_vl_positions:
            # Remove the key from all associated values in value_to_keys
            for vl_pos in self.l_to_vl_positions[l_pos]:
                self.vl_to_l_positions[vl_pos].discard(l_pos)
            # Delete the key from key_to_values
            del self.l_to_vl_positions[l_pos]

    def remove_vl_pos(self, vl_pos):
        if vl_pos in self.vl_to_l_positions:
            # Remove the value from all associated keys in key_to_values
            for l_pos in self.vl_to_l_positions[vl_pos]:
                self.l_to_vl_positions[l_pos].discard(vl_pos)
            # Delete the value from value_to_keys
            del self.vl_to_l_positions[vl_pos]


# Length in tokens of the chat template prefixes (i.e. before any prompt-specific / modality-specific tokens appear)
QWEN_PREFIX_LEN = 14
PIXTRAL_PREFIX_LEN = 2
GEMMA_PREFIX_LEN = 4


def convert_components_modality(components, pos_mapping, l_to_vl=True):
    """
    Convert the components from one modality to another using the position mapping.
    For example, when the mapping maps l position 4 to vl positions 5,6,7, a component C at position 4
    in l will be converted to components (C, 5); (C, 6); (C, 7) in vl.

    Args:
        components: List of components to convert.
        pos_mapping: PositionMapping object.
        l_to_vl: If True, convert from l to vl. Otherwise, convert from vl to l.
    Returns:
        A set of converted components.
    """
    new_components = set()
    if len(components) == 0:
        return new_components
    elif isinstance(components[0], Component):
        # "Components" are Component objects
        # In this case, each component is converted to multiple components based on the position mapping.
        # This functionallity should be used for conversion of specific attention heads (that don't support multiple neurons)
        for comp in set(components):
            if l_to_vl:
                new_positions = pos_mapping.get_vl_positions(comp.pos)
            else:
                new_positions = pos_mapping.get_l_positions(comp.pos)
            for new_pos in new_positions:
                new_components.add(
                    Component(
                        comp.hook_name,
                        layer=comp.layer,
                        head=comp.head_idx,
                        position=new_pos,
                        neurons=comp.neuron_indices,
                    )
                )
    elif isinstance(components[0], (tuple, list)):
        # "Components" are tuples of (layer, pos, neuron)
        # In this case, we convert the position of each tuple to the corresponding dst positions;
        # This functionallity should be used for conversion of MLP neurons.
        for layer, pos, neuron in components:
            if l_to_vl:
                new_positions = pos_mapping.get_vl_positions(pos)
            else:
                new_positions = pos_mapping.get_l_positions(pos)

            for new_pos in new_positions:
                new_components.add((layer, new_pos, neuron))
    else:
        raise ValueError("Unexpected components type: ", type(components[0]))

    return new_components


def get_image_positions(model_name, task_name, return_range=False):
    """
    Return the start and end positions of the image in the prompt for a given model and task.
    These positions are constant and based on the otherwise-defined templates.

    Args:
        model_name (str): Name of the model.
        task_name (str): Name of the task.
        return_range (bool): If True, return a list of values in the range [start, end). Otherwise, returns a tuple (start, end).
    """
    if "qwen" in model_name.lower():
        limits = {
            "counting": [15, 15 + 81],
            "arithmetic": [15, 15 + 36],
            "color_ordering": [15, 15 + 81],
            "sentiment_analysis": [15, 15 + 81],
            "factual_recall": [15, 15 + 81],
            "vqa": [15, 15 + 81],
        }[task_name]
    elif "pixtral" in model_name.lower():
        limits = {
            "counting": [2, 2 + 271],
            "arithmetic": [2, 2 + 114],
            "color_ordering": [2, 2 + 271],
            "sentiment_analysis": [2, 2 + 271],
            "factual_recall": [2, 2 + 271],
            "vqa": [2, 2 + 271],
        }[task_name]
    elif "gemma" in model_name.lower():
        limits = {
            "counting": [4, 4 + 257],
            "arithmetic": [4, 4 + 257],
            "color_ordering": [4, 4 + 257],
            "sentiment_analysis": [4, 4 + 257],
            "factual_recall": [4, 4 + 257],
            "vqa": [4, 4 + 257],
        }[task_name]
    else:
        raise ValueError(f"Model name {model_name} not recognized")

    if return_range:
        return list(range(*limits))
    else:
        return limits


def get_text_sequence_positions(model_name, task_name, return_range=False):
    """
    Return the start and end positions of the text sequence in the prompt for a given model and task.
    These positions are constant and based on the otherwise-defined templates.

    Args:
        model_name (str): Name of the model.
        task_name (str): Name of the task.
        return_range (bool): If True, return a list of values in the range [start, end). Otherwise, returns a tuple (start, end).
    """
    if "qwen" in model_name:
        limits = {
            "counting": [16, 16 + 7],
            "arithmetic": [17, 17 + 5],
            "color_ordering": [14, 14 + 26],
            "sentiment_analysis": [14, 14 + 21],
            "factual_recall": [14, 14 + 4],
        }[task_name]
    elif "pixtral" in model_name:
        limits = {
            "counting": [2, 2 + 9],
            "arithmetic": [2, 2 + 8],
            "color_ordering": [2, 2 + 26],
            "sentiment_analysis": [2, 2 + 20],
            "factual_recall": [2, 2 + 4],
        }[task_name]
    elif "gemma" in model_name:
        limits = {
            "counting": [4, 4 + 9],
            "arithmetic": [4, 4 + 8],
            "color_ordering": [4, 4 + 26],
            "sentiment_analysis": [4, 4 + 21],
            "factual_recall": [4, 4 + 4],
        }[task_name]
    else:
        raise ValueError(f"{model_name=}, {task_name=} not recognized")

    if return_range:
        return list(range(*limits))
    else:
        return limits


# Position mapping definitions
pos_mapping_qwen_counting = PositionMapping(
    [(i, i) for i in range(QWEN_PREFIX_LEN)]  # Prefix
    + [
        (QWEN_PREFIX_LEN, QWEN_PREFIX_LEN),
        (QWEN_PREFIX_LEN + 1, QWEN_PREFIX_LEN),
    ]  # "Sequence:" -> Vision start
    + [
        (seq_tok, image_pad_tok)
        for seq_tok in get_text_sequence_positions("qwen", "counting", True)
        for image_pad_tok in get_image_positions("qwen", "counting", True)
    ]  # sequence -> image
    + [
        (
            get_text_sequence_positions("qwen", "counting")[1] + i,
            get_image_positions("qwen", "counting")[1] + i,
        )
        for i in range(23)
        # (i, i + (15 + 81 - (16 + 7))) for i in range(16 + 7, 46)
    ]  # "." -> "Vision end"; Query
)

pos_mapping_qwen_arithmetic = PositionMapping(
    [(i, i) for i in range(QWEN_PREFIX_LEN)]  # Prefix
    + [
        (QWEN_PREFIX_LEN, QWEN_PREFIX_LEN),
        (QWEN_PREFIX_LEN + 1, QWEN_PREFIX_LEN),
        (QWEN_PREFIX_LEN + 2, QWEN_PREFIX_LEN),
    ]  # "Question: " -> Vision start
    + [
        (text_data_pos, img_data_pos)
        for text_data_pos in get_text_sequence_positions("qwen", "arithmetic", True)
        for img_data_pos in get_image_positions("qwen", "arithmetic", True)
    ]  # "a+b" -> image
    + [
        (
            get_text_sequence_positions("qwen", "arithmetic")[1],
            get_image_positions("qwen", "arithmetic")[1],
        )
    ]  # "." -> "Vision end"
    + [
        (
            get_text_sequence_positions("qwen", "arithmetic")[1] + 1 + i,
            get_image_positions("qwen", "arithmetic")[1] + 1 + i,
        )
        for i in range(21)
    ]  # Query
)

pos_mapping_qwen_ordering = PositionMapping(
    [(i, i) for i in range(QWEN_PREFIX_LEN)]  # Prefix
    + [
        (text_data_pos, img_data_pos)
        for text_data_pos in get_text_sequence_positions("qwen", "color_ordering", True)
        for img_data_pos in range(QWEN_PREFIX_LEN, QWEN_PREFIX_LEN + 82)
    ]  # Textual scene -> image
    + [
        (get_text_sequence_positions("qwen", "color_ordering")[1], QWEN_PREFIX_LEN + 82)
    ]  # "." -> "vision end"
    + [
        (
            get_text_sequence_positions("qwen", "color_ordering")[1] + 1 + i,
            QWEN_PREFIX_LEN + 82 + 1 + i,
        )
        for i in range(23)
    ]  # Query
)

pos_mapping_qwen_sentiment = PositionMapping(
    [(i, i) for i in range(QWEN_PREFIX_LEN)]  # Prefix
    + [
        (text_data_pos, img_data_pos)
        for text_data_pos in get_text_sequence_positions(
            "qwen", "sentiment_analysis", True
        )
        for img_data_pos in range(QWEN_PREFIX_LEN, QWEN_PREFIX_LEN + 82)
    ]  # '"""<SCENE_DESCRIPTION>..."""\n -> image
    + [
        (
            get_text_sequence_positions("qwen", "sentiment_analysis")[1],
            QWEN_PREFIX_LEN + 82,
        ),
        (
            get_text_sequence_positions("qwen", "sentiment_analysis")[1],
            QWEN_PREFIX_LEN + 82 + 1,
        ),
    ]  # '"""\n' -> "vision end" and '"""\n' -> "\n"
    + [
        (
            get_text_sequence_positions("qwen", "sentiment_analysis")[1] + 1 + i,
            QWEN_PREFIX_LEN + 82 + 2 + i,
        )
        for i in range(21)
    ]  # Query
)

pos_mapping_qwen_factual = PositionMapping(
    [(i, i) for i in range(QWEN_PREFIX_LEN)]  # Prefix
    + [
        (text_data_pos, img_data_pos)
        for text_data_pos in get_text_sequence_positions("qwen", "factual_recall", True)
        for img_data_pos in range(QWEN_PREFIX_LEN, QWEN_PREFIX_LEN + 82)
    ]  # "Consider ENTITY" -> image
    + [
        (get_text_sequence_positions("qwen", "factual_recall")[1], QWEN_PREFIX_LEN + 82)
    ]  # ".\n" -> "vision end"
    + [
        (
            get_text_sequence_positions("qwen", "factual_recall")[1] + 1 + i,
            QWEN_PREFIX_LEN + 82 + 1 + i,
        )
        for i in range(18)
    ]  # Query
)

pos_mapping_pixtral_counting = PositionMapping(
    [(i, i) for i in range(PIXTRAL_PREFIX_LEN)]  # Prefix
    + [
        (text_data_pos, img_data_pos)
        for text_data_pos in get_text_sequence_positions("pixtral", "counting", True)
        for img_data_pos in get_image_positions("pixtral", "counting", True)
    ]  # "Sequence: <seq>" -> image
    + [
        (
            get_text_sequence_positions("pixtral", "counting")[1],
            get_image_positions("pixtral", "counting")[1],
        )
    ]  # "." -> "[IMG_END]"
    + [
        (
            get_text_sequence_positions("pixtral", "counting")[1] + 1 + i,
            get_image_positions("pixtral", "counting")[1] + 1 + i,
        )
        for i in range(18)
    ]  # Query
)

pos_mapping_pixtral_arithmetic = PositionMapping(
    [(i, i) for i in range(PIXTRAL_PREFIX_LEN)]  # Prefix
    + [
        (text_data_pos, img_data_pos)
        for text_data_pos in get_text_sequence_positions("pixtral", "arithmetic", True)
        for img_data_pos in get_image_positions("pixtral", "arithmetic", True)
    ]  # "Question: a+b" -> image
    + [
        (
            get_text_sequence_positions("pixtral", "arithmetic")[1],
            get_image_positions("pixtral", "arithmetic")[1],
        )
    ]  # "." -> "[IMG_END]"
    + [
        (
            get_text_sequence_positions("pixtral", "arithmetic")[1] + 1 + i,
            get_image_positions("pixtral", "arithmetic")[1] + 1 + i,
        )
        for i in range(17)
    ]  # Query
)

pos_mapping_pixtral_ordering = PositionMapping(
    [(i, i) for i in range(PIXTRAL_PREFIX_LEN)]  # Prefix
    + [
        (text_data_pos, img_data_pos)
        for text_data_pos in get_text_sequence_positions(
            "pixtral", "color_ordering", True
        )
        for img_data_pos in get_image_positions("pixtral", "color_ordering", True)
    ]  # Textual scene -> image
    + [
        (
            get_text_sequence_positions("pixtral", "color_ordering")[1],
            get_image_positions("pixtral", "color_ordering")[1],
        )
    ]  # "." -> "[IMG_END]"
    + [
        (
            get_text_sequence_positions("pixtral", "color_ordering")[1] + 1 + i,
            get_image_positions("pixtral", "color_ordering")[1] + 1 + i,
        )
        for i in range(19)
    ]  # Query
)

pos_mapping_pixtral_sentiment = PositionMapping(
    [(i, i) for i in range(PIXTRAL_PREFIX_LEN)]  # Prefix
    + [
        (text_data_pos, img_data_pos)
        for text_data_pos in get_text_sequence_positions(
            "pixtral", "sentiment_analysis", True
        )
        for img_data_pos in get_image_positions("pixtral", "sentiment_analysis", True)
    ]  # '"""<SCENE_DESCRIPTION>..' -> image
    + [
        (
            get_text_sequence_positions("pixtral", "sentiment_analysis")[1],
            get_image_positions("pixtral", "sentiment_analysis")[1],
        ),
        (
            get_text_sequence_positions("pixtral", "sentiment_analysis")[1] + 1,
            get_image_positions("pixtral", "sentiment_analysis")[1] + 1,
        ),
    ]  # '..' -> "[IMG_END]" and "."""\n" -> "\n"
    + [
        (
            get_text_sequence_positions("pixtral", "sentiment_analysis")[1] + 2 + i,
            get_image_positions("pixtral", "sentiment_analysis")[1] + 2 + i,
        )
        for i in range(17)
    ]  # Query
)

pos_mapping_pixtral_factual = PositionMapping(
    [(i, i) for i in range(PIXTRAL_PREFIX_LEN)]  # Prefix
    + [
        (text_data_pos, img_data_pos)
        for text_data_pos in get_text_sequence_positions(
            "pixtral", "factual_recall", True
        )
        for img_data_pos in get_image_positions("pixtral", "factual_recall", True)
    ]  # "Consider ENTITY" -> image
    + [
        (
            get_text_sequence_positions("pixtral", "factual_recall")[1],
            get_image_positions("pixtral", "factual_recall")[1],
        )
    ]  # ".\n" -> "[IMG_END]"
    + [
        (
            get_text_sequence_positions("pixtral", "factual_recall")[1] + 1 + i,
            get_image_positions("pixtral", "factual_recall")[1] + 1 + i,
        )
        for i in range(14)
    ]  # Query
)

pos_mapping_gemma_counting = PositionMapping(
    [(i, i) for i in range(GEMMA_PREFIX_LEN)]  # Prefix
    + [
        (text_data_pos, img_data_pos)
        for text_data_pos in get_text_sequence_positions("gemma", "counting", True)
        for img_data_pos in get_image_positions("gemma", "counting", True)
    ]  # "Sequence: <seq>" -> image
    + [
        (
            get_text_sequence_positions("gemma", "counting")[1],
            get_image_positions("gemma", "counting")[1],
        ),
        (
            get_text_sequence_positions("gemma", "counting")[1],
            get_image_positions("gemma", "counting")[1] + 1,
        ),
    ]  # "." -> "<end_of_image>" & "." -> "\n\n"
    + [
        (
            get_text_sequence_positions("gemma", "counting")[1] + 1 + i,
            get_image_positions("gemma", "counting")[1] + 2 + i,
        )
        for i in range(21)
    ]  # Query
)

pos_mapping_gemma_arithmetic = PositionMapping(
    [(i, i) for i in range(GEMMA_PREFIX_LEN)]  # Prefix
    + [
        (text_data_pos, img_data_pos)
        for text_data_pos in get_text_sequence_positions("gemma", "arithmetic", True)
        for img_data_pos in get_image_positions("gemma", "arithmetic", True)
    ]  # "Question: a+b" -> image
    + [
        (
            get_text_sequence_positions("gemma", "arithmetic")[1],
            get_image_positions("gemma", "arithmetic")[1],
        ),
        (
            get_text_sequence_positions("gemma", "arithmetic")[1],
            get_image_positions("gemma", "arithmetic")[1] + 1,
        ),
    ]  # "." -> "<end_of_image>" & "." -> "\n\n"
    + [
        (
            get_text_sequence_positions("gemma", "arithmetic")[1] + 1 + i,
            get_image_positions("gemma", "arithmetic")[1] + 2 + i,
        )
        for i in range(21)
    ]  # Query
)

pos_mapping_gemma_ordering = PositionMapping(
    [(i, i) for i in range(GEMMA_PREFIX_LEN)]  # Prefix
    + [
        (text_data_pos, img_data_pos)
        for text_data_pos in get_text_sequence_positions(
            "gemma", "color_ordering", True
        )
        for img_data_pos in get_image_positions("gemma", "color_ordering", True)
    ]  # Textual scene -> image
    + [
        (
            get_text_sequence_positions("gemma", "color_ordering")[1],
            get_image_positions("gemma", "color_ordering")[1],
        ),
        (
            get_text_sequence_positions("gemma", "color_ordering")[1],
            get_image_positions("gemma", "color_ordering")[1] + 1,
        ),
    ]  # "." -> "<end_of_image>" & "." -> "\n\n"
    + [
        (
            get_text_sequence_positions("gemma", "color_ordering")[1] + 1 + i,
            get_image_positions("gemma", "color_ordering")[1] + 2 + i,
        )
        for i in range(23)
    ]  # Query
)

pos_mapping_gemma_sentiment = PositionMapping(
    [(i, i) for i in range(GEMMA_PREFIX_LEN)]  # Prefix
    + [
        (text_data_pos, img_data_pos)
        for text_data_pos in get_text_sequence_positions(
            "gemma", "sentiment_analysis", True
        )
        for img_data_pos in get_image_positions("gemma", "sentiment_analysis", True)
    ]  # '"""<SCENE_DESCRIPTION>..."""\n' -> image
    + [
        (
            get_text_sequence_positions("gemma", "sentiment_analysis")[1],
            get_image_positions("gemma", "sentiment_analysis")[1],
        ),
        (
            get_text_sequence_positions("gemma", "sentiment_analysis")[1] + 1,
            get_image_positions("gemma", "sentiment_analysis")[1] + 1,
        ),
    ]  # '"""' -> "<end_of_image>" and \n' -> "\n\n"
    + [
        (
            get_text_sequence_positions("gemma", "sentiment_analysis")[1] + 2 + i,
            get_image_positions("gemma", "sentiment_analysis")[1] + 2 + i,
        )
        for i in range(21)
    ]  # Query
)

pos_mapping_gemma_factual = PositionMapping(
    [(i, i) for i in range(GEMMA_PREFIX_LEN)]  # Prefix
    + [
        (text_data_pos, img_data_pos)
        for text_data_pos in get_text_sequence_positions(
            "gemma", "factual_recall", True
        )
        for img_data_pos in get_image_positions("gemma", "factual_recall", True)
    ]  # "Consider ENTITY" -> image
    + [
        (
            get_text_sequence_positions("gemma", "factual_recall")[1],
            get_image_positions("gemma", "factual_recall")[1],
        ),
        (
            get_text_sequence_positions("gemma", "factual_recall")[1] + 1,
            get_image_positions("gemma", "factual_recall")[1] + 1,
        ),
    ]  # "." -> "<end_of_image>" & "\n" -> "\n\n"
    + [
        (
            get_text_sequence_positions("gemma", "factual_recall")[1] + 2 + i,
            get_image_positions("gemma", "factual_recall")[1] + 2 + i,
        )
        for i in range(18)
    ]  # Query
)

# Model names truncated to 4 characters;
POS_MAPPINGS = {
    "qwen_counting": pos_mapping_qwen_counting,
    "qwen_arithmetic": pos_mapping_qwen_arithmetic,
    "qwen_color_ordering": pos_mapping_qwen_ordering,
    "qwen_sentiment_analysis": pos_mapping_qwen_sentiment,
    "qwen_factual_recall": pos_mapping_qwen_factual,
    "pixt_counting": pos_mapping_pixtral_counting,
    "pixt_arithmetic": pos_mapping_pixtral_arithmetic,
    "pixt_color_ordering": pos_mapping_pixtral_ordering,
    "pixt_sentiment_analysis": pos_mapping_pixtral_sentiment,
    "pixt_factual_recall": pos_mapping_pixtral_factual,
    "gemm_counting": pos_mapping_gemma_counting,
    "gemm_arithmetic": pos_mapping_gemma_arithmetic,
    "gemm_color_ordering": pos_mapping_gemma_ordering,
    "gemm_sentiment_analysis": pos_mapping_gemma_sentiment,
    "gemm_factual_recall": pos_mapping_gemma_factual,
}
