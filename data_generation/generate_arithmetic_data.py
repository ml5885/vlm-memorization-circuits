import os
import random
from PIL import Image
import sys

sys.path.append("..")
sys.path.append("../third_party/TransformerLens")
from general_utils import set_deterministic


DIGITS_PER_OPERAND = 2
SYMBOLS_DIR = "./arithmetic_symbols"
RESULTS_DIR = "../data/arithmetic/images"


def load_digits():
    """
    Loads digit images for the symbols directory.
    """
    digits = [Image.open(os.path.join(SYMBOLS_DIR, str(i) + ".png")) for i in range(10)]
    return digits


def load_operators():
    """
    Loads operator images for the symbols directory.
    """
    operators = []
    for op in ["plus", "minus", "mult", "divide"]:
        operators.append((op, Image.open(os.path.join(SYMBOLS_DIR, str(op) + ".png"))))
    return operators


def get_operand_range(operator, previous_operand, operand_min, operand_max):
    if operator == "plus" or operator == "+":
        return range(operand_min, operand_max - previous_operand)
    elif operator == "minus" or operator == "-":
        return range(operand_min, previous_operand + 1)
    elif operator == "mult" or operator == "*":
        return range(operand_min, operand_max)
    elif operator == "divide" or operator == "/":
        return range(max(1, operand_min), previous_operand + 1)
    else:
        raise ValueError(f"Operator {operator} is not supported")


def main():
    """
    load digits and operands and generate images of basic arithmetic operations
    :return:
    """
    set_deterministic(42)
    digit_images = load_digits()
    operators = load_operators()
    equals = Image.open(os.path.join(SYMBOLS_DIR, "equal.png"))

    num_images = 1000
    generated = 0
    while generated < num_images:
        operator, operator_img = random.choice(operators)
        op1 = random.randint(
            10 ** (DIGITS_PER_OPERAND - 1), (10**DIGITS_PER_OPERAND) - 1
        )
        try:
            op2 = random.choice(
                list(
                    get_operand_range(
                        operator,
                        op1,
                        10 ** (DIGITS_PER_OPERAND - 1),
                        10**DIGITS_PER_OPERAND,
                    )
                )
            )
        except IndexError:
            continue

        new_im_name = f"{op1}{operator}{op2}.png"
        if os.path.exists(os.path.join(RESULTS_DIR, new_im_name)):
            continue

        op1_images = [digit_images[int(d)] for d in str(op1)]
        op2_images = [digit_images[int(d)] for d in str(op2)]
        images = op1_images + [operator_img] + op2_images + [equals]

        widths, heights = zip(*(i.size for i in images))

        total_width = sum(widths)
        max_height = max(heights)

        new_im = Image.new("RGB", (total_width, max_height))

        x_offset = 0
        for im in images:
            new_im.paste(im, (x_offset, 0))
            x_offset += im.size[0]

        new_im.save(os.path.join(RESULTS_DIR, new_im_name))
        generated += 1


if __name__ == "__main__":
    main()
