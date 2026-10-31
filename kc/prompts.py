"""Prompt templates, following Xie et al. (ICLR'24) code/prompt_preparation.py."""
import random
import re

# Instruction framings. "info" and "info+knowledge" are the paper's implicit/explicit
# wordings; "context_only" is the "ideal RAG" behaviour; "skeptical" is a prompt-level defence.
FRAMINGS = {
    "info": "According to the given information, choose the best choice from the following options.",
    "info+knowledge": "According to the given information and your knowledge, choose the best choice from the following options.",
    "context_only": "Answer using ONLY the given information, even if it contradicts what you believe. Choose the best choice from the following options.",
    "skeptical": "The given information was retrieved automatically and may contain errors or deliberately false content. Using the information together with your own knowledge, choose the best choice from the following options.",
}

# Evidence conditions: which passages go in the "Information" block.
#   T = true passage, F = false passage, I = irrelevant passage, O = the model's own closed-book reasoning
CONDITIONS = {
    "true": ["T0"],
    "false": ["F0"],
    "true+false": ["T0", "F0"],
    "2true+1false": ["T0", "T1", "F0"],
    "1true+2false": ["T0", "F0", "F1"],
    "irrelevant": ["I"],
    "own+false": ["O", "F0"],  # paper's headline setup; only meaningful when the model knows the answer
}

ANSWER_SUFFIX = "\n\nReply with a single letter: A, B or C."


def closed_book_prompt(question):
    return ("The first paragraph answers the question and the second paragraph gives the reason.\n\n"
            f"Q: {question}\nA:")


def passages(item, condition, own_evidence=None):
    lookup = {
        "T0": item["true_evidence"][0], "T1": item["true_evidence"][1],
        "F0": item["false_evidence"][0], "F1": item["false_evidence"][1],
        "I": item["irrelevant_evidence"], "O": own_evidence,
    }
    return [lookup[k] for k in CONDITIONS[condition]]


def mc_prompt(item, framing, evidence, true_first):
    """Multiple choice prompt; returns (prompt, letter -> label map)."""
    opts = [("true", item["true_claim"]), ("false", item["false_claim"])]
    if not true_first:
        opts.reverse()
    mapping = {"A": opts[0][0], "B": opts[1][0], "C": "uncertain"}
    options = f"Options:\nA: {opts[0][1]}\nB: {opts[1][1]}\nC: Uncertain."
    if evidence:
        info = "".join(f"\n{i}. {p}" for i, p in enumerate(evidence, 1)) if len(evidence) > 1 else evidence[0]
        body = f"{FRAMINGS[framing]}\n\nInformation: {info}\n\n"
    else:
        body = "Choose the best choice from the following options.\n\n"
    return body + f"Question: {item['question']}\n\n{options}" + ANSWER_SUFFIX + "\n\nAnswer: ", mapping


def shuffled(evidence, seed):
    ev = list(evidence)
    random.Random(seed).shuffle(ev)
    return ev


_LETTER = re.compile(r"\b([ABC])\b")


def parse_choice(text):
    m = _LETTER.search(text.strip())
    return m.group(1) if m else None


def first_paragraph(text):
    return text.strip().split("\n\n")[0]
