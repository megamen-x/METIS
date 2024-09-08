from transformers import pipeline
from transformers import AutoTokenizer
import torch
import re

pt = "RUPunct/RUPunct_big"
device = "cuda" if torch.cuda.is_available() else "cpu"
tk = AutoTokenizer.from_pretrained(pt, strip_accents=False, add_prefix_space=True)
classifier = pipeline("ner", model=pt, tokenizer=tk, aggregation_strategy="first", device=device)


def process_token(token, label):
    if label == "LOWER_O":
        return token
    if label == "LOWER_PERIOD":
        return token + "."
    if label == "LOWER_COMMA":
        return token + ","
    if label == "LOWER_QUESTION":
        return token + "?"
    if label == "LOWER_TIRE":
        return token + "—"
    if label == "LOWER_DVOETOCHIE":
        return token + ":"
    if label == "LOWER_VOSKL":
        return token + "!"
    if label == "LOWER_PERIODCOMMA":
        return token + ";"
    if label == "LOWER_DEFIS":
        return token + "-"
    if label == "LOWER_MNOGOTOCHIE":
        return token + "..."
    if label == "LOWER_QUESTIONVOSKL":
        return token + "?!"
    if label == "UPPER_O":
        return token.capitalize()
    if label == "UPPER_PERIOD":
        return token.capitalize() + "."
    if label == "UPPER_COMMA":
        return token.capitalize() + ","
    if label == "UPPER_QUESTION":
        return token.capitalize() + "?"
    if label == "UPPER_TIRE":
        return token.capitalize() + " —"
    if label == "UPPER_DVOETOCHIE":
        return token.capitalize() + ":"
    if label == "UPPER_VOSKL":
        return token.capitalize() + "!"
    if label == "UPPER_PERIODCOMMA":
        return token.capitalize() + ";"
    if label == "UPPER_DEFIS":
        return token.capitalize() + "-"
    if label == "UPPER_MNOGOTOCHIE":
        return token.capitalize() + "..."
    if label == "UPPER_QUESTIONVOSKL":
        return token.capitalize() + "?!"
    if label == "UPPER_TOTAL_O":
        return token.upper()
    if label == "UPPER_TOTAL_PERIOD":
        return token.upper() + "."
    if label == "UPPER_TOTAL_COMMA":
        return token.upper() + ","
    if label == "UPPER_TOTAL_QUESTION":
        return token.upper() + "?"
    if label == "UPPER_TOTAL_TIRE":
        return token.upper() + " —"
    if label == "UPPER_TOTAL_DVOETOCHIE":
        return token.upper() + ":"
    if label == "UPPER_TOTAL_VOSKL":
        return token.upper() + "!"
    if label == "UPPER_TOTAL_PERIODCOMMA":
        return token.upper() + ";"
    if label == "UPPER_TOTAL_DEFIS":
        return token.upper() + "-"
    if label == "UPPER_TOTAL_MNOGOTOCHIE":
        return token.upper() + "..."
    if label == "UPPER_TOTAL_QUESTIONVOSKL":
        return token.upper() + "?!"

def update_punctuation(hyp: str):
    full_text = []
    tmp = re.split(" |\n", hyp)
    part = ''
    for el in tmp:
        if len(part.split()) < 250:
            part += el + ' '
        else:
            full_text.append(part)
            part = ''
    if part != '':
        full_text.append(part)
    full_answer = []
    for el in full_text:
        output = ""
        preds = classifier(el)
        for item in preds:
            output += " " + process_token(item['word'].strip(), item['entity_group'])
        full_answer.append(output.lstrip())
    return ' '.join(full_answer)
