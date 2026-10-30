import re
from datasets import load_dataset

ANSWER_PAT = re.compile(r"\banswer\s*:?\s*\(?\s*([A-D])\s*\)?", re.IGNORECASE)

def load_medqa(n_eval: int, dataset_name: str, config_name: str, split: str):
    return load_dataset(dataset_name, config_name, split=f"{split}[:{n_eval}]")

def format_question(ex):
    opts = "\n".join([f"{o['key']}. {o['value']}" for o in ex["options"]])
    return f"{ex['question']}\n\nOptions:\n{opts}"

def build_messages(question_block):
    return [{
        "role": "user", 
        "content": question_block + "\n\nThink step by step in 2-4 sentences, then end your response with 'Answer: X' where X is one of A, B, C, D."
    }]

def extract_generated_letter(text):
    m = ANSWER_PAT.search(text)
    return m.group(1).upper() if m else None

def strip_answer(text):
    m = ANSWER_PAT.search(text)
    return text[:m.start()].strip() if m else text.strip()

def split_sentences(text):
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sents if len(s) > 10]

def normalize_letter(x):
    if x is None: return None
    for ch in str(x).upper():
        if ch in "ABCD": return ch
    return None
