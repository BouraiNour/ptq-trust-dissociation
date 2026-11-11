import re

LETTERS = "ABCD"

# ----------------------------------------------------------------------------- pure helpers
PRIMARY = re.compile(r"Answer\s*:\s*\*{0,2}\s*\(?([A-D])\b")
LOOSE = re.compile(r"(?i:\banswer\s+is\s*:?\s*\*{0,2}\s*\(?)([A-D])\b")


def parse_answer(text):
    """Return (letter_primary, letter_loose, parse_method, rationale_text)."""
    p = list(PRIMARY.finditer(text))
    if p:
        m = p[-1]
        return m.group(1), m.group(1), "primary", text[:m.start()].rstrip("* \n")
    l = list(LOOSE.finditer(text))
    if l:
        m = l[-1]
        cut = max(text.rfind(c, 0, m.start()) for c in ".!?\n") + 1   # start of the sentence containing the match
        return None, m.group(1), "loose", text[:cut].strip()
    return None, None, "none", text.strip()


def split_sentences(text, min_len=10):
    """Split on sentence punctuation; merge fragments <= min_len chars into a neighbour (never drop text)."""
    raw = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    out, carry = [], ""
    for s in raw:
        if carry:
            s, carry = carry + " " + s, ""
        if len(s) <= min_len:
            if out:
                out[-1] += " " + s
            else:
                carry = s
        else:
            out.append(s)
    if carry:
        out.append(carry)
    return out


def format_question(ex):
    opts = "\n".join(f"{o['key']}. {o['value']}" for o in ex["options"])
    return f"{ex['question']}\n\nOptions:\n{opts}"


def build_messages(block):
    return [{"role": "user", "content": block +
             "\n\nThink step by step in 2-4 sentences, then end your response with "
             "'Answer: X' where X is one of A, B, C, D."}]


def margin(logits, t):
    others = [v for i, v in enumerate(logits) if i != t]
    return float(logits[t] - max(others))
