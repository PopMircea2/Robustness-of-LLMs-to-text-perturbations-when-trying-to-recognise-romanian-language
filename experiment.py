"""
Pilot experiment: how do missing diacritics and random typos affect fastText
language identification on Romanian article.txt?

HOW TO RUN IN PYCHARM
1. Put this file, article.txt, lid.176.ftz and lid.176.bin in the SAME folder
   (your PyCharm project folder).
2. Make sure the project interpreter has: fasttext-wheel, pandas, numpy<2
   (Settings > Project > Python Interpreter > "+").
3. Right-click this file > Run 'experiment'  (or press the green play button).
"""
import collections
import random
import re
import statistics
import unicodedata
from pathlib import Path

# ---------------------------------------------------------------- settings
BASE = Path(__file__).resolve().parent           # folder of this script
ARTICLE_PATH = BASE / "article.txt"              # your Romanian news article.txt
MODEL_FILES = {
    "small (ftz)": BASE / "lid.176.ftz",
    "large (bin)": BASE / "lid.176.bin",
}
MIN_SENTENCE_LEN = 40                            # drop very short fragments
NOISE_RATES = (0.05, 0.10, 0.20)                 # typo rates to test
N_SEEDS = 5                                      # repetitions for random noise
# -------------------------------------------------------------------------

LETTERS = "abcdefghijklmnopqrstuvwxyz"

# Legacy cedilla forms -> correct comma-below forms
CEDILLA_TO_COMMA = str.maketrans({
    "\u015f": "\u0219", "\u015e": "\u0218",
    "\u0163": "\u021b", "\u0162": "\u021a",
})
# Comma-below -> cedilla (the "wrong" but common encoding)
COMMA_TO_CEDILLA = str.maketrans({
    "\u0219": "\u015f", "\u0218": "\u015e",
    "\u021b": "\u0163", "\u021a": "\u0162",
})
# Remove all Romanian diacritics
STRIP = str.maketrans({
    "\u0103": "a", "\u0102": "A", "\u00e2": "a", "\u00c2": "A",
    "\u00ee": "i", "\u00ce": "I",
    "\u0219": "s", "\u0218": "S", "\u015f": "s", "\u015e": "S",
    "\u021b": "t", "\u021a": "T", "\u0163": "t", "\u0162": "T",
})


def load_sentences(path, min_len=MIN_SENTENCE_LEN):
    text = Path(path).read_text(encoding="utf-8")
    text = unicodedata.normalize("NFC", text).translate(CEDILLA_TO_COMMA)
    text = re.sub(r"\s+", " ", text)
    sents = re.split(r"(?<=[.!?\u2026])\s+", text)
    return [s.strip() for s in sents if len(s.strip()) >= min_len]


def strip_diacritics(t):
    return t.translate(STRIP)


def cedilla_variant(t):
    return t.translate(COMMA_TO_CEDILLA)


def char_noise(text, p, rnd):
    """Each letter is hit with probability p: delete, substitute, insert or swap."""
    chars, out, i = list(text), [], 0
    while i < len(chars):
        c = chars[i]
        if c.isalpha() and rnd.random() < p:
            op = rnd.choice(("del", "sub", "ins", "swap"))
            if op == "del":
                pass
            elif op == "sub":
                out.append(rnd.choice(LETTERS))
            elif op == "ins":
                out.extend([c, rnd.choice(LETTERS)])
            elif op == "swap" and i + 1 < len(chars):
                out.extend([chars[i + 1], c])
                i += 1
            else:
                out.append(c)
        else:
            out.append(c)
        i += 1
    return "".join(out)


def build_conditions():
    """name -> (function(sentence, rnd) -> sentence, number of random seeds)"""
    conds = {
        "clean": (lambda s, r: s, 1),
        "no_diacritics": (lambda s, r: strip_diacritics(s), 1),
        "cedilla_variant": (lambda s, r: cedilla_variant(s), 1),
    }
    for p in NOISE_RATES:
        conds[f"typos_{int(p * 100)}%"] = (
            lambda s, r, p=p: char_noise(s, p, r), N_SEEDS)
    conds["no_diacritics+typos_10%"] = (
        lambda s, r: char_noise(strip_diacritics(s), 0.10, r), N_SEEDS)
    return conds


def predict(model, sentences):
    labels, probs = [], []
    for s in sentences:
        (lab,), (pr,) = model.predict(s, k=1)
        labels.append(lab.replace("__label__", ""))
        probs.append(float(pr))
    return labels, probs


def check_files():
    missing = [p for p in [ARTICLE_PATH, *MODEL_FILES.values()] if not p.exists()]
    if missing:
        names = "\n  ".join(str(p) for p in missing)
        raise FileNotFoundError(
            f"These files are missing:\n  {names}\n"
            "Put them in the same folder as experiment.py.")


def main():
    import fasttext
    import pandas as pd

    check_files()
    sents = load_sentences(ARTICLE_PATH)
    print(f"{len(sents)} sentences loaded from {ARTICLE_PATH.name}")
    if len(sents) < 20:
        print("Warning: very few sentences - results will be noisy.")

    rows, examples = [], []
    for mname, mfile in MODEL_FILES.items():
        model = fasttext.load_model(str(mfile))
        for cname, (fn, n_seeds) in build_conditions().items():
            accs, confs = [], []
            wrong = collections.Counter()
            for seed in range(n_seeds):
                rnd = random.Random(seed)
                perturbed = [fn(s, rnd) for s in sents]
                if seed == 0 and mname.startswith("small"):
                    examples.append((cname, perturbed[0]))
                labels, probs = predict(model, perturbed)
                accs.append(sum(l == "ro" for l in labels) / len(labels))
                confs.append(statistics.mean(probs))
                wrong.update(l for l in labels if l != "ro")
            rows.append({
                "model": mname, "condition": cname,
                "accuracy_ro": round(statistics.mean(accs), 3),
                "acc_std": round(statistics.pstdev(accs), 3),
                "mean_confidence": round(statistics.mean(confs), 3),
                "top_confusions": dict(wrong.most_common(3)),
            })

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200, "display.max_colwidth", 60)
    print(df.to_string(index=False))
    df.to_csv(BASE / "results.csv", index=False)
    with open(BASE / "examples.txt", "w", encoding="utf-8") as f:
        for cname, text in examples:
            f.write(f"[{cname}] {text}\n")
    print("Saved results_examples.csv and examples.txt next to the script")


if __name__ == "__main__":
    main()