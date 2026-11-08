"""
Test fastText and Naive Bayes on the SAME informal Romanian comments that DistilBERT was tested on.

INPUT (put these next to this script):
  informal_ro.txt            one comment per line (exported from Colab, see the export cell)
  nb_lid.joblib              trained Naive Bayes (made by naive_bayes_lid.py)
  lid.176.ftz, lid.176.bin   fastText models
  informal_<model>_correct.csv      OPTIONAL, exported from Colab (column: correct); every file
                                    matching informal_*_correct.csv is added as one more model

OUTPUT:
  informal_models_results.csv     accuracy per model and group, with 95 % bootstrap interval
  informal_models_paired.csv      paired diacritics / lowercase tests per model
  informal_correct_by_model.csv   1/0 per comment and model (NO comment text, safe to commit)

Every model answers only among the four languages es/fr/it/ro, so the comparison is fair.
accuracy = share of comments predicted as Romanian (the comments are assumed to be Romanian).

PRIVACY: informal_ro.txt contains real user comments. Do NOT commit it to git
(add it to .gitignore) and quote at most a few short examples in the paper.
"""
import math
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from experiment import STRIP

# ---------------------------------------------------------------- settings
BASE = Path(__file__).resolve().parent
TEXT_FILE = BASE / "informal_ro.txt"
NB_FILE = BASE / "nb_lid.joblib"
FT_FILES = {"fastText small (ftz)": BASE / "lid.176.ftz",
            "fastText large (bin)": BASE / "lid.176.bin"}
LANGS = ("es", "fr", "it", "ro")
N_BOOTSTRAP = 1000
SEED = 0
# -------------------------------------------------------------------------

DIAC = set("ăâîșțĂÂÎȘȚşţŞŢ")


def strip_all(t):
    return t.translate(STRIP)


def is_caps(t):
    letters = [c for c in t if c.isalpha()]
    return len(letters) > 0 and sum(c.isupper() for c in letters) > 0.7 * len(letters)


def make_nb(path):
    model = joblib.load(path)

    def predict(texts):
        return list(model.predict(list(texts)))
    return predict


def make_ft(path):
    import fasttext
    model = fasttext.load_model(str(path))

    def all_labels(text):
        """(label, probability) for every language. Calls the low-level function directly, because
        model.predict() crashes with numpy 2 ('Unable to avoid copy')."""
        try:
            pairs = model.f.predict(text + "\n", -1, 0.0, "strict")      # list of (probability, label)
            return [(l.replace("__label__", ""), p) for p, l in pairs]
        except AttributeError:                                            # other fastText builds
            labels, probs = model.predict(text, k=-1)
            return [(l.replace("__label__", ""), p) for l, p in zip(labels, probs)]

    def predict(texts):
        out = []
        for t in texts:
            cand = [(p, l) for l, p in all_labels(t.replace("\n", " ")) if l in LANGS]
            out.append(max(cand)[1] if cand else "other")
        return out
    return predict


def boot_ci(correct, rng):
    n = len(correct)
    if n == 0:
        return float("nan"), float("nan")
    idx = rng.integers(0, n, size=(N_BOOTSTRAP, n))
    return tuple(np.percentile(correct[idx].mean(axis=1), [2.5, 97.5]))


def mcnemar_exact(a, b):
    """Exact two-sided McNemar test on two boolean arrays of the same sentences."""
    n01 = int(np.sum(~a & b))     # only model b correct
    n10 = int(np.sum(a & ~b))     # only model a correct
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    p = 2 * sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return n01, n10, min(1.0, p)


def main():
    for p in [TEXT_FILE, NB_FILE, *FT_FILES.values()]:
        if not p.exists():
            raise FileNotFoundError(f"Missing {p.name} - see the instructions at the top of this file.")
    texts = [t.strip() for t in TEXT_FILE.read_text(encoding="utf-8").splitlines() if t.strip()]
    print(f"{len(texts)} comments loaded")

    caps = np.array([is_caps(t) for t in texts])
    dia = np.array([any(c in DIAC for c in t) for t in texts])
    short = np.array([len(t) < 30 for t in texts])
    groups = {
        "all": np.ones(len(texts), bool),
        "ALL CAPS": caps,
        "not caps": ~caps,
        "not caps, with diacritics": ~caps & dia,
        "not caps, no diacritics": ~caps & ~dia,
        "under 30 chars": short,
        "30+ chars": ~short,
    }

    models = {"Naive Bayes": make_nb(NB_FILE)}
    for name, path in FT_FILES.items():
        models[name] = make_ft(path)

    rng = np.random.default_rng(SEED)
    rows, paired_rows, correct = [], [], {}
    for mname, predict in models.items():
        c = np.array([p == "ro" for p in predict(texts)])
        correct[mname] = c
        for gname, mask in groups.items():
            ci = boot_ci(c[mask].astype(float), rng)
            rows.append({"model": mname, "group": gname, "n": int(mask.sum()),
                         "accuracy_ro": round(float(c[mask].mean()), 4) if mask.any() else None,
                         "ci_low": round(float(ci[0]), 4), "ci_high": round(float(ci[1]), 4)})

        # paired test: same sentences with diacritics, original vs diacritics removed
        sel = [t for t, m in zip(texts, ~caps & dia) if m]
        orig = np.array([p == "ro" for p in predict(sel)])
        stripped = np.array([p == "ro" for p in predict([strip_all(t) for t in sel])])
        n01, n10, pval = mcnemar_exact(orig, stripped)
        paired_rows.append({"model": mname, "test": "diacritics removed (not caps, with diacritics)",
                            "n": len(sel), "acc_original": round(float(orig.mean()), 4),
                            "acc_changed": round(float(stripped.mean()), 4),
                            "only_original_correct": n10, "only_changed_correct": n01,
                            "mcnemar_p": round(pval, 6)})
        low = np.array([p == "ro" for p in predict([t.lower() for t in texts])])
        n01, n10, pval = mcnemar_exact(c, low)
        paired_rows.append({"model": mname, "test": "lowercased (all comments)", "n": len(texts),
                            "acc_original": round(float(c.mean()), 4),
                            "acc_changed": round(float(low.mean()), 4),
                            "only_original_correct": n10, "only_changed_correct": n01,
                            "mcnemar_p": round(pval, 6)})

    found = sorted(BASE.glob("informal_*_correct.csv"))
    if found:
        print("Per-comment result files found:", ", ".join(f.name for f in found))
    else:
        print(f"No informal_*_correct.csv found in {BASE} - the Colab models (DistilBERT, XLM-R) are NOT included.")
        print("CSV files in this folder:", ", ".join(sorted(q.name for q in BASE.glob("*.csv"))) or "none")
    for f in found:
        label = f.stem[len("informal_"):-len("_correct")]
        df = pd.read_csv(f)
        d = df["correct"].astype(bool).to_numpy()
        if len(d) != len(texts):
            print(f"WARNING: {f.name} has {len(d)} rows but there are {len(texts)} comments - skipped")
            continue
        correct[label] = d
        # paired tests from the Colab export (same comments, one change), if the columns exist
        if "correct_no_diacritics" in df.columns:
            sel = ~caps & dia
            o, ch = d[sel], df["correct_no_diacritics"].astype(bool).to_numpy()[sel]
            n01, n10, pval = mcnemar_exact(o, ch)
            paired_rows.append({"model": label, "test": "diacritics removed (not caps, with diacritics)",
                                "n": int(sel.sum()), "acc_original": round(float(o.mean()), 4),
                                "acc_changed": round(float(ch.mean()), 4),
                                "only_original_correct": n10, "only_changed_correct": n01,
                                "mcnemar_p": round(pval, 6)})
        if "correct_lowercase" in df.columns:
            lc = df["correct_lowercase"].astype(bool).to_numpy()
            n01, n10, pval = mcnemar_exact(d, lc)
            paired_rows.append({"model": label, "test": "lowercased (all comments)", "n": len(d),
                                "acc_original": round(float(d.mean()), 4), "acc_changed": round(float(lc.mean()), 4),
                                "only_original_correct": n10, "only_changed_correct": n01,
                                "mcnemar_p": round(pval, 6)})
        for gname, mask in groups.items():
            ci = boot_ci(d[mask].astype(float), rng)
            rows.append({"model": label, "group": gname, "n": int(mask.sum()),
                         "accuracy_ro": round(float(d[mask].mean()), 4) if mask.any() else None,
                         "ci_low": round(float(ci[0]), 4), "ci_high": round(float(ci[1]), 4)})

    res = pd.DataFrame(rows)
    table = res.pivot(index="group", columns="model", values="accuracy_ro").reindex(list(groups))
    pd.set_option("display.width", 220, "display.max_columns", 20)
    print("\nACCURACY ON ROMANIAN (rows = group, columns = model)")
    print(table.round(3).to_string())
    print("\nPAIRED TESTS (same comments, one change)")
    paired = pd.DataFrame(paired_rows)
    print(paired.to_string(index=False))

    names = list(correct)
    print("\nMcNEMAR between models (all comments; p < 0.05 means the models differ)")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            n01, n10, pval = mcnemar_exact(correct[names[i]], correct[names[j]])
            print(f"  {names[i]} vs {names[j]}: only first correct={n10}, only second correct={n01}, p={pval:.4g}")

    res.to_csv(BASE / "informal_models_results.csv", index=False, encoding="utf-8")
    paired.to_csv(BASE / "informal_models_paired.csv", index=False, encoding="utf-8")
    pd.DataFrame({m: c.astype(int) for m, c in correct.items()}).to_csv(
        BASE / "informal_correct_by_model.csv", index_label="comment_index", encoding="utf-8")
    print("\nSaved informal_models_results.csv, informal_models_paired.csv, informal_correct_by_model.csv")


if __name__ == "__main__":
    main()