"""
Naive Bayes language identification (character n-grams) + robustness grid.

WHAT IT DOES
1. Loads sentences for Romanian (article.txt) and the comparison languages
   (voxpopuli_it.txt, voxpopuli_fr.txt, voxpopuli_es.txt) - same splitting as experiment.py.
2. Removes duplicates, balances the languages, splits 80/20 (stratified, fixed seed).
3. Trains Naive Bayes on character 1-4-grams (case and diacritics kept).
4. Prints the clean test report and a confusion matrix.
5. ROBUSTNESS GRID on the Romanian test sentences:
      diacritics removed (0/25/50/75/100 % of them)  x  random typos (0/5/10/20/30/40 %)
   - diacritics only          -> row with typo rate 0
   - typos only               -> column with 0 % diacritics removed
   - combination              -> all other cells
   Diacritics are removed first, then typos are added.
6. Saves: nb_lid.joblib, nb_train.csv, nb_test.csv, nb_results.csv (every condition),
   nb_results_accuracy.csv (accuracy grid).

HOW TO RUN IN PYCHARM
- Needs scikit-learn, pandas, numpy. Keep this file next to experiment.py and the text files.
- Right-click > Run 'naive_bayes_lid'.

NOTES
- Perturbations are applied to the TEST sentences only, never to the training data.
- accuracy_ro = share of Romanian test sentences predicted as Romanian.
- CI = 95 % bootstrap interval over sentences (resampling the sentences 1000 times).
- margin = gap between the log-probabilities of the best and second-best language,
  divided by sentence length. Naive Bayes probabilities saturate at 1.0, so the margin is
  a more informative "confidence" measure. It is NOT comparable to fastText confidence.
- If everything stays near 100 %, set TRUNCATE_CHARS = 40 to test very short text.
"""
import collections
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

from experiment import STRIP, char_noise, cedilla_variant, load_sentences

# ---------------------------------------------------------------- settings
BASE = Path(__file__).resolve().parent
DATA_FILES = {
    "ro": BASE / "article.txt",
    "it": BASE / "voxpopuli_it.txt",
    "fr": BASE / "voxpopuli_fr.txt",
    "es": BASE / "voxpopuli_es.txt",
}
MAX_PER_LANGUAGE = 2000
TEST_SIZE = 0.2
NGRAM_RANGE = (1, 4)
ALPHA = 0.1                 # smoothing; fixed in advance, do not tune on the test set
SEED = 0

DIAC_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)       # share of diacritics removed
TYPO_RATES = (0.0, 0.05, 0.10, 0.20, 0.30, 0.40)   # share of letters hit by a typo
N_SEEDS = 5                                        # repetitions for random conditions
N_BOOTSTRAP = 1000
TRUNCATE_CHARS = None       # e.g. 40 -> cut every test sentence to its first 40 characters
# -------------------------------------------------------------------------


def load_data():
    per_lang = {}
    for lang, path in DATA_FILES.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path.name}. Run load_voxpopuli.py or convert_voxpopuli_tsv.py first.")
        sents = list(dict.fromkeys(load_sentences(path)))
        random.Random(SEED).shuffle(sents)
        per_lang[lang] = sents[:MAX_PER_LANGUAGE]
        print(f"{lang}: {len(per_lang[lang])} unique sentences")
    n = min(len(v) for v in per_lang.values())
    if n < 200:
        print(f"WARNING: only {n} sentences per language - results will be unreliable.")
    rows = [(lang, s) for lang, v in per_lang.items() for s in v[:n]]
    print(f"Using {n} sentences per language ({len(rows)} in total)")
    return pd.DataFrame(rows, columns=["lang", "text"])


def build_model():
    return make_pipeline(
        CountVectorizer(analyzer="char", ngram_range=NGRAM_RANGE,
                        lowercase=False, strip_accents=None),
        MultinomialNB(alpha=ALPHA),
    )


def partial_strip(text, fraction, rnd):
    """Remove each Romanian diacritic with probability `fraction`."""
    if fraction <= 0:
        return text
    out = []
    for ch in text:
        if ord(ch) in STRIP and (fraction >= 1 or rnd.random() < fraction):
            out.append(STRIP[ord(ch)])
        else:
            out.append(ch)
    return "".join(out)


def perturb(text, diac_fraction, typo_rate, rnd):
    text = partial_strip(text, diac_fraction, rnd)
    if typo_rate > 0:
        text = char_noise(text, typo_rate, rnd)
    return text


def margins(model, sentences):
    logp = model.predict_log_proba(sentences)
    top2 = np.sort(logp, axis=1)[:, -2:]
    gap = top2[:, 1] - top2[:, 0]
    lengths = np.array([max(len(s), 1) for s in sentences])
    return gap / lengths


def bootstrap_ci(values, rng):
    n = len(values)
    idx = rng.integers(0, n, size=(N_BOOTSTRAP, n))
    means = values[idx].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def evaluate_condition(model, sentences, diac_fraction, typo_rate, label):
    deterministic = (diac_fraction in (0.0, 1.0)) and typo_rate == 0.0
    n_seeds = 1 if deterministic else N_SEEDS
    correct = np.zeros((n_seeds, len(sentences)))
    seed_acc, seed_margin = [], []
    wrong = collections.Counter()
    for s in range(n_seeds):
        rnd = random.Random(f"{label}-{s}")
        perturbed = [perturb(t, diac_fraction, typo_rate, rnd) for t in sentences]
        pred = model.predict(perturbed)
        correct[s] = (pred == "ro")
        seed_acc.append(correct[s].mean())
        seed_margin.append(margins(model, perturbed).mean())
        wrong.update(p for p in pred if p != "ro")
    per_sentence = correct.mean(axis=0)
    low, high = bootstrap_ci(per_sentence, np.random.default_rng(SEED))
    return {
        "diacritics_removed": diac_fraction, "typo_rate": typo_rate,
        "accuracy_ro": round(float(np.mean(seed_acc)), 4),
        "ci_low": round(float(low), 4), "ci_high": round(float(high), 4),
        "std_over_seeds": round(float(np.std(seed_acc)), 4),
        "mean_margin": round(float(np.mean(seed_margin)), 4),
        "top_confusions": dict(wrong.most_common(3)),
    }


def main():
    df = load_data()
    train, test = train_test_split(df, test_size=TEST_SIZE, stratify=df["lang"],
                                   random_state=SEED)
    model = build_model()
    model.fit(train["text"], train["lang"])

    pred = model.predict(test["text"])
    print("\n=== Clean test set ===")
    print(classification_report(test["lang"], pred, digits=3))
    labels = sorted(df["lang"].unique())
    cm = pd.DataFrame(confusion_matrix(test["lang"], pred, labels=labels),
                      index=[f"true_{l}" for l in labels],
                      columns=[f"pred_{l}" for l in labels])
    print(cm)

    ro = test.loc[test["lang"] == "ro", "text"].tolist()
    if TRUNCATE_CHARS:
        ro = [s[:TRUNCATE_CHARS] for s in ro]
        print(f"\nNOTE: Romanian test sentences cut to the first {TRUNCATE_CHARS} characters")
    print(f"\n=== Robustness grid on {len(ro)} Romanian test sentences ===")

    rows = []
    for f in DIAC_FRACTIONS:
        for p in TYPO_RATES:
            rows.append(evaluate_condition(model, ro, f, p, f"d{f}-t{p}"))
        print(f"  diacritics_removed={int(f * 100)}% done")
    res = pd.DataFrame(rows)

    # the cedilla encoding as one extra row
    ced = np.mean(model.predict([cedilla_variant(s) for s in ro]) == "ro")
    print(f"\ncedilla_variant (wrong ş/ţ encoding) accuracy_ro = {ced:.3f}")

    acc = res.pivot(index="diacritics_removed", columns="typo_rate", values="accuracy_ro")
    mar = res.pivot(index="diacritics_removed", columns="typo_rate", values="mean_margin")
    pd.set_option("display.width", 220, "display.max_columns", 20, "display.max_colwidth", 60)
    print("\nACCURACY (rows = share of diacritics removed, columns = typo rate)")
    print(acc.round(3).to_string())
    print("\nMEAN MARGIN per character (higher = more certain)")
    print(mar.round(3).to_string())
    print("\nFULL TABLE")
    print(res.to_string(index=False))

    res.to_csv(BASE / "nb_results.csv", index=False, encoding="utf-8")
    acc.to_csv(BASE / "nb_results_accuracy.csv", encoding="utf-8")
    joblib.dump(model, BASE / "nb_lid.joblib")
    train.to_csv(BASE / "nb_train.csv", index=False, encoding="utf-8")
    test.to_csv(BASE / "nb_test.csv", index=False, encoding="utf-8")
    size_mb = (BASE / "nb_lid.joblib").stat().st_size / 1e6
    print(f"\nSaved nb_results.csv, nb_results_accuracy.csv, nb_lid.joblib "
          f"({size_mb:.1f} MB, {len(model[0].vocabulary_)} n-gram features), "
          "nb_train.csv, nb_test.csv")


if __name__ == "__main__":
    main()