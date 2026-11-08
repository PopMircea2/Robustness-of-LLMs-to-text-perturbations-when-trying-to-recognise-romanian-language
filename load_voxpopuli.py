"""
Get TEXT ONLY from the VoxPopuli dataset (European Parliament speech transcripts)
without downloading the multi-gigabyte audio files.

WHY A FIXED REVISION: the current "main" version of the dataset no longer has the
small transcript files (asr_*.tsv) - that is why data/ro/asr_dev.tsv gave a 404.
An older revision (commit 0ab0420...) still has them, so we download from there.

IMPORTANT - which text column?
 * raw_text         = original orthographic text (keeps diacritics and punctuation)
 * normalized_text  = normalized text; it can DROP non-ASCII letters (the dataset card
                      example for Croatian shows letters missing). Do NOT use it for
                      Romanian diacritics experiments.
 Some rows have an EMPTY raw_text. The script skips those and tells you how many
 usable sentences it found. If Romanian raw_text is mostly empty, VoxPopuli is not
 suitable for the diacritics experiment (see the message printed at the end).

HOW TO USE IN PYCHARM
1. Install: huggingface_hub, pandas
2. BACK UP your own article.txt first - this script overwrites it.
3. Run this file. It writes article.txt (Romanian) and voxpopuli_<lang>.txt
   (comparison languages).
4. Open the files and check them by eye (diacritics present? punctuation present?).
"""
import random
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

BASE = Path(__file__).resolve().parent
REPO = "facebook/voxpopuli"
REVISION = "0ab04205657e6f50c8f34954af41633aa9879f12"   # older commit that still has the .tsv files
TARGET = "ro"
OTHER_LANGS = ["it", "fr", "es"]
SPLITS = ["asr_dev.tsv", "asr_test.tsv"]                # small files; asr_train.tsv is bigger
TEXT_COLUMN = "raw_text"
MAX_UTTERANCES = 2000
SEED = 0
DIACRITICS = "ăâîșțĂÂÎȘȚşţŞŢ"


def load_texts(lang):
    texts, total_rows = [], 0
    for split in SPLITS:
        try:
            path = hf_hub_download(REPO, f"data/{lang}/{split}",
                                   repo_type="dataset", revision=REVISION)
        except Exception as e:
            print(f"  [{lang}] could not download {split}: {str(e)[:150]}")
            continue
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=3)
        if TEXT_COLUMN not in df.columns:
            print(f"  [{lang}] columns are {list(df.columns)} - change TEXT_COLUMN")
            continue
        total_rows += len(df)
        texts += [t.strip() for t in df[TEXT_COLUMN] if t.strip()]
    print(f"  [{lang}] {len(texts)} usable (non-empty {TEXT_COLUMN}) of {total_rows} rows")
    random.Random(SEED).shuffle(texts)
    return texts[:MAX_UTTERANCES]


def save(lang, texts, filename):
    lines = [t if t[-1] in ".!?\u2026" else t + "." for t in texts]
    (BASE / filename).write_text("\n".join(lines), encoding="utf-8")
    print(f"[{lang}] saved {len(lines)} utterances to {filename}")
    for t in lines[:3]:
        print("     e.g.", t[:120])


def main():
    texts = load_texts(TARGET)
    if texts:
        save(TARGET, texts, "article.txt")
        n_dia = sum(ch in DIACRITICS for t in texts for ch in t)
        print(f"Diacritic characters found in Romanian text: {n_dia}")
        if n_dia < len(texts):
            print("WARNING: very few diacritics - the text may be normalized or stripped.\n"
                  "VoxPopuli may be unsuitable for the diacritics experiment; use another\n"
                  "Romanian source (Romanian Wikipedia, Tatoeba, MOROCO news text).")
    else:
        print("No Romanian text found. Check the messages above, or use another source "
              "(Romanian Wikipedia, Tatoeba, MOROCO).")
    for lang in OTHER_LANGS:
        texts = load_texts(lang)
        if texts:
            save(lang, texts, f"voxpopuli_{lang}.txt")


if __name__ == "__main__":
    main()