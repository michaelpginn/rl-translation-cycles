"""Cycle-consistency translation environment for Prime/Verifiers.

Each rollout is a 2-turn exchange:
  Turn 1 — model translates English → target language
  Turn 2 — model translates target language → English (back-translation)

Reward: chrF (or BLEU) of the back-translated English vs. the original,
        normalised to [0, 1] by dividing by 100.

Dataset: sentences streamed from FineWeb (HuggingFaceFW/fineweb, sample-10BT).
"""

from __future__ import annotations

import json
import os
from typing import Literal

import pandas as pd
import sacrebleu
import verifiers as vf
from datasets import Dataset, load_dataset

# ---------------------------------------------------------------------------
# Language-name lookup (mirrors src/data.py LANG_NAMES, replicated here so
# the environment package stays self-contained).
# ---------------------------------------------------------------------------

# Common FLORES-200 language codes → display names.  Extend as needed.
LANG_NAMES: dict[str, str] = {
    "ace_Arab": "Acehnese (Jawi)",
    "ace_Latn": "Acehnese (Latin)",
    "afr_Latn": "Afrikaans",
    "amh_Ethi": "Amharic",
    "arb_Arab": "Modern Standard Arabic",
    "asm_Beng": "Assamese",
    "azj_Latn": "North Azerbaijani",
    "bel_Cyrl": "Belarusian",
    "ben_Beng": "Bengali",
    "bos_Latn": "Bosnian",
    "bul_Cyrl": "Bulgarian",
    "cat_Latn": "Catalan",
    "ces_Latn": "Czech",
    "ckb_Arab": "Central Kurdish",
    "cym_Latn": "Welsh",
    "dan_Latn": "Danish",
    "deu_Latn": "German",
    "ell_Grek": "Greek",
    "eng_Latn": "English",
    "est_Latn": "Estonian",
    "eus_Latn": "Basque",
    "fin_Latn": "Finnish",
    "fra_Latn": "French",
    "gle_Latn": "Irish",
    "glg_Latn": "Galician",
    "guj_Gujr": "Gujarati",
    "hau_Latn": "Hausa",
    "heb_Hebr": "Hebrew",
    "hin_Deva": "Hindi",
    "hrv_Latn": "Croatian",
    "hun_Latn": "Hungarian",
    "hye_Armn": "Armenian",
    "ibo_Latn": "Igbo",
    "ind_Latn": "Indonesian",
    "isl_Latn": "Icelandic",
    "ita_Latn": "Italian",
    "jpn_Jpan": "Japanese",
    "kan_Knda": "Kannada",
    "kat_Geor": "Georgian",
    "kaz_Cyrl": "Kazakh",
    "khm_Khmr": "Khmer",
    "kor_Hang": "Korean",
    "lit_Latn": "Lithuanian",
    "lug_Latn": "Luganda",
    "lvs_Latn": "Standard Latvian",
    "mal_Mlym": "Malayalam",
    "mar_Deva": "Marathi",
    "mkd_Cyrl": "Macedonian",
    "mlt_Latn": "Maltese",
    "mya_Mymr": "Burmese",
    "nld_Latn": "Dutch",
    "nob_Latn": "Norwegian Bokmål",
    "npi_Deva": "Nepali",
    "nya_Latn": "Nyanja",
    "ory_Orya": "Odia",
    "pan_Guru": "Eastern Panjabi",
    "pol_Latn": "Polish",
    "por_Latn": "Portuguese",
    "ron_Latn": "Romanian",
    "rus_Cyrl": "Russian",
    "sin_Sinh": "Sinhala",
    "slk_Latn": "Slovak",
    "slv_Latn": "Slovenian",
    "sna_Latn": "Shona",
    "som_Latn": "Somali",
    "spa_Latn": "Spanish",
    "srp_Cyrl": "Serbian",
    "swe_Latn": "Swedish",
    "swh_Latn": "Swahili",
    "tam_Taml": "Tamil",
    "tel_Telu": "Telugu",
    "tgk_Cyrl": "Tajik",
    "tgl_Latn": "Tagalog",
    "tha_Thai": "Thai",
    "tur_Latn": "Turkish",
    "ukr_Cyrl": "Ukrainian",
    "urd_Arab": "Urdu",
    "uzn_Latn": "Northern Uzbek",
    "vie_Latn": "Vietnamese",
    "yor_Latn": "Yoruba",
    "zho_Hans": "Chinese (Simplified)",
    "zho_Hant": "Chinese (Traditional)",
    "zsm_Latn": "Standard Malay",
    "zul_Latn": "Zulu",
}


def _load_flores(language: str, split: str, token: str):
    eng = load_dataset(
        "openlanguagedata/flores_plus",
        "eng_Latn",
        # trust_remote_code=True,
        split=split,
        token=token,
    ).to_pandas()  # type:ignore
    tgt = load_dataset(
        "openlanguagedata/flores_plus",
        language,
        # trust_remote_code=True,
        split=split,
        token=token,
    ).to_pandas()  # type:ignore
    eng["text"] = eng["text"].str.replace("\xa0", " ")  # type:ignore
    tgt["text"] = tgt["text"].str.replace("\xa0", " ")  # type:ignore
    return pd.merge(tgt, eng, on=["id", "split"], suffixes=("_tgt", "_eng"))  # type:ignore


# ---------------------------------------------------------------------------
# Prompt helpers (mirrors src/modeling/prompts.py, replicated here).
# ---------------------------------------------------------------------------


def _make_forward_prompt(sentence: str, lang_name: str) -> str:
    return (
        f"Translate the following English sentence into {lang_name}. Output only the translated text and no other text.\n\n"
        f"English: {sentence}"
    )


def _make_backward_prompt(sentence: str, lang_name: str) -> str:
    return (
        f"Translate the following {lang_name} sentence into English. Output only the translated text and no other text.\n\n"
        f"{lang_name}: {sentence}"
    )


def _compute_sentence_metric(
    predictions: list[str],
    references: list[str],
    metric: str = "chrf",
) -> list[float]:
    scores: list[float] = []
    for pred, ref in zip(predictions, references):
        if metric == "bleu":
            scores.append(sacrebleu.sentence_bleu(pred, [ref]).score)
        elif metric == "chrf":
            scores.append(sacrebleu.sentence_chrf(pred, [ref]).score)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    return scores


def load_environment(
    target_lang: str = "deu_Latn",
    metric: Literal["chrf", "bleu"] = "chrf",
    split: Literal["dev", "devtest"] = "dev",
    direction: Literal["forward", "backward"] = "forward",
    seed: int = 42,
) -> vf.Environment:
    """Load the cycle-consistency translation environment.

    Args:
        target_lang: FLORES-200 language code for the target language.
        metric: Scoring metric — "chrf" (default) or "bleu".
        split: dev or devtest
        seed: Random seed (reserved for future shuffling).
    """
    vf.ensure_keys(["HF_TOKEN_FLORES"])
    # login(token=os.environ["HF_TOKEN_FLORES"])

    if target_lang not in LANG_NAMES:
        raise ValueError(
            f"Unknown target_lang {target_lang!r}. "
            f"Supported codes: {sorted(LANG_NAMES)}"
        )
    lang_name = LANG_NAMES[target_lang]
    sentences = _load_flores(target_lang, split, token=os.environ["HF_TOKEN_FLORES"])

    rows = []
    for _, sentence in sentences.iterrows():
        if direction == "forward":
            prompt = _make_forward_prompt(sentence["text_eng"], lang_name)  # type:ignore
            answer = sentence["text_tgt"]
        else:
            prompt = _make_backward_prompt(sentence["text_tgt"], lang_name)  # type:ignore
            answer = sentence["text_eng"]
        rows.append(
            {
                "prompt": [{"role": "user", "content": prompt}],
                "answer": answer,
                "info": json.dumps(
                    {"target_lang": target_lang, "lang_name": lang_name}
                ),
            }
        )
    dataset = Dataset.from_list(rows)

    parser = vf.MaybeThinkParser()

    async def translation_metric(completion, answer) -> float:
        translation = parser.parse_answer(completion)
        assert isinstance(translation, str)
        scores = _compute_sentence_metric([translation], [answer], metric=metric)
        return scores[0]

    rubric = vf.Rubric(funcs=[translation_metric])
    return vf.SingleTurnEnv(dataset=dataset, rubric=rubric)
