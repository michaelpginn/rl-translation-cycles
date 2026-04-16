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
from typing import Literal

import nltk
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

# ---------------------------------------------------------------------------
# Sentence extraction (mirrors src/data.py _extract_sentences, replicated
# here without tokenizer — uses a word-count proxy for the length limit).
# ---------------------------------------------------------------------------

_MAX_WORDS = 90  # proxy for max_tokens=128 from the original code


def _extract_sentences(text: str) -> list[str]:
    sentences = nltk.sent_tokenize(text)
    result = []
    for s in sentences:
        s = s.strip()
        words = s.split()
        if len(words) < 5 or len(words) >= _MAX_WORDS:
            continue
        result.append(s)
    return result


def _load_fineweb_sentences(num_sentences: int, seed: int) -> list[str]:
    ds = load_dataset(
        "HuggingFaceFW/fineweb",
        name="sample-10BT",
        split="train",
        streaming=True,
    )
    sentences: list[str] = []
    for example in ds:
        if len(sentences) >= num_sentences:
            break
        sentences.extend(_extract_sentences(example["text"]))  # type: ignore[index]
    return sentences[:num_sentences]


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


# ---------------------------------------------------------------------------
# Reward (mirrors src/modeling/rewards.py compute_sentence_metric,
# replicated here).
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class TranslationCyclesEnv(vf.MultiTurnEnv):
    """2-turn cycle-consistency translation environment.

    Turn 1: English -> target language
    Turn 2: target language -> English (back-translation)
    """

    def __init__(self, metric: str = "chrf", **kwargs):
        self._metric = metric
        self.parser = vf.MaybeThinkParser()
        super().__init__(**kwargs)

    async def env_response(
        self, messages: vf.Messages, state: vf.State, **kwargs
    ) -> vf.Messages:
        # The last assistant message is the forward translation.
        # Strip thinking tokens (e.g. <think>…</think>) before using as input.
        forward_translation = self.parser.parse_answer(messages)
        assert isinstance(forward_translation, str)
        # Retrieve lang_name from state (set in setup_state from info).
        info_raw = state.get("info", {})
        info = json.loads(info_raw) if isinstance(info_raw, str) else info_raw
        lang_name = info.get("lang_name", "the target language")
        backward_prompt = _make_backward_prompt(forward_translation, lang_name)
        new_messages: vf.Messages = [vf.UserMessage(content=backward_prompt)]
        return new_messages


# ---------------------------------------------------------------------------
# load_environment
# ---------------------------------------------------------------------------


def load_environment(
    target_lang: str = "deu_Latn",
    num_sentences: int = 10000,
    metric: Literal["chrf", "bleu"] = "chrf",
    seed: int = 42,
) -> vf.Environment:
    """Load the cycle-consistency translation environment.

    Args:
        target_lang: FLORES-200 language code for the target language.
        num_sentences: Number of FineWeb sentences to stream for the dataset.
        metric: Scoring metric — "chrf" (default) or "bleu".
        seed: Random seed (reserved for future shuffling).
    """
    if target_lang not in LANG_NAMES:
        raise ValueError(
            f"Unknown target_lang {target_lang!r}. "
            f"Supported codes: {sorted(LANG_NAMES)}"
        )
    lang_name = LANG_NAMES[target_lang]

    # --- Dataset ----------------------------------------------------------
    sentences = _load_fineweb_sentences(num_sentences, seed)

    rows = []
    for sentence in sentences:
        forward_prompt = _make_forward_prompt(sentence, lang_name)
        rows.append(
            {
                "prompt": [{"role": "user", "content": forward_prompt}],
                "answer": sentence,
                "info": json.dumps(
                    {"target_lang": target_lang, "lang_name": lang_name}
                ),
            }
        )
    dataset = Dataset.from_list(rows)

    parser = vf.MaybeThinkParser()

    # --- Rubric -----------------------------------------------------------
    async def cycle_consistency_reward(completion, answer, info) -> float:
        """Score the back-translated English against the original."""
        back_translated = parser.parse_answer(completion)
        assert isinstance(back_translated, str)
        scores = _compute_sentence_metric([back_translated], [answer], metric=metric)
        return scores[0]

    rubric = vf.Rubric(funcs=[cycle_consistency_reward])

    # --- Environment ------------------------------------------------------
    return TranslationCyclesEnv(
        metric=metric,
        dataset=dataset,
        rubric=rubric,
        max_turns=2,
    )
