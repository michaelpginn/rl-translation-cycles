"""Data loading for training (FineWeb) and evaluation (FLORES-200)."""

import json
import logging
import os
import re

from datasets import load_dataset
from torch.utils.data import Dataset

from src.config.experiment_config import ExperimentConfig

logger = logging.getLogger(__name__)

# FLORES-200 language codes (BCP-47 style used by the dataset)
FLORES_LANG_CODES = {
    "yo": "yor_Latn",
    "ig": "ibo_Latn",
    "ha": "hau_Latn",
    "zu": "zul_Latn",
    "xh": "xho_Latn",
    "gn": "grn_Latn",
    "qu": "que_Latn",
    "ay": "aym_Latn",
    "lo": "lao_Laoo",
    "my": "mya_Mymr",
    "km": "khm_Khmr",
    "am": "amh_Ethi",
}

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")


def _extract_sentences(text: str, max_len: int) -> list[str]:
    """Split text into sentences and filter by length."""
    # Simple sentence splitting on period/question/exclamation followed by space
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    result = []
    for s in sentences:
        s = s.strip()
        words = s.split()
        if 5 <= len(words) <= max_len and s[-1] in ".!?":
            result.append(s)
    return result


class FineWebTrainDataset(Dataset):  # type: ignore[type-arg]
    """English sentences from FineWeb for cycle-consistency training."""

    def __init__(self, config: ExperimentConfig):
        cache_path = os.path.join(
            CACHE_DIR,
            f"fineweb_{config.train_dataset_subset}_{config.train_num_sentences}.json",
        )

        if os.path.exists(cache_path):
            logger.info(f"Loading cached training sentences from {cache_path}")
            with open(cache_path) as f:
                self.sentences: list[str] = json.load(f)
        else:
            logger.info(
                f"Loading FineWeb ({config.train_dataset_subset}), "
                f"extracting {config.train_num_sentences} sentences"
            )
            ds = load_dataset(
                config.train_dataset,
                name=config.train_dataset_subset,
                split="train",
                streaming=True,
            )
            sentences: list[str] = []
            for example in ds:
                if len(sentences) >= config.train_num_sentences:
                    break
                extracted = _extract_sentences(
                    example["text"], config.train_max_sentence_len
                )
                sentences.extend(extracted)

            self.sentences = sentences[: config.train_num_sentences]

            # Cache to disk
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(self.sentences, f)
            logger.info(f"Cached {len(self.sentences)} sentences to {cache_path}")

        logger.info(f"Loaded {len(self.sentences)} English sentences for training")

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> str:
        return self.sentences[idx]


class FloresEvalDataset(Dataset):  # type: ignore[type-arg]
    """FLORES-200 parallel data for evaluation."""

    def __init__(self, config: ExperimentConfig, target_lang: str):
        flores_code = FLORES_LANG_CODES[target_lang]
        logger.info(
            f"Loading FLORES-200 eval data: eng_Latn -> {flores_code}"
        )

        ds = load_dataset(
            config.eval_dataset,
            f"eng_Latn-{flores_code}",
            split=config.eval_split,
            trust_remote_code=True,
        )

        self.source_sentences: list[str] = []  # English
        self.target_sentences: list[str] = []  # Target language
        self.target_lang = target_lang

        eng_col = "sentence_eng_Latn"
        tgt_col = f"sentence_{flores_code}"

        for example in ds:
            self.source_sentences.append(example[eng_col])
            self.target_sentences.append(example[tgt_col])

        logger.info(
            f"Loaded {len(self.source_sentences)} FLORES pairs for {target_lang}"
        )

    def __len__(self) -> int:
        return len(self.source_sentences)

    def __getitem__(self, idx: int) -> dict[str, str]:
        return {
            "source": self.source_sentences[idx],
            "target": self.target_sentences[idx],
        }
