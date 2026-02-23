"""Data loading for training (FineWeb) and evaluation (FLORES-200)."""

import csv
import logging
import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from torch.utils.data import Dataset

from src.config.experiment_config import ExperimentConfig

logger = logging.getLogger(__name__)

_LANGS_CSV = Path(__file__).resolve().parent.parent / "data" / "flores_plus_langs.csv"
with open(_LANGS_CSV) as _f:
    LANG_NAMES: dict[str, str] = {
        row["code"]: row["name"] for row in csv.DictReader(_f)
    }


def _extract_sentences(text: str, max_len: int, tokenizer) -> list[str]:
    """Split text into sentences and filter by length."""
    # Simple sentence splitting on period/question/exclamation followed by space
    # TODO: Fix edge cases (Dr., Mr., etc)
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    result = []
    for s in sentences:
        s = s.strip()
        words = s.split()
        num_tokens = len(tokenizer(s)["input_ids"])
        if 5 <= len(words) and num_tokens < max_len and s[-1] in ".!?":
            result.append(s)
    return result


class FineWebTrainDataset(Dataset):  # type: ignore[type-arg]
    """English sentences from FineWeb for cycle-consistency training."""

    def __init__(self, config: ExperimentConfig, tokenizer):
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
        # Stream more until we reach the desired number of sentences
        sentences: list[str] = []
        for example in ds:
            if len(sentences) >= config.train_num_sentences:
                break
            extracted = _extract_sentences(
                example["text"],  # type:ignore
                config.train_max_sentence_len,
                tokenizer,
            )
            sentences.extend(extracted)
        self.sentences = sentences[: config.train_num_sentences]
        logger.info(f"Loaded {len(self.sentences)} English sentences for training")

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> str:
        return self.sentences[idx]


class FloresEvalDataset(Dataset):  # type: ignore[type-arg]
    """FLORES-200 parallel data for evaluation."""

    def __init__(self, config: ExperimentConfig):
        self.target_lang = config.language

        logger.info(f"Loading FLORES-200 eval data: eng_Latn -> {self.target_lang}")
        eng = load_dataset(
            config.eval_dataset,
            "eng_Latn",
            trust_remote_code=True,
            split=config.eval_split,
        ).to_pandas()  # type:ignore
        tgt = load_dataset(
            config.eval_dataset,
            self.target_lang,
            trust_remote_code=True,
            split=config.eval_split,
        ).to_pandas()  # type:ignore
        eng["text"] = eng["text"].str.replace("\xa0", " ")  # type:ignore
        tgt["text"] = tgt["text"].str.replace("\xa0", " ")  # type:ignore
        parallel = pd.merge(tgt, eng, on=["id", "split"], suffixes=("_tgt", "_eng"))  # type:ignore
        if config.eval_num_sentences is not None:
            parallel = parallel.iloc[: config.eval_num_sentences]
        self.parallel_df = parallel

    def __len__(self) -> int:
        return len(self.parallel_df)

    def __getitem__(self, idx: int) -> dict:
        return {
            "eng": self.parallel_df.iloc[idx]["text_eng"],
            "tgt": self.parallel_df.iloc[idx]["text_tgt"],
        }
