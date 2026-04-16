"""Data loading for training (FineWeb) and evaluation (FLORES-200)."""

import csv
import logging
from pathlib import Path
from pprint import pformat
from typing import Literal

import nltk
import pandas as pd
from datasets import load_dataset
from torch.utils.data import Dataset

from src.config.config import ExperimentConfig

logger = logging.getLogger(__name__)

_LANGS_CSV = Path(__file__).resolve().parent.parent / "data" / "flores_plus_langs.csv"
with open(_LANGS_CSV) as _f:
    LANG_NAMES: dict[str, str] = {
        row["code"]: row["name"] for row in csv.DictReader(_f)
    }


def _extract_sentences(text: str, max_len: int, tokenizer) -> list[str]:
    """Split text into sentences and filter by length."""
    sentences = nltk.sent_tokenize(text)
    result = []
    for s in sentences:
        s = s.strip()
        words: list[str] = s.split()
        num_tokens = len(tokenizer(s)["input_ids"])
        if len(words) < 5 or num_tokens >= max_len:
            continue
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
        logger.info(f"First twenty sentences: {pformat(self.sentences[:20])}")

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> str:
        return self.sentences[idx]


class EvalDataset(Dataset):
    pass


class FloresEvalDataset(EvalDataset):  # type: ignore[type-arg]
    """FLORES-200 parallel data for evaluation.
    See https://huggingface.co/datasets/openlanguagedata/flores_plus#dataset-structure
    """

    def __init__(self, split: Literal["dev", "test"], config: ExperimentConfig):
        self.target_lang = config.language

        logger.info(f"Loading FLORES-200 eval data: eng_Latn -> {self.target_lang}")
        real_split = "devtest" if split == "test" else "dev"
        eng = load_dataset(
            config.eval_dataset,
            "eng_Latn",
            trust_remote_code=True,
            split=real_split,
        ).to_pandas()  # type:ignore
        tgt = load_dataset(
            config.eval_dataset,
            self.target_lang,
            trust_remote_code=True,
            split=real_split,
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


class NLLBTrainDataset(Dataset):  # type: ignore[type-arg]
    """English sentences from NLLB_MD (to repro)."""

    def __init__(
        self,
        config: ExperimentConfig,
        tokenizer,
    ):
        logger.info(
            f"Loading NLLB ({config.train_dataset_subset}), "
            f"extracting {config.train_num_sentences} sentences"
        )
        ds = load_dataset(
            config.train_dataset,
            name=f"eng_Latn-{config.language}",
            split="train",
            streaming=True,
        )
        # Stream more until we reach the desired number of sentences
        sentences: list[str] = []
        for example in ds:
            if len(sentences) >= config.train_num_sentences:
                break
            sentences.extend(example["sentence_eng_Latn"])  # type:ignore
        self.sentences = sentences[: config.train_num_sentences]
        logger.info(f"Loaded {len(self.sentences)} English sentences for training")
        logger.info(f"First twenty sentences: {pformat(self.sentences[:20])}")

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> str:
        return self.sentences[idx]


class NLLBEvalDataset(EvalDataset):  # type: ignore[type-arg]
    """Paired sentences from NLLB_MD (to repro)."""

    def __init__(
        self,
        split: Literal["dev", "test"],
        config: ExperimentConfig,
    ):
        logger.info(
            f"Loading FineWeb ({config.train_dataset_subset}), "
            f"extracting {config.train_num_sentences} sentences"
        )
        real_split = "valid" if split == "dev" else "test"
        self.dataset: Dataset = load_dataset(  # type:ignore
            config.eval_dataset,
            name=f"eng_Latn-{config.language}",
            split=real_split,
        )
        self.lang = config.language

    def __len__(self) -> int:
        return len(self.dataset)  # type:ignore

    def __getitem__(self, idx: int) -> dict:
        return {
            "eng": self.dataset[idx]["sentence_eng_Latn"],
            "tgt": self.dataset[idx][f"sentence_{self.lang}"],
        }
