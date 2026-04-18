"""Prompt templates for translation tasks."""

from src.config.config import ExperimentConfig
from src.data import LANG_NAMES


def make_forward_prompt(sentence: str, config: ExperimentConfig) -> str:
    """Prompt for English -> target language translation."""
    if config.is_nllb:
        return sentence

    lang_name = LANG_NAMES[config.language]
    return (
        f"Translate the following English sentence into {lang_name}.\n\n"
        f"English: {sentence}\n"
        f"{lang_name}:"
    )


def make_backward_prompt(sentence: str, config: ExperimentConfig) -> str:
    """Prompt for target language -> English back-translation."""
    if config.is_nllb:
        return sentence

    lang_name = LANG_NAMES[config.language]
    return (
        f"Translate the following {lang_name} sentence into English.\n\n"
        f"{lang_name}: {sentence}\n"
        f"English:"
    )
