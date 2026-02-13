"""Prompt templates for translation tasks."""

from src.data import LANG_NAMES


def make_forward_prompt(sentence: str, target_lang: str) -> str:
    """Prompt for English -> target language translation."""
    lang_name = LANG_NAMES[target_lang]
    return (
        f"Translate the following English sentence into {lang_name}.\n\n"
        f"English: {sentence}\n"
        f"{lang_name}:"
    )


def make_backward_prompt(sentence: str, source_lang: str) -> str:
    """Prompt for target language -> English back-translation."""
    lang_name = LANG_NAMES[source_lang]
    return (
        f"Translate the following {lang_name} sentence into English.\n\n"
        f"{lang_name}: {sentence}\n"
        f"English:"
    )
