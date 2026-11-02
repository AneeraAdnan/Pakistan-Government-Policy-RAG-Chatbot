import re


def clean_text(text):

    if not text:
        return ""

    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove multiple spaces and tabs
    text = re.sub(r"[ \t]+", " ", text)

    # Remove unnecessary special characters
    # Preserve policy numbering, punctuation and symbols
    text = re.sub(
        r"[^\w\s.,;:!?()/%\-'\u2018\u2019\u201c\u201d]",
        " ",
        text
    )

    # Remove spaces before punctuation
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)

    # Remove excessive blank lines
    text = re.sub(r"\n\s*\n+", "\n", text)

    return text.strip()