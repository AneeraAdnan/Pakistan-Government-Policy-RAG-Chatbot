import pymupdf
import easyocr
import numpy as np

from app.utils.cleaner import clean_text


# Load OCR model only when needed
ocr_reader = None


def get_ocr_reader():

    global ocr_reader

    if ocr_reader is None:
        ocr_reader = easyocr.Reader(
            ["en"],
            gpu=False
        )

    return ocr_reader


def needs_ocr(text):

    text = text.strip()

    # Very little extracted text
    if len(text) < 80:
        return True

    words = text.split()

    if len(words) < 8:
        return True

    # Detect pages containing many isolated characters
    short_words = sum(
        1 for word in words
        if len(word.strip(".,;:()[]")) <= 2
    )

    short_ratio = short_words / max(len(words), 1)

    return len(words) >= 12 and short_ratio > 0.45


def extract_pdf_text(pdf_path):

    pages = []

    document = pymupdf.open(pdf_path)

    try:

        for page_number, page in enumerate(
            document,
            start=1
        ):

            # First attempt: normal PDF extraction
            text = page.get_text("text") or ""

            # Second attempt: OCR if extraction is poor
            if needs_ocr(text):

                print(
                    f"Using OCR for page {page_number}..."
                )

                pix = page.get_pixmap(
                    matrix=pymupdf.Matrix(2.5, 2.5),
                    alpha=False
                )

                image = np.frombuffer(
                    pix.samples,
                    dtype=np.uint8
                ).reshape(
                    pix.height,
                    pix.width,
                    pix.n
                )

                reader = get_ocr_reader()

                detected_lines = reader.readtext(
                    image,
                    detail=0,
                    paragraph=True
                )

                ocr_text = "\n".join(detected_lines)

                # Use OCR when it produces more text
                if len(ocr_text.strip()) > len(text.strip()):
                    text = ocr_text

            # Clean extracted text
            text = clean_text(text)

            if text:

                pages.append({
                    "text": text,
                    "page": page_number
                })

    finally:
        document.close()

    return pages