import hashlib
import re
from pathlib import Path
from fastapi import APIRouter, File, HTTPException, UploadFile
from app.services.pdf_processor import extract_pdf_text
from app.services.chunker import create_chunks
from app.services.embeddings import create_embeddings
from app.services.vector_store import (
    PROJECT_ROOT,
    add_to_vector_store,
    is_file_processed
)


router = APIRouter(tags=["Documents"])


UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def process_pdf_bytes(content, filename):

    if not content.startswith(b"%PDF"):
        raise ValueError("The uploaded file is not a valid PDF.")

    # Generate unique hash for this PDF
    file_hash = hashlib.sha256(
        content
    ).hexdigest()

    # Check whether this PDF was already processed
    if is_file_processed(file_hash):

        return {
            "filename": filename,
            "already_processed": True,
            "message": "This PDF has already been processed."
        }

    # Prevent unsafe filenames
    filename = Path(filename).name

    filename = re.sub(
        r"[^A-Za-z0-9._() -]",
        "_",
        filename
    ).strip(" .")

    if not filename.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are allowed.")

    if not filename:
        filename = "uploaded.pdf"

    pdf_path = UPLOAD_DIR / filename

    # Do not overwrite a different PDF with the same filename
    if pdf_path.exists():

        existing_hash = hashlib.sha256(
            pdf_path.read_bytes()
        ).hexdigest()

        if existing_hash != file_hash:

            pdf_path = UPLOAD_DIR / (
                f"{pdf_path.stem}_{file_hash[:8]}.pdf"
            )

    # Save uploaded PDF
    pdf_path.write_bytes(content)

    print("\nExtracting PDF text...")

    pages = extract_pdf_text(
        str(pdf_path)
    )

    if not pages:
        raise ValueError("No readable text found in the PDF.")

    print("Pages extracted:", len(pages))

    print("\nCreating chunks...")

    chunks = create_chunks(
        pages,
        pdf_path.name
    )

    if not chunks:
        raise ValueError("No chunks were generated.")

    print("Chunks created:", len(chunks))

    print("\nGenerating embeddings...")

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = create_embeddings(texts)

    print("\nAdding vectors to FAISS...")

    result = add_to_vector_store(
        chunks=chunks,
        embeddings=embeddings,
        file_hash=file_hash,
        filename=pdf_path.name
    )

    return {
        "filename": pdf_path.name,
        "pages_extracted": len(pages),
        "chunks_created": len(chunks),
        "added_vectors": result["added_vectors"],
        "total_vectors_in_database": result["total_vectors"],
        "message": "PDF successfully processed and added to FAISS."
    }


@router.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...)
):

    try:

        filename = Path(file.filename or "").name

        if not filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are allowed."
            )

        content = await file.read()

        if not content:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty."
            )

        result = process_pdf_bytes(
            content,
            filename
        )

        return result

    except HTTPException:
        raise

    except (ValueError, FileNotFoundError) as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"PDF processing failed: {error}"
        )

    finally:
        await file.close()






















# @router.post("/documents/process")
# def process_existing_pdfs():

#     results = []

#     # Process PDFs already copied into data/uploads
#     for pdf_path in sorted(UPLOAD_DIR.glob("*.pdf")):

#         try:

#             content = pdf_path.read_bytes()

#             result = process_pdf_bytes(
#                 content,
#                 pdf_path.name
#             )

#             results.append(result)

#         except Exception as error:

#             results.append({
#                 "filename": pdf_path.name,
#                 "error": str(error)
#             })

#     return {
#         "files_found": len(results),
#         "results": results
#     }