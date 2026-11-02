import os
import pickle
import threading

from pathlib import Path

import faiss
import numpy as np


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Storage paths
VECTORSTORE_DIR = (
    PROJECT_ROOT / "vectorstore" / "faiss_index"
)

STORAGE_DIR = PROJECT_ROOT / "storage"


INDEX_PATH = VECTORSTORE_DIR / "index.faiss"

CHUNKS_PATH = STORAGE_DIR / "processed_chunks.pkl"

FILES_PATH = STORAGE_DIR / "processed_files.pkl"


# Prevent simultaneous updates inside this process
store_lock = threading.Lock()


def create_directories():

    VECTORSTORE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    STORAGE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


def load_chunks():

    if not CHUNKS_PATH.exists():
        return []

    with open(CHUNKS_PATH, "rb") as file:
        return pickle.load(file)


def load_processed_files():

    if not FILES_PATH.exists():
        return {}

    with open(FILES_PATH, "rb") as file:
        return pickle.load(file)


def save_pickle(path, data):

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with open(temporary_path, "wb") as file:
        pickle.dump(data, file)

    os.replace(temporary_path, path)


def load_faiss_index():

    create_directories()

    if not INDEX_PATH.exists():
        return None

    return faiss.read_index(str(INDEX_PATH))


def is_file_processed(file_hash):

    with store_lock:

        processed_files = load_processed_files()

        return file_hash in processed_files


def add_to_vector_store(
    chunks,
    embeddings,
    file_hash,
    filename
):

    vectors = np.asarray(
        embeddings,
        dtype="float32"
    )

    if len(chunks) == 0:
        raise ValueError("No chunks to store.")

    if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
        raise ValueError("Embedding count does not match chunk count.")

    with store_lock:

        create_directories()

        processed_files = load_processed_files()

        # Prevent duplicate PDF processing
        if file_hash in processed_files:

            index = load_faiss_index()

            return {
                "already_processed": True,
                "added_vectors": 0,
                "total_vectors": (
                    index.ntotal if index is not None else 0
                )
            }

        old_chunks = load_chunks()

        # Load existing index
        if INDEX_PATH.exists():

            index = faiss.read_index(
                str(INDEX_PATH)
            )

            if index.ntotal != len(old_chunks):
                raise RuntimeError(
                    "FAISS index and chunk metadata are inconsistent."
                )

            if index.d != vectors.shape[1]:
                raise ValueError(
                    "Embedding dimensions do not match."
                )

        else:

            if old_chunks:
                raise RuntimeError(
                    "Chunk metadata exists but FAISS index is missing."
                )

            # Inner product with normalized embeddings = cosine similarity
            index = faiss.IndexFlatIP(
                vectors.shape[1]
            )

        # IMPORTANT:
        # Add new vectors to the existing index.
        # Do not create a new index here.
        index.add(vectors)

        # Append new chunks to old chunks
        updated_chunks = old_chunks + chunks

        # Record processed PDF hash
        processed_files[file_hash] = {
            "filename": filename,
            "sha256": file_hash,
            "chunks_added": len(chunks)
        }

        # Save updated FAISS index
        temporary_index = INDEX_PATH.with_suffix(
            ".faiss.tmp"
        )

        faiss.write_index(
            index,
            str(temporary_index)
        )

        os.replace(
            temporary_index,
            INDEX_PATH
        )

        # Save updated metadata
        save_pickle(
            CHUNKS_PATH,
            updated_chunks
        )

        save_pickle(
            FILES_PATH,
            processed_files
        )

        return {
            "already_processed": False,
            "added_vectors": len(chunks),
            "total_vectors": index.ntotal
        }


def load_knowledge_base():

    with store_lock:

        index = load_faiss_index()
        chunks = load_chunks()

    if index is None:

        if chunks:
            raise RuntimeError(
                "Chunks exist but FAISS index is missing."
            )

        return None, []

    if index.ntotal != len(chunks):

        raise RuntimeError(
            "FAISS vector count does not match chunk count."
        )

    return index, chunks