import numpy as np

from sentence_transformers import SentenceTransformer


MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)


# Load the model once
model = SentenceTransformer(MODEL_NAME)


def create_embeddings(texts):

    if not texts:
        return np.empty((0, 384), dtype="float32")

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    return np.asarray(
        embeddings,
        dtype="float32"
    )


def create_query_embedding(question):

    embedding = model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    return np.asarray(
        embedding,
        dtype="float32"
    )