import numpy as np

from app.services.embeddings import create_query_embedding

from app.services.vector_store import load_knowledge_base


def mmr(
    query_embedding,
    candidate_embeddings,
    candidate_ids,
    top_k=5,
    lambda_value=0.7
):

    selected = []

    remaining = list(
        range(len(candidate_ids))
    )

    while remaining and len(selected) < top_k:

        best_score = -float("inf")
        best_index = None

        for i in remaining:

            relevance = np.dot(
                query_embedding,
                candidate_embeddings[i]
            )

            if not selected:

                diversity = 0

            else:

                similarities = []

                for selected_index in selected:

                    similarity = np.dot(
                        candidate_embeddings[i],
                        candidate_embeddings[selected_index]
                    )

                    similarities.append(similarity)

                diversity = max(similarities)

            mmr_score = (
                lambda_value * relevance
                -
                (1 - lambda_value) * diversity
            )

            if mmr_score > best_score:

                best_score = mmr_score
                best_index = i

        selected.append(best_index)
        remaining.remove(best_index)

    return [
        candidate_ids[i]
        for i in selected
    ]


def retrieve_documents(question, top_k=5):

    index, chunks = load_knowledge_base()

    if index is None or index.ntotal == 0:
        return []

    query_embedding = create_query_embedding(
        question
    )[0]

    candidate_k = min(
        top_k * 3,
        index.ntotal
    )

    scores, indices = index.search(
        np.asarray(
            [query_embedding],
            dtype="float32"
        ),
        candidate_k
    )

    candidate_ids = [
        int(index_id)
        for index_id in indices[0]
        if index_id != -1
    ]

    if not candidate_ids:
        return []

    # Retrieve stored embeddings directly from FAISS.
    # No need to regenerate candidate embeddings.
    candidate_embeddings = np.vstack([
        index.reconstruct(index_id)
        for index_id in candidate_ids
    ]).astype("float32")

    selected_ids = mmr(
        query_embedding,
        candidate_embeddings,
        candidate_ids,
        top_k=top_k
    )

    score_map = {
        int(index_id): float(score)
        for score, index_id in zip(
            scores[0],
            indices[0]
        )
        if index_id != -1
    }

    results = []

    for index_id in selected_ids:

        result = chunks[index_id].copy()

        result["score"] = score_map.get(index_id)

        results.append(result)

    return results