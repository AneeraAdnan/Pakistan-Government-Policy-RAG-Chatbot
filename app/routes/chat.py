from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.services.retriever import retrieve_documents

from app.services.llm import generate_answer


router = APIRouter(tags=["Chat"])


class ChatRequest(BaseModel):

    question: str = Field(min_length=2)


@router.post("/chat")
def chat_endpoint(request: ChatRequest):

    results = retrieve_documents(
        request.question,
        top_k=5
    )

    if not results:

        return {
            "answer": "No relevant information was found in the knowledge base.",
            "sources": []
        }

    context_parts = []

    sources = []

    seen_sources = set()

    for result in results:

        text = result["text"]

        metadata = result["metadata"]

        source = metadata["source"]

        page = metadata["page"]

        context_parts.append(
            f"""
Source: {source}
Page: {page}

{text}
"""
        )

        # Avoid duplicate source/page entries
        source_key = (source, page)

        if source_key not in seen_sources:

            seen_sources.add(source_key)

            sources.append({
                "document": source,
                "page": page
            })

    context = "\n\n".join(context_parts)

    answer = generate_answer(
        request.question,
        context
    )

    return {
        "answer": answer,
        "sources": sources
    }