from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.services.retriever import retrieve_documents


router = APIRouter(tags=["Retrieval"])


class RetrieveRequest(BaseModel):

    question: str = Field(min_length=2)

    top_k: int = Field(
        default=5,
        ge=1,
        le=20
    )


@router.post("/retrieve")
def retrieve(request: RetrieveRequest):

    results = retrieve_documents(
        request.question,
        top_k=request.top_k
    )

    return {
        "question": request.question,
        "results": results
    }

## This endpoint lets you test FAISS retrieval without calling Ollama.