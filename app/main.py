from fastapi import FastAPI

from app.routes.documents import router as documents_router

from app.routes.chat import router as chat_router

from app.routes.retrieve import router as retrieve_router


app = FastAPI(
    title="Pakistan Government Policy RAG Chatbot",
    description="RAG chatbot for Pakistan government policy documents",
    version="1.0.0"
)


@app.get("/")
def home():

    return {
        "message": "Pakistan Government Policy RAG Chatbot is running"
    }


# Register API routes
app.include_router(documents_router)

app.include_router(retrieve_router)

app.include_router(chat_router)