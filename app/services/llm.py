from ollama import chat


MODEL_NAME = "llama3.2:3b"


def generate_answer(question, context):

    prompt = f"""
You are an assistant that answers questions
about Pakistan government policy documents.

Rules:

1. Answer using ONLY the provided context.
2. Do not use outside knowledge.
3. Do not invent policy information.
4. If the answer is not available in the context,
   say:

   "The information is not available in the
   provided documents."

5. Keep answers clear and concise.
6. Preserve policy section numbers, dates,
   amounts and names accurately.

Retrieved Context:

{context}

User Question:

{question}

Answer:
"""

    response = chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.message.content