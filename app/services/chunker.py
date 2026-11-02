from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.utils.cleaner import clean_text


def create_chunks(pages, source_name):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    all_chunks = []

    chunk_id = 0

    for page in pages:

        text = clean_text(page["text"])

        if not text:
            continue

        chunks = splitter.split_text(text)

        for chunk in chunks:

            all_chunks.append({

                "text": chunk,

                "metadata": {
                    "source": source_name,
                    "page": page["page"],
                    "chunk_id": chunk_id
                }

            })

            chunk_id += 1

    return all_chunks
