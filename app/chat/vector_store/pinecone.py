import os
import pinecone
from langchain.vectorstores.pinecone import Pinecone

from app.chat.embeddings.openai import embeddings

pinecone.Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENV_NAME"),
)

vector_store = Pinecone.from_existing_index(
    os.getenv("PINECONE_INDEX_NAME"),
    embeddings
)


# write doc for following function

def build_retriever(chat_args, k):
    """
    Build a retriever for the chat application using the provided chat arguments.

    This function constructs a search argument dictionary with a filter based on the
    `pdf_id` from the `chat_args`.
    Args:
        chat_args (ChatArgs): An object containing the chat parameters, including `pdf_id`.

    Returns:
        Retriever: A retriever instance configured with the specified document.
    """
    search_args = {
        "filter": {"pdf_id": chat_args.pdf_id},
        "k": k,
    }
    return vector_store.as_retriever(
        search_args=search_args,
    )
