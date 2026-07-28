from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_embedding_model():
    # load the local model used to convert text into vectors
    return SentenceTransformer(MODEL_NAME)


def embed_chunks(chunks, model):
    # collect only the text from each chunk
    chunk_texts = []

    for chunk in chunks:
        chunk_texts.append(chunk["text"])

    # convert each chunk into a normalized embedding vector
    embeddings = model.encode_document(
        chunk_texts,
        normalize_embeddings=True,
    )

    return embeddings