import numpy as np


def search_chunks(query, chunks, embeddings, model, top_k=5):
    # reject an empty student question
    if not query.strip():
        raise ValueError("query must not be empty")

    # each embedding must correspond to exactly one chunk
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must have the same length")

    # at least one result must be requested
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    # there is nothing to search if the document produced no chunks
    if not chunks:
        return []

    # convert the student's question into a normalized vector
    query_embedding = model.encode_query(
        query,
        normalize_embeddings=True,
    )

    # compare the query vector against every chunk vector
    similarity_scores = embeddings @ query_embedding

    # sort the scores from highest to lowest and keep the best indexes
    top_indices = np.argsort(similarity_scores)[::-1][:top_k]

    # build and return the search results
    results = []

    for index in top_indices:
        # convert the NumPy index into a regular Python integer
        chunk_index = int(index)

        # get the matching chunk using the embedding's position
        chunk = chunks[chunk_index]

        # create a result containing its text, page number,
        # chunk number, and similarity score
        result = {
            "text": chunk["text"],
            "page_number": chunk["page_number"],
            "chunk_number": chunk["chunk_number"],
            "score": float(similarity_scores[chunk_index]),
        }

        # append the result
        results.append(result)

    return results