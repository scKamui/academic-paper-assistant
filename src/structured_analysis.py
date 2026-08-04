from src.search import search_chunks


# use a separate search question for each part of the paper
FIELD_QUERIES = {
    "hypothesis": (
        "What hypothesis, research question, or objective do the authors state?"
    ),
    "methodology": (
        "What methods, study design, participants, data, and procedures were used?"
    ),
    "findings": (
        "What results, findings, or conclusions do the authors report?"
    ),
    "author_stated_limitations": (
        "What limitations, weaknesses, or uncertainties do the authors explicitly state?"
    ),
}


def retrieve_field_evidence(
    chunks,
    embeddings,
    embedding_model,
    top_k=3,
):
    # create a dictionary for the evidence found for each field
    evidence_by_field = {}

    # search the paper separately for every structured field
    for field_name, query in FIELD_QUERIES.items():
        search_results = search_chunks(
            query=query,
            chunks=chunks,
            embeddings=embeddings,
            model=embedding_model,
            top_k=top_k,
        )

        evidence_by_field[field_name] = search_results

    return evidence_by_field