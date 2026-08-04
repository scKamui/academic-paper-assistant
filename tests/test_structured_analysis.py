import pytest

from src.structured_analysis import (
    FIELD_QUERIES,
    generate_structured_analysis,
    retrieve_field_evidence,
)


def test_retrieves_evidence_for_each_field(monkeypatch):
    # create simple fake document data
    chunks = [{"page_number": 1, "chunk_number": 1, "text": "Sample text"}]
    embeddings = object()
    embedding_model = object()

    # store every search call so we can inspect it
    search_calls = []

    def fake_search_chunks(query, chunks, embeddings, model, top_k):
        # record the information passed into the search function
        search_calls.append({
            "query": query,
            "chunks": chunks,
            "embeddings": embeddings,
            "model": model,
            "top_k": top_k,
        })

        # return predictable fake evidence
        return [
            {
                "page_number": 2,
                "chunk_number": 1,
                "text": f"Evidence for: {query}",
                "score": 0.9,
            }
        ]

    # temporarily replace the real semantic search with our fake search
    monkeypatch.setattr(
        "src.structured_analysis.search_chunks",
        fake_search_chunks,
    )

    evidence = retrieve_field_evidence(
        chunks=chunks,
        embeddings=embeddings,
        embedding_model=embedding_model,
        top_k=2,
    )

    # check that evidence was created for every required field
    assert set(evidence.keys()) == set(FIELD_QUERIES.keys())

    # check that the search function was called once for every field
    assert len(search_calls) == len(FIELD_QUERIES)

    # check that every field received the expected fake evidence
    for field_name, query in FIELD_QUERIES.items():
        assert evidence[field_name][0]["text"] == f"Evidence for: {query}"

    # check that every search requested two passages
    for search_call in search_calls:
        assert search_call["top_k"] == 2



def test_parses_structured_analysis_json(monkeypatch):
    # replace the real API call with a predictable JSON response
    def fake_generate_answer(prompt, client=None, json_mode=False):
        assert prompt == "Analyze this paper."
        assert json_mode is True

        return """
        {
            "hypothesis": {
                "found": false,
                "summary": "Not found in the provided evidence.",
                "evidence": []
            }
        }
        """

    monkeypatch.setattr(
        "src.structured_analysis.generate_answer",
        fake_generate_answer,
    )

    analysis = generate_structured_analysis("Analyze this paper.")

    # check that the JSON text became a Python dictionary
    assert analysis["hypothesis"]["found"] is False
    assert analysis["hypothesis"]["evidence"] == []


def test_rejects_invalid_structured_json(monkeypatch):
    # make the fake model return text that is not JSON
    monkeypatch.setattr(
        "src.structured_analysis.generate_answer",
        lambda *args, **kwargs: "This is not JSON.",
    )

    with pytest.raises(ValueError, match="the model returned invalid JSON"):
        generate_structured_analysis("Analyze this paper.")