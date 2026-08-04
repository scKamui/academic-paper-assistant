import copy
import json

import pytest

from src.structured_analysis import (
    FIELD_QUERIES,
    generate_structured_analysis,
    retrieve_field_evidence,
    validate_structured_analysis,
    verify_structured_analysis,
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


def create_valid_analysis():
    # create one complete response that can be reused by the validator tests
    return {
        "document_type": {
            "type": "review_article",
            "explanation": "The authors summarize earlier research.",
            "evidence": [
                {
                    "page_number": 1,
                    "passage": "This article reviews earlier research.",
                }
            ],
        },
        "hypothesis": {
            "found": False,
            "statement_type": "Not applicable",
            "summary": "Not found in the provided evidence.",
            "evidence": [],
        },
        "methodology": {
            "found": True,
            "summary": "The authors searched a research database.",
            "evidence": [
                {
                    "page_number": 2,
                    "passage": "The authors searched PubMed.",
                }
            ],
        },
        "findings": {
            "found": True,
            "items": [
                {
                    "claim": "Vaping affected lung function.",
                    "evidence": [
                        {
                            "page_number": 3,
                            "passage": "Vaping affected lung function.",
                        }
                    ],
                }
            ],
        },
        "author_stated_limitations": {
            "found": False,
            "items": [],
        },
        "ai_suggested_limitations": [
            {
                "suggestion": "The evidence may be short term.",
                "reason": "The retrieved research covered short periods.",
                "based_on_pages": [3],
            }
        ],
    }


def create_retrieved_evidence():
    # imitate the page-numbered passages returned by semantic search
    return {
        "hypothesis": [
            {
                "page_number": 1,
                "chunk_number": 1,
                "text": "This article reviews earlier research.",
                "score": 0.9,
            }
        ],
        "methodology": [
            {
                "page_number": 2,
                "chunk_number": 1,
                "text": "The authors searched PubMed.",
                "score": 0.8,
            }
        ],
        "findings": [
            {
                "page_number": 3,
                "chunk_number": 1,
                "text": "Vaping affected lung function.",
                "score": 0.7,
            }
        ],
        "author_stated_limitations": [],
    }


def test_validates_grounded_structured_analysis():
    analysis = create_valid_analysis()
    evidence = create_retrieved_evidence()

    # a complete response with real passages should pass validation
    assert validate_structured_analysis(analysis, evidence) == analysis


def test_rejects_a_page_that_was_not_retrieved():
    analysis = create_valid_analysis()
    evidence = create_retrieved_evidence()

    # change one citation to a page the search never returned
    analysis["findings"]["items"][0]["evidence"][0]["page_number"] = 99

    with pytest.raises(ValueError, match="PDF page 99, which was not retrieved"):
        validate_structured_analysis(analysis, evidence)


def test_rejects_a_passage_that_is_not_on_the_cited_page():
    analysis = create_valid_analysis()
    evidence = create_retrieved_evidence()

    # replace the real quote with text that does not exist in the source
    analysis["methodology"]["evidence"][0]["passage"] = "An invented passage."

    with pytest.raises(ValueError, match="was not found on PDF page 2"):
        validate_structured_analysis(analysis, evidence)


def test_rejects_a_found_field_without_evidence():
    analysis = create_valid_analysis()
    evidence = create_retrieved_evidence()

    # a model cannot claim it found methodology without showing its source
    analysis["methodology"]["evidence"] = []

    with pytest.raises(ValueError, match="methodology needs evidence"):
        validate_structured_analysis(analysis, evidence)


def test_verifies_analysis_using_a_second_json_response(monkeypatch):
    analysis = create_valid_analysis()
    evidence = create_retrieved_evidence()

    # return predictable decisions instead of contacting Groq
    def fake_generate_answer(prompt, client=None, json_mode=False):
        assert "verifying a structured academic-paper analysis" in prompt
        assert "Vaping affected lung function." in prompt
        assert json_mode is True
        return json.dumps({
            "document_type": {
                "supported": True,
                "revised_explanation": "The paper reviews earlier research.",
            },
            "hypothesis": {
                "supported": True,
                "revised_summary": "Not found in the provided evidence.",
            },
            "methodology": {
                "supported": True,
                "revised_summary": "The authors searched PubMed.",
            },
            "findings": [
                {
                    "item_number": 1,
                    "supported": True,
                    "revised_claim": "Vaping affected lung function.",
                }
            ],
            "author_stated_limitations": [],
            "ai_suggested_limitations": [
                {
                    "item_number": 1,
                    "supported": False,
                    "revised_suggestion": "",
                    "revised_reason": "",
                }
            ],
        })

    monkeypatch.setattr(
        "src.structured_analysis.generate_answer",
        fake_generate_answer,
    )

    verified_analysis = verify_structured_analysis(
        analysis,
        evidence,
    )

    # the verifier may revise wording or remove an unsupported item
    assert verified_analysis["document_type"]["explanation"] == (
        "The paper reviews earlier research."
    )
    assert verified_analysis["methodology"]["summary"] == (
        "The authors searched PubMed."
    )
    assert verified_analysis["ai_suggested_limitations"] == []

    # page numbers and passages remain exactly as they were before verification
    assert verified_analysis["findings"]["items"][0]["evidence"] == (
        analysis["findings"]["items"][0]["evidence"]
    )


def test_verifier_can_narrow_a_claim_without_changing_evidence(monkeypatch):
    analysis = create_valid_analysis()
    evidence = create_retrieved_evidence()

    def fake_generate_answer(*args, **kwargs):
        return json.dumps({
            "document_type": {
                "supported": True,
                "revised_explanation": "The authors summarize earlier research.",
            },
            "hypothesis": {
                "supported": True,
                "revised_summary": "Not found in the provided evidence.",
            },
            "methodology": {
                "supported": True,
                "revised_summary": "The authors searched a database.",
            },
            "findings": [
                {
                    "item_number": 1,
                    "supported": True,
                    "revised_claim": "The paper reports effects on lung function.",
                }
            ],
            "author_stated_limitations": [],
            "ai_suggested_limitations": [
                {
                    "item_number": 1,
                    "supported": True,
                    "revised_suggestion": "The evidence may be short term.",
                    "revised_reason": "The research covered short periods.",
                }
            ],
        })

    monkeypatch.setattr(
        "src.structured_analysis.generate_answer",
        fake_generate_answer,
    )

    original_evidence = copy.deepcopy(
        analysis["findings"]["items"][0]["evidence"]
    )
    verified = verify_structured_analysis(analysis, evidence)

    assert verified["findings"]["items"][0]["claim"] == (
        "The paper reports effects on lung function."
    )
    assert verified["findings"]["items"][0]["evidence"] == original_evidence
