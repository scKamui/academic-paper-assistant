from src.prompt import (
    build_analysis_verification_prompt,
    build_rag_prompt,
    build_structured_analysis_prompt,
)


def test_builds_grounded_prompt_with_sources():
    # create sample search results with known citations
    search_results = [
        {
            "page_number": 6,
            "chunk_number": 5,
            "text": "Vaping may worsen asthma and COPD.",
            "score": 0.664,
        },
        {
            "page_number": 5,
            "chunk_number": 1,
            "text": "Studies examined changes in lung function.",
            "score": 0.663,
        },
    ]

    query = "What damage can vaping cause to the lungs?"

    prompt = build_rag_prompt(query, search_results)

    # check that the student's question appears in the prompt
    assert query in prompt

    # check that both passages appear in the prompt
    for result in search_results:
        assert result["text"] in prompt

    # check that each passage has the correct page and chunk label
    for source_number, result in enumerate(search_results, start=1):
        source_label = (
            f"[Source {source_number} | "
            f"PDF page {result['page_number']} | "
            f"Chunk {result['chunk_number']}]"
        )
        assert source_label in prompt

    # check that the prompt tells the model not to invent information
    assert "Do not invent facts, methods, findings, or limitations." in prompt
    assert "If the sources do not provide enough evidence, say so clearly." in prompt

def test_builds_structured_analysis_prompt():
    # create sample evidence for each structured field
    evidence_by_field = {
        "hypothesis": [
            {
                "page_number": 2,
                "chunk_number": 1,
                "text": "The authors expected vaping to affect lung function.",
                "score": 0.9,
            }
        ],
        "methodology": [],
        "findings": [
            {
                "page_number": 6,
                "chunk_number": 5,
                "text": "Vaping was associated with greater loss of lung function.",
                "score": 0.8,
            }
        ],
        "author_stated_limitations": [
            {
                "page_number": 4,
                "chunk_number": 5,
                "text": "Most human studies examined short-term exposure.",
                "score": 0.7,
            }
        ],
    }

    prompt = build_structured_analysis_prompt(evidence_by_field)

    # check that each field has its own evidence section
    assert "Hypothesis evidence:" in prompt
    assert "Methodology evidence:" in prompt
    assert "Findings evidence:" in prompt
    assert "Author Stated Limitations evidence:" in prompt

    # check that the passages and page labels appear
    assert "The authors expected vaping to affect lung function." in prompt
    assert "[Source 1 | PDF page 2 | Chunk 1]" in prompt
    assert "[Source 1 | PDF page 6 | Chunk 5]" in prompt
    assert "[Source 1 | PDF page 4 | Chunk 5]" in prompt

    # check that an empty field is clearly identified
    assert "No passages were retrieved for this field." in prompt

    # check that the required JSON fields appear
    assert '"document_type"' in prompt
    assert '"hypothesis"' in prompt
    assert '"methodology"' in prompt
    assert '"findings"' in prompt
    assert '"author_stated_limitations"' in prompt
    assert '"ai_suggested_limitations"' in prompt
    assert '"items"' in prompt

    # check the important grounding instructions
    assert "Every extracted claim must include its own supporting passage" in prompt
    assert "Do not describe a review article as an experiment" in prompt
    assert "return 2 to 4 distinct major claims" in prompt
    assert "return 2 to 4 distinct items" in prompt
    assert "Do not repeat an author-stated limitation" in prompt
    assert "must be a new inference" in prompt
    assert "Return only valid JSON" in prompt
    assert '"found": false' in prompt
    assert '"evidence": []' in prompt


def test_builds_analysis_verification_prompt():
    # create a claim that is broader than the passage attached to it
    analysis = {
        "findings": {
            "found": True,
            "items": [
                {
                    "claim": "Vaping causes several lung diseases.",
                    "evidence": [
                        {
                            "page_number": 6,
                            "passage": "Vaping may worsen asthma.",
                        }
                    ],
                }
            ],
        }
    }
    evidence_by_field = {
        "findings": [
            {
                "page_number": 6,
                "chunk_number": 2,
                "text": "Vaping may worsen asthma.",
                "score": 0.8,
            }
        ]
    }

    prompt = build_analysis_verification_prompt(
        analysis,
        evidence_by_field,
    )

    # check that the claim, evidence, and source location are available for review
    assert "Vaping causes several lung diseases." in prompt
    assert "Vaping may worsen asthma." in prompt
    assert "PDF page 6" in prompt

    # check that the verifier is told how to handle an overbroad claim
    assert "rewrite the claim more narrowly" in prompt
    assert "Do not return or rewrite evidence passages" in prompt
    assert '"item_number": 1' in prompt
    assert '"revised_claim"' in prompt
