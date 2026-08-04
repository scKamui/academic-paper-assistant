from src.prompt import (
    build_rag_prompt,
    build_structured_analysis_prompt
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
    assert '"hypothesis"' in prompt
    assert '"methodology"' in prompt
    assert '"findings"' in prompt
    assert '"author_stated_limitations"' in prompt
    assert '"ai_suggested_limitations"' in prompt

    # check the important grounding instructions
    assert "Every extracted claim must include a supporting passage" in prompt
    assert "Return only valid JSON" in prompt
    assert '"found": false' in prompt
    assert '"evidence": []' in prompt