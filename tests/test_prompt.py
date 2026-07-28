from src.prompt import build_rag_prompt


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