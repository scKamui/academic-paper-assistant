from src.clean_text import remove_reference_content


def test_removes_references_but_keeps_later_figures():
    # create pages that imitate the structure of the sample paper
    pages = [
        {
            "page_number": 14,
            "text": "Conclusion text.",
        },
        {
            "page_number": 15,
            "text": "Author information\nREFERENCES\nCitation A",
        },
        {
            "page_number": 16,
            "text": "Citation B",
        },
        {
            "page_number": 20,
            "text": "Figure 1. Lung effects",
        },
        {
            "page_number": 21,
            "text": "Table 1. Results",
        },
    ]

    cleaned_pages = remove_reference_content(pages)

    # check that only the reference-only page was removed
    assert len(cleaned_pages) == 4

    # check that the correct original page numbers were retained
    assert [page["page_number"] for page in cleaned_pages] == [14, 15, 20, 21]

    # check that content before the References heading was retained
    assert cleaned_pages[1]["text"] == "Author information"

    # check that the figure and table pages were retained
    assert cleaned_pages[2]["text"] == "Figure 1. Lung effects"
    assert cleaned_pages[3]["text"] == "Table 1. Results"

    # combine the remaining text so removed citations are easy to check
    cleaned_text = "\n".join(page["text"] for page in cleaned_pages)

    assert "Citation A" not in cleaned_text
    assert "Citation B" not in cleaned_text