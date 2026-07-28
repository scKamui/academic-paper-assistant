from src.clean_text import (
    remove_reference_content,
    remove_repeated_lines,
)



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



def test_removes_lines_repeated_across_pages():
    pages = [
        {
            "page_number": 1,
            "text": "Study Name Page 1\nBody A\nJournal Footer\nShared twice",
        },
        {
            "page_number": 2,
            "text": "Study Name Page 2\nBody B\nJournal Footer\nShared twice",
        },
        {
            "page_number": 3,
            "text": "Study Name Page 3\nBody C\nJournal Footer",
        },
    ]

    cleaned_pages = remove_repeated_lines(pages, min_occurrences=3)

    # page-numbered headers should be recognized as the same repeated line
    assert not any(
        "Study Name Page" in page["text"]
        for page in cleaned_pages
    )

    # the identical journal footer should be removed
    assert not any("Journal Footer" in page["text"] for page in cleaned_pages)

    # unique body text should remain
    assert any("Body A" in page["text"] for page in cleaned_pages)
    assert any("Body B" in page["text"] for page in cleaned_pages)
    assert any("Body C" in page["text"] for page in cleaned_pages)

    # a line appearing on only two pages should remain
    assert sum(
        "Shared twice" in page["text"]
        for page in cleaned_pages
    ) == 2
