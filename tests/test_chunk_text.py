from src.chunk_text import chunk_pages


def test_chunks_page_text_with_overlap():
    # create one sample page with known text
    pages = [
        {
            "page_number": 5,
            "text": "abcdefghij"
        }
    ]

    # split the page into chunks of four characters with one overlapping character
    chunks = chunk_pages(pages, chunk_size=4, overlap=1)

    # check the number of chunks
    assert len(chunks) == 3

    # check the text stored in each chunk
    assert chunks[0]["text"] == "abcd"
    assert chunks[1]["text"] == "defg"
    assert chunks[2]["text"] == "ghij"

    # check that every chunk retains page number 5
    for chunk in chunks:
        assert chunk["page_number"] == 5

    # check that the chunk numbers are 1, 2, and 3
    assert chunks[0]["chunk_number"] == 1
    assert chunks[1]["chunk_number"] == 2
    assert chunks[2]["chunk_number"] == 3