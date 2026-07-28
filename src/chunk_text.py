from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_pages(extracted_pages, chunk_size=1000, overlap=200):

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
    )

    # create an empty list for all chunks
    chunks = []

    # go through each extracted page
    for page_data in extracted_pages:

        # get the page number and full page text
        page_number = page_data["page_number"]
        page_text = page_data["text"]

        # split the page while preferring paragraph, line, and word boundaries
        page_chunks = text_splitter.split_text(page_text)


        for chunk_number, chunk_text in enumerate(page_chunks, start=1):
            # store the chunk while retaining its original page number
            chunks.append({
                "page_number": page_number,
                "chunk_number": chunk_number,
                "text": chunk_text
            })

    # return all chunks
    return chunks