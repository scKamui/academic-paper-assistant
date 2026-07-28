def chunk_pages(extracted_pages, chunk_size=1000, overlap=200):

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size")

    step_size = chunk_size - overlap
    
    # create an empty list for all chunks
    chunks = []

    # go through each extracted page
    for page_data in extracted_pages:

        # get the page number and full page text
        page_number = page_data["page_number"]
        page_text = page_data["text"]

        # begin at the first character
        # begin chunk numbering at 1
        start = 0
        chunk_number = 1

        # continue while there is page text remaining
        while start < len(page_text):

            # calculate where the current chunk ends
            end = min(start + chunk_size, len(page_text))

            # slice and clean the chunk text
            chunk_text = page_text[start:end].strip()

            # store non-empty chunks with their page and chunk numbers
            if chunk_text:
                chunks.append({
                    "page_number": page_number,
                    "chunk_number": chunk_number,
                    "text": chunk_text
                })
                chunk_number += 1

            if end == len(page_text):
                break

            # move forward by chunk_size minus overlap
            start += step_size

    # return all chunks
    return chunks