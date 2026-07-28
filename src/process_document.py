import argparse
from pathlib import Path

from src.chunk_text import chunk_pages
from src.extract_pdf import extract_pages
from src.embeddings import embed_chunks, load_embedding_model


def process_document(pdf_path, model, chunk_size=1000, overlap=200):
    # extract page-numbered text from the PDF
    extracted_pages = extract_pages(pdf_path)

    # split the extracted pages into page-aware chunks
    chunks = chunk_pages(extracted_pages, chunk_size=chunk_size, overlap=overlap)

    # convert chunks into embedding vectors
    embeddings = embed_chunks(chunks, model)

    # return both results in one dictionary
    return {
        "pages": extracted_pages,
        "chunks": chunks,
        "embeddings": embeddings,
    }


def main():
    # create a command-line parser and explain what the program does
    parser = argparse.ArgumentParser(
        description="Extract and chunk text from an academic PDF."
    )

    # require the user to provide a PDF path
    parser.add_argument(
        "pdf_path",
        type=Path,
        help="Path to the PDF that should be processed.",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Maximum number of characters in each chunk.",
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=200,
        help="Target number of overlapping characters between chunks.",
    )
    
    # read the command-line arguments
    args = parser.parse_args()

    # load the embedding model once for this processing session
    model = load_embedding_model()

    # process the document
    try:
        result = process_document(
            args.pdf_path,
            model=model,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
        )
    except ValueError as error:
        parser.error(str(error))

    page_count = len(result["pages"])
    chunk_count = len(result["chunks"])

    print(f"Pages extracted: {page_count}")
    print(f"Chunks created: {chunk_count}")
    print(f"Embedding matrix: {result['embeddings'].shape}")


if __name__ == "__main__":
    main()
    