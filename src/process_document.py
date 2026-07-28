import argparse
from pathlib import Path

from src.chunk_text import chunk_pages
from src.extract_pdf import extract_pages
from src.embeddings import embed_chunks, load_embedding_model
from src.search import search_chunks


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

    # allow the student to search the processed paper
    parser.add_argument(
        "--query",
        type=str,
        help="Question or topic to search for in the PDF.",
    )

    # control how many matching passages are returned
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of matching passages to return.",
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

    if args.query:
        try:
            search_results = search_chunks(
                query=args.query,
                chunks=result["chunks"],
                embeddings=result["embeddings"],
                model=model,
                top_k=args.top_k,
            )
        except ValueError as error:
            parser.error(str(error))

        print("\nTop matching passages:")

        for rank, search_result in enumerate(search_results, start=1):
            print(
                f"\n{rank}. Page {search_result['page_number']}, "
                f"chunk {search_result['chunk_number']}, "
                f"similarity {search_result['score']:.3f}"
            )
            print(search_result["text"][:500])


if __name__ == "__main__":
    main()
    