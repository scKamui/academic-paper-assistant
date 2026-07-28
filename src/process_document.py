import argparse
from pathlib import Path

from src.chunk_text import chunk_pages
from src.extract_pdf import extract_pages


def process_document(pdf_path, chunk_size=1000, overlap=200):
    # extract page-numbered text from the PDF
    extracted_pages = extract_pages(pdf_path)

    # split the extracted pages into page-aware chunks
    chunks = chunk_pages(extracted_pages, chunk_size=chunk_size, overlap=overlap)

    # return both results in one dictionary
    return {
        "pages": extracted_pages,
        "chunks": chunks
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

    # read the command-line arguments
    args = parser.parse_args()

    # process the document
    result = process_document(args.pdf_path)

    page_count = len(result["pages"])
    chunk_count = len(result["chunks"])

    print(f"Pages extracted: {page_count}")
    print(f"Chunks created: {chunk_count}")


if __name__ == "__main__":
    main()
    