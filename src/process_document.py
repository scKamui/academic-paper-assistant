import argparse
import json
from pathlib import Path

from src.chunk_text import chunk_pages
from src.clean_text import remove_reference_content, remove_repeated_lines
from src.embeddings import embed_chunks, load_embedding_model
from src.extract_pdf import extract_pages
from src.generate_answer import generate_answer
from src.prompt import (
    build_rag_prompt,
    build_structured_analysis_prompt,
)
from src.search import search_chunks
from src.structured_analysis import (
    generate_structured_analysis,
    retrieve_field_evidence,
    validate_structured_analysis,
    verify_structured_analysis,
)


def process_document(pdf_path, model, chunk_size=1000, overlap=200):
    # extract page-numbered text from the PDF
    extracted_pages = extract_pages(pdf_path)

    # remove headers and footers that repeat across several pages
    pages_without_repeated_lines = remove_repeated_lines(extracted_pages)

    # remove bibliography content before creating searchable chunks
    searchable_pages = remove_reference_content(
        pages_without_repeated_lines
    )

    # split the extracted pages into page-aware chunks
    chunks = chunk_pages(
        searchable_pages,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    # convert chunks into embedding vectors
    embeddings = embed_chunks(chunks, model)

    # return all the processed document data
    return {
        "pages": extracted_pages,
        "searchable_pages": searchable_pages,
        "chunks": chunks,
        "embeddings": embeddings,
    }


def main():
    # create a command-line parser and explain what the program does
    parser = argparse.ArgumentParser(
        description="Extract and analyze text from an academic PDF."
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

    # retrieve more evidence when analyzing several paper fields
    parser.add_argument(
        "--analysis-top-k",
        type=int,
        default=5,
        help="Number of passages to retrieve for each structured field.",
    )

    # optionally display the complete grounded prompt
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="Display the grounded prompt created from the retrieved passages.",
    )

    # allow the student to extract the main parts of the paper
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Extract the hypothesis, methodology, findings, and limitations.",
    )

    # read the command-line arguments
    args = parser.parse_args()

    if args.show_prompt and not args.query:
        parser.error("--show-prompt requires --query")

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
    searchable_page_count = len(result["searchable_pages"])
    chunk_count = len(result["chunks"])

    # display a summary of the processed document
    print(f"Pages extracted: {page_count}")
    print(f"Searchable pages: {searchable_page_count}")
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

        # build the grounded prompt using the retrieved passages
        prompt = build_rag_prompt(args.query, search_results)

        # display the prompt only when requested
        if args.show_prompt:
            print("\nGrounded prompt:")
            print("-" * 80)
            print(prompt)
            print("-" * 80)

        # send the grounded prompt to Groq and display its answer
        answer = generate_answer(prompt)

        print("\nGenerated answer:")
        print(answer)

    if args.analyze:
        # retrieve separate evidence for each part of the paper
        evidence_by_field = retrieve_field_evidence(
            chunks=result["chunks"],
            embeddings=result["embeddings"],
            embedding_model=model,
            top_k=args.analysis_top_k,
        )

        # build one grounded prompt from all the retrieved evidence
        structured_prompt = build_structured_analysis_prompt(
            evidence_by_field
        )

        # generate and parse the structured response
        try:
            analysis = generate_structured_analysis(
                structured_prompt
            )

            # check the response before displaying it to the student
            validate_structured_analysis(
                analysis,
                evidence_by_field,
            )

            # use a second model pass to narrow claims that exceed their evidence
            analysis = verify_structured_analysis(
                analysis,
                evidence_by_field,
            )

            # validate again in case the verifier changed a quote or page number
            validate_structured_analysis(
                analysis,
                evidence_by_field,
            )
        except ValueError as error:
            parser.error(str(error))

        print("\nStructured paper analysis:")
        print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
