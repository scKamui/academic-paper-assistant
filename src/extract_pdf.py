import sys
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError


if len(sys.argv) < 2:
    print("Usage: python src/extract_pdf.py <pdf_path>")
    sys.exit(1)

# Open the PDF file supplied by the user
pdf_path = Path(sys.argv[1])

# Stop early with a clear error if the supplied path is missing or is not a file.
if not pdf_path.is_file():
    print(f"Error: The file '{pdf_path}' does not exist or is not a valid file.")
    sys.exit(1)

# Reject files that do not have a PDF extension.
if pdf_path.suffix.lower() != ".pdf":
    print(f"Error: The file '{pdf_path}' is not a PDF.")
    sys.exit(1)

# Catch files that have a PDF extension but contain invalid or corrupted data.
try:
    reader = PdfReader(pdf_path)
except PdfReadError:
    print(f"Error: The file '{pdf_path}' could not be read as a valid PDF.")
    sys.exit(1)


# store extracted text from each page in a list
extracted_pages = []

for index, page in enumerate(reader.pages):
    # extract text from each page
    text = page.extract_text() or ""
    # print the extracted text
    page_number = index + 1

    # create a dictionary and append it to the list
    page_data = {
        "page_number": page_number,
        "text": text
    }
    extracted_pages.append(page_data)

    print(f"Page {page_number}")
    print(f"Characters: {len(text)}")
    print(f"Preview: {text[:150]}")
    print()

print(f"Total pages stored: {len(extracted_pages)}")