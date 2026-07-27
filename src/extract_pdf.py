from pypdf import PdfReader

# open pdf file
reader = PdfReader("data/e-cigarette-review.pdf")

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