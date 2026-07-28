from pypdf import PdfWriter

from src.extract_pdf import extract_pages


def test_extracts_pages_from_pdf(tmp_path):
    # create a temporary path for the test PDF
    pdf_path = tmp_path / "test.pdf"

    # create a new PDF with two blank pages
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.add_blank_page(width=612, height=792)

    # save the temporary PDF so extract_pages can open it
    with pdf_path.open("wb") as output_file:
        writer.write(output_file)

    # run our extraction function on the temporary PDF
    extracted_pages = extract_pages(pdf_path)

    # check that two pages were extracted
    assert len(extracted_pages) == 2

    # check that the stored page numbers are 1 and 2
    assert extracted_pages[0]["page_number"] == 1
    assert extracted_pages[1]["page_number"] == 2

    # check that both blank pages contain empty text
    assert extracted_pages[0]["text"] == ""
    assert extracted_pages[1]["text"] == "" 