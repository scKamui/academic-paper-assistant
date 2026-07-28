import re
from collections import Counter

def normalize_repeated_line(line):
    # normalize spacing and capitalization for comparison
    normalized_line = line.strip().lower()

    # replace changing page numbers with one common placeholder
    normalized_line = re.sub(
        r"\bpage\s+\d+\b",
        "page #",
        normalized_line,
    )

    return normalized_line

def remove_repeated_lines(extracted_pages, min_occurrences=3):
    # count how many different pages contain each normalized line
    line_counts = Counter()

    for page_data in extracted_pages:
        lines_on_page = {
            normalize_repeated_line(line)
            for line in page_data["text"].splitlines()
            if line.strip()
        }
        line_counts.update(lines_on_page)

    # identify lines that appear on at least the minimum number of pages
    repeated_lines = {
        line
        for line, count in line_counts.items()
        if count >= min_occurrences
    }

    # rebuild every page without the repeated lines
    cleaned_pages = []

    for page_data in extracted_pages:
        kept_lines = [
            line
            for line in page_data["text"].splitlines()
            if normalize_repeated_line(line) not in repeated_lines
        ]

        cleaned_pages.append({
            **page_data,
            "text": "\n".join(kept_lines).strip(),
        })

    return cleaned_pages


def remove_reference_content(extracted_pages):
    # create a list for pages that remain searchable
    cleaned_pages = []

    # track whether we are currently inside the reference section
    inside_references = False

    # go through pages in their original order
    for page_data in extracted_pages:
        # split the page text into individual lines
        lines = page_data["text"].splitlines()

        # if currently inside references:
        if inside_references:
            # find the first non-empty line
            first_line = next(
                (line.strip().lower() for line in lines if line.strip()), 
                ""
            )
            # if it begins with Figure, Table, Appendix, or Supplement:
            if first_line.startswith(("figure", "table", "appendix", "supplement")):
                # leave reference mode
                inside_references = False
            # otherwise skip this page
            else:
                continue

        reference_index = next(
            (
                index
                for index, line in enumerate(lines)
                if line.strip().lower() in ("references", "bibliography")
            ),
            None,
        )

        if reference_index is not None:
            # keep any useful content that appears before the reference heading
            text_before_references = "\n".join(lines[:reference_index]).strip()

            if text_before_references:
                cleaned_pages.append({
                    **page_data,
                    "text": text_before_references,
                })

            inside_references = True
        else:
            cleaned_pages.append(page_data)

    return cleaned_pages