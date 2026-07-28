def remove_reference_content(extracted_pages):
    # create a list for pages that remain searchable
    cleaned_pages = []

    # track whether we are currently inside the reference section
    inside_references = False

    # go through pages in their original order
    for page_data in extracted_pages:
        # split the page text into individual lines
        lines = page_data['text'].splitlines()

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