def build_source_index(evidence_by_field):
    # give every retrieved passage its own ID so the model can point to it
    source_index = {}

    for field_name, search_results in evidence_by_field.items():
        for source_number, result in enumerate(search_results, start=1):
            source_id = f"{field_name}-{source_number}"
            source_index[source_id] = result

    return source_index
