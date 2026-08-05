import copy
import json

from src.generate_answer import generate_answer
from src.evidence_sources import build_source_index
from src.prompt import (
    build_analysis_verification_prompt,
    build_structured_analysis_prompt,
)
from src.search import search_chunks

# use a separate search question for each part of the paper
FIELD_QUERIES = {
    "document_type": (
        "What evidence identifies the document as an original study, review "
        "article, reference entry, or another type of academic text?"
    ),
    "hypothesis": (
        "What hypothesis, research question, aim, purpose, or objective do the "
        "authors state?"
    ),
    "methodology": (
        "What methods did the authors use, such as participants, data collection, "
        "analysis, experiments, or literature search and selection?"
    ),
    "findings": (
        "What main results, conclusions, arguments, or themes do the authors report?"
    ),
    "author_stated_limitations": (
        "What limitations, weaknesses, uncertainties, evidence gaps, or boundaries "
        "do the authors explicitly describe?"
    ),
}

# list the fields that every structured analysis must contain
REQUIRED_ANALYSIS_FIELDS = {
    "document_type",
    "hypothesis",
    "methodology",
    "findings",
    "author_stated_limitations",
    "ai_suggested_limitations",
}


def retrieve_field_evidence(
    chunks,
    embeddings,
    embedding_model,
    top_k=5,
):
    # create a dictionary for the evidence found for each field
    evidence_by_field = {}

    # search the paper separately for every structured field
    for field_name, query in FIELD_QUERIES.items():
        search_results = search_chunks(
            query=query,
            chunks=chunks,
            embeddings=embeddings,
            model=embedding_model,
            top_k=top_k,
        )

        evidence_by_field[field_name] = search_results

    return evidence_by_field


def analyze_document(chunks, embeddings, embedding_model, top_k=5, client=None):
    # retrieve different evidence for each part of the paper
    evidence_by_field = retrieve_field_evidence(
        chunks=chunks,
        embeddings=embeddings,
        embedding_model=embedding_model,
        top_k=top_k,
    )

    # create the first analysis using only the retrieved paper text
    prompt = build_structured_analysis_prompt(evidence_by_field)
    referenced_analysis = generate_structured_analysis(prompt, client=client)

    # replace the model's source IDs with the original page text from our search
    analysis = hydrate_analysis_evidence(
        referenced_analysis,
        evidence_by_field,
    )

    # check citations before asking the model to verify its claims
    validate_structured_analysis(analysis, evidence_by_field)

    # narrow or remove claims that are not fully supported
    verified_analysis = verify_structured_analysis(
        analysis,
        evidence_by_field,
        client=client,
    )

    # make sure the verification step did not create invalid output
    validate_structured_analysis(verified_analysis, evidence_by_field)

    return verified_analysis

def generate_structured_analysis(prompt, client=None):
    # ask the model to return the analysis as JSON
    raw_answer = generate_answer(
        prompt,
        client=client,
        json_mode=True,
    )

    # convert the JSON text into a Python dictionary
    try:
        analysis = json.loads(raw_answer)
    except json.JSONDecodeError as error:
        raise ValueError("the model returned invalid JSON") from error

    # the structured result must be one JSON object
    if not isinstance(analysis, dict):
        raise ValueError("the structured analysis must be a JSON object")

    return analysis


def hydrate_analysis_evidence(analysis, evidence_by_field):
    # make a copy so the raw model response is not changed in place
    hydrated_analysis = copy.deepcopy(analysis)
    source_index = build_source_index(evidence_by_field)

    def hydrate_evidence_list(evidence, field_name):
        if not isinstance(evidence, list):
            raise ValueError(f"{field_name} evidence must be a list")

        hydrated_evidence = []

        for evidence_item in evidence:
            if not isinstance(evidence_item, dict):
                raise ValueError(f"{field_name} evidence must contain objects")

            source_id = evidence_item.get("source_id")

            # reject IDs the model made up instead of silently trusting them
            if source_id not in source_index:
                raise ValueError(
                    f"{field_name} uses an unknown source ID: {source_id}"
                )

            source = source_index[source_id]
            hydrated_evidence.append({
                "page_number": source["page_number"],
                "passage": source["text"],
            })

        return hydrated_evidence

    document_type = hydrated_analysis.get("document_type")

    if isinstance(document_type, dict):
        document_type["evidence"] = hydrate_evidence_list(
            document_type.get("evidence"),
            "document_type",
        )

    for field_name in ("hypothesis", "methodology"):
        field_data = hydrated_analysis.get(field_name)

        if isinstance(field_data, dict):
            field_data["evidence"] = hydrate_evidence_list(
                field_data.get("evidence"),
                field_name,
            )

    for field_name in ("findings", "author_stated_limitations"):
        field_data = hydrated_analysis.get(field_name)

        if not isinstance(field_data, dict):
            continue

        items = field_data.get("items")

        if not isinstance(items, list):
            continue

        for item_number, item in enumerate(items, start=1):
            if isinstance(item, dict):
                item["evidence"] = hydrate_evidence_list(
                    item.get("evidence"),
                    f"{field_name} item {item_number}",
                )

    ai_limitations = hydrated_analysis.get("ai_suggested_limitations")

    if isinstance(ai_limitations, list):
        for item_number, item in enumerate(ai_limitations, start=1):
            if not isinstance(item, dict):
                continue

            source_ids = item.pop("based_on_source_ids", None)

            if not isinstance(source_ids, list):
                raise ValueError(
                    f"AI limitation {item_number} source IDs must be a list"
                )

            pages = []

            for source_id in source_ids:
                if source_id not in source_index:
                    raise ValueError(
                        f"AI limitation {item_number} uses an unknown source ID: "
                        f"{source_id}"
                    )

                pages.append(source_index[source_id]["page_number"])

            # remove duplicates while keeping the pages in their original order
            item["based_on_pages"] = list(dict.fromkeys(pages))

    return hydrated_analysis


def verify_structured_analysis(analysis, evidence_by_field, client=None):
    # ask the model to compare its first analysis with the original sources
    verification_prompt = build_analysis_verification_prompt(
        analysis,
        evidence_by_field,
    )

    # ask only for decisions so the model cannot replace citations or quotes
    decisions = generate_structured_analysis(
        verification_prompt,
        client=client,
    )

    # copy the original so its evidence remains unchanged
    verified_analysis = copy.deepcopy(analysis)

    document_decision = decisions.get("document_type")

    if not isinstance(document_decision, dict):
        raise ValueError("verification is missing the document_type decision")

    if document_decision.get("supported") is not True:
        raise ValueError("the document type was not supported by its evidence")

    revised_explanation = document_decision.get("revised_explanation")

    if not isinstance(revised_explanation, str) or not revised_explanation.strip():
        raise ValueError("verification returned an empty document explanation")

    verified_analysis["document_type"]["explanation"] = revised_explanation

    # update simple summaries without allowing the model to change their evidence
    for field_name in ("hypothesis", "methodology"):
        decision = decisions.get(field_name)

        if not isinstance(decision, dict):
            raise ValueError(f"verification is missing the {field_name} decision")

        if decision.get("supported") is False:
            verified_analysis[field_name]["found"] = False
            verified_analysis[field_name]["summary"] = (
                "Not found in the provided evidence."
            )
            verified_analysis[field_name]["evidence"] = []

            if field_name == "hypothesis":
                verified_analysis[field_name]["statement_type"] = "Not applicable"
        elif decision.get("supported") is True:
            revised_summary = decision.get("revised_summary")

            if not isinstance(revised_summary, str) or not revised_summary.strip():
                raise ValueError(
                    f"verification returned an empty {field_name} summary"
                )

            # only rewrite summaries that the first analysis actually found
            if verified_analysis[field_name]["found"]:
                verified_analysis[field_name]["summary"] = revised_summary
        else:
            raise ValueError(
                f"verification returned an invalid {field_name} decision"
            )

    # apply decisions by item number while preserving each item's evidence
    for field_name in ("findings", "author_stated_limitations"):
        original_items = verified_analysis[field_name]["items"]
        field_decisions = decisions.get(field_name)

        if not isinstance(field_decisions, list):
            raise ValueError(f"verification is missing {field_name} decisions")

        decisions_by_number = {
            decision.get("item_number"): decision
            for decision in field_decisions
            if isinstance(decision, dict)
        }
        checked_items = []

        for item_number, item in enumerate(original_items, start=1):
            decision = decisions_by_number.get(item_number)

            if not isinstance(decision, dict):
                raise ValueError(
                    f"verification is missing {field_name} item {item_number}"
                )

            if decision.get("supported") is False:
                continue

            if decision.get("supported") is not True:
                raise ValueError(
                    f"verification returned an invalid {field_name} item decision"
                )

            revised_claim = decision.get("revised_claim")

            if not isinstance(revised_claim, str) or not revised_claim.strip():
                raise ValueError(
                    f"verification returned an empty {field_name} claim"
                )

            item["claim"] = revised_claim
            checked_items.append(item)

        verified_analysis[field_name]["items"] = checked_items
        verified_analysis[field_name]["found"] = bool(checked_items)

    ai_decisions = decisions.get("ai_suggested_limitations")

    if not isinstance(ai_decisions, list):
        raise ValueError("verification is missing AI limitation decisions")

    ai_decisions_by_number = {
        decision.get("item_number"): decision
        for decision in ai_decisions
        if isinstance(decision, dict)
    }
    checked_ai_limitations = []

    for item_number, item in enumerate(
        verified_analysis["ai_suggested_limitations"],
        start=1,
    ):
        decision = ai_decisions_by_number.get(item_number)

        if not isinstance(decision, dict):
            raise ValueError(
                f"verification is missing AI limitation {item_number}"
            )

        if decision.get("supported") is False:
            continue

        if decision.get("supported") is not True:
            raise ValueError("verification returned an invalid AI limitation decision")

        revised_suggestion = decision.get("revised_suggestion")
        revised_reason = decision.get("revised_reason")

        if not isinstance(revised_suggestion, str) or not revised_suggestion.strip():
            raise ValueError("verification returned an empty AI suggestion")

        if not isinstance(revised_reason, str) or not revised_reason.strip():
            raise ValueError("verification returned an empty AI limitation reason")

        item["suggestion"] = revised_suggestion
        item["reason"] = revised_reason
        checked_ai_limitations.append(item)

    verified_analysis["ai_suggested_limitations"] = checked_ai_limitations

    return verified_analysis


def normalize_evidence_text(text):
    # ignore differences in capitalization and extra spacing when comparing text
    return " ".join(text.lower().split())


def validate_evidence_list(evidence, field_name, source_text_by_page):
    # every evidence value must be a list, even when it is empty
    if not isinstance(evidence, list):
        raise ValueError(f"{field_name} evidence must be a list")

    for evidence_item in evidence:
        if not isinstance(evidence_item, dict):
            raise ValueError(f"{field_name} evidence must contain objects")

        page_number = evidence_item.get("page_number")
        passage = evidence_item.get("passage")

        # make sure the citation uses a real retrieved PDF page
        if not isinstance(page_number, int) or page_number not in source_text_by_page:
            raise ValueError(
                f"{field_name} cites PDF page {page_number}, which was not retrieved"
            )

        if not isinstance(passage, str) or not passage.strip():
            raise ValueError(f"{field_name} contains an empty evidence passage")

        normalized_passage = normalize_evidence_text(passage)
        page_sources = source_text_by_page[page_number]

        # confirm that the quoted passage really appears in a source from this page
        if not any(normalized_passage in source for source in page_sources):
            raise ValueError(
                f"{field_name} contains a passage that was not found on "
                f"PDF page {page_number}"
            )


def validate_structured_analysis(analysis, evidence_by_field):
    # stop if the model left out any part of the required response
    missing_fields = REQUIRED_ANALYSIS_FIELDS - set(analysis.keys())

    if missing_fields:
        missing_names = ", ".join(sorted(missing_fields))
        raise ValueError(f"the structured analysis is missing: {missing_names}")

    # group all retrieved text by its original PDF page number
    source_text_by_page = {}

    for search_results in evidence_by_field.values():
        for result in search_results:
            page_number = result["page_number"]
            normalized_text = normalize_evidence_text(result["text"])
            source_text_by_page.setdefault(page_number, []).append(normalized_text)

    document_type = analysis["document_type"]

    if not isinstance(document_type, dict):
        raise ValueError("document_type must be an object")

    allowed_document_types = (
        "review_article",
        "original_study",
        "reference_entry",
        "other_academic_text",
    )

    if document_type.get("type") not in allowed_document_types:
        raise ValueError(
            "document_type must be review_article, original_study, "
            "reference_entry, or other_academic_text"
        )

    if not isinstance(document_type.get("explanation"), str) or not document_type["explanation"].strip():
        raise ValueError("document_type needs an explanation")

    document_evidence = document_type.get("evidence")
    validate_evidence_list(
        document_evidence,
        "document_type",
        source_text_by_page,
    )

    if not document_evidence:
        raise ValueError("document_type needs supporting evidence")

    # hypothesis and methodology both use a summary and one evidence list
    for field_name in ("hypothesis", "methodology"):
        field_data = analysis[field_name]

        if not isinstance(field_data, dict):
            raise ValueError(f"{field_name} must be an object")

        if not isinstance(field_data.get("found"), bool):
            raise ValueError(f"{field_name} found must be true or false")

        if not isinstance(field_data.get("summary"), str) or not field_data["summary"].strip():
            raise ValueError(f"{field_name} summary must be text")

        evidence = field_data.get("evidence")
        validate_evidence_list(evidence, field_name, source_text_by_page)

        # a missing field should not claim to have supporting evidence
        if field_data["found"] is False and evidence:
            raise ValueError(f"{field_name} cannot have evidence when found is false")

        if field_data["found"] is True and not evidence:
            raise ValueError(f"{field_name} needs evidence when found is true")

    # findings and author limitations contain separate claim objects
    for field_name in ("findings", "author_stated_limitations"):
        field_data = analysis[field_name]

        if not isinstance(field_data, dict):
            raise ValueError(f"{field_name} must be an object")

        if not isinstance(field_data.get("found"), bool):
            raise ValueError(f"{field_name} found must be true or false")

        items = field_data.get("items")

        if not isinstance(items, list):
            raise ValueError(f"{field_name} items must be a list")

        if field_data["found"] is False and items:
            raise ValueError(f"{field_name} cannot have items when found is false")

        if field_data["found"] is True and not items:
            raise ValueError(f"{field_name} needs items when found is true")

        for item_number, item in enumerate(items, start=1):
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("claim"), str)
                or not item["claim"].strip()
            ):
                raise ValueError(f"{field_name} item {item_number} needs a claim")

            item_evidence = item.get("evidence")
            validate_evidence_list(
                item_evidence,
                f"{field_name} item {item_number}",
                source_text_by_page,
            )

            if not item_evidence:
                raise ValueError(f"{field_name} item {item_number} needs evidence")

    ai_limitations = analysis["ai_suggested_limitations"]

    if not isinstance(ai_limitations, list):
        raise ValueError("ai_suggested_limitations must be a list")

    for item_number, item in enumerate(ai_limitations, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"AI limitation {item_number} must be an object")

        if not isinstance(item.get("suggestion"), str) or not item["suggestion"].strip():
            raise ValueError(f"AI limitation {item_number} needs a suggestion")

        if not isinstance(item.get("reason"), str) or not item["reason"].strip():
            raise ValueError(f"AI limitation {item_number} needs a reason")

        based_on_pages = item.get("based_on_pages")

        if not isinstance(based_on_pages, list):
            raise ValueError(f"AI limitation {item_number} pages must be a list")

        if not based_on_pages:
            raise ValueError(f"AI limitation {item_number} needs supporting pages")

        # inferred limitations may only refer to pages supplied to the model
        for page_number in based_on_pages:
            if not isinstance(page_number, int) or page_number not in source_text_by_page:
                raise ValueError(
                    f"AI limitation {item_number} uses PDF page {page_number}, "
                    "which was not retrieved"
                )

    return analysis
