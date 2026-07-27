
## PROBLEM
-   Academic papers are long, dense, and written using specialized language.
-   Students often need to locate specific information under time constraints.
-   Important details may be scattered across abstracts, methods, results, and discussion sections.

## USER
- The primary users are college and university students who need to understand academic research papers for coursework. However, It could be used by anybody who needs to read, understand and or summarize a research paper or academic article. 

## Initial Scope

The first version will:

- Process one English-language PDF at a time.
- Support PDFs containing selectable text.
- Extract the research question or hypothesis, methodology, findings, and limitations.
- Return “Not stated in the paper” when information cannot be located.
- Provide page-level evidence for extracted information and answers.

## CORE FEATURES 
-   Upload a text-based PDF.
-   Extract its text while retaining page numbers.
-   Identify the hypothesis, methodology, findings, and limitations.
-   Show supporting passages and page citations.
-   Let the student ask questions about the paper.
-   Answer using only information found in the uploaded document.

## TRUST REQUIREMENTS 
-   Every extracted claim must include supporting evidence and a page number.
-   The system must say when information cannot be found.
-   It must distinguish author-stated limitations from AI-suggested limitations.
-   Uploaded papers should not be permanently retained without permission.
-   The application assists reading; it does not replace reading or guarantee academic correctness.
-   Students should verify important claims against the original paper.


## Personal Motivation
I started this project after experiencing how time-consuming it can be to find and understand important information in dense academic papers. I wanted to explore whether an evidence-grounded assistant could make the reading process more approachable while still directing students back to the original source.