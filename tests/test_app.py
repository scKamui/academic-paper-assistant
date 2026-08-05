from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_app_starts_without_an_error():
    # move from the tests folder to the app file in the project folder
    app_path = Path(__file__).parent.parent / "app.py"

    # run the interface without opening a browser or contacting the API
    app = AppTest.from_file(app_path).run(timeout=30)

    # check that Streamlit displayed the main page successfully
    assert not app.exception
    assert app.title[0].value == "CiteBack"
    assert app.file_uploader[0].label == "Start with a research paper"

    # check that students are warned about hosted AI processing
    assert any(
        "relevant text passages are sent" in caption.value
        for caption in app.caption
    )


def test_app_displays_question_history_without_an_error():
    # move from the tests folder to the app file in the project folder
    app_path = Path(__file__).parent.parent / "app.py"
    app = AppTest.from_file(app_path)

    # imitate a processed document without loading the real embedding model
    app.session_state["file_name"] = "sample-paper.pdf"
    app.session_state["document"] = {
        "pages": [{"page_number": 1, "text": "Sample paper text."}],
        "searchable_pages": [
            {"page_number": 1, "text": "Sample paper text."}
        ],
        "chunks": [
            {
                "page_number": 1,
                "chunk_number": 1,
                "text": "Sample paper text.",
            }
        ],
        "embeddings": [],
    }
    app.session_state["question_history"] = [
        {
            "question": "What is this paper about?",
            "answer": "It contains sample paper text [PDF page 1].",
            "sources": [
                {
                    "page_number": 1,
                    "chunk_number": 1,
                    "text": "Sample paper text.",
                    "score": 0.9,
                }
            ],
        }
    ]

    app.run(timeout=30)

    # check that both chat messages render instead of crashing the interface
    assert not app.exception
    assert len(app.chat_message) == 2
