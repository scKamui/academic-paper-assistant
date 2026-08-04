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
