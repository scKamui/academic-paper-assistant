from unittest.mock import MagicMock

import pytest

from src.generate_answer import DEFAULT_MODEL, generate_answer


def test_generates_answer_using_fake_client():
    # create a fake Groq client and fake model response
    fake_client = MagicMock()
    fake_completion = MagicMock()
    fake_completion.choices[0].message.content = "Grounded answer [PDF page 6]."

    # make the fake client return the fake response
    fake_client.chat.completions.create.return_value = fake_completion

    # generate an answer without making a real API request
    answer = generate_answer(
        "Use the evidence to answer the question.",
        client=fake_client,
    )

    # check that the response text was returned
    assert answer == "Grounded answer [PDF page 6]."

    # check that the correct request was sent to the fake client
    fake_client.chat.completions.create.assert_called_once_with(
        model=DEFAULT_MODEL,
        messages=[
            {
                "role": "user",
                "content": "Use the evidence to answer the question.",
            }
        ],
        temperature=0.1,
    )


def test_rejects_empty_prompt():
    # an empty question should fail before contacting the API
    with pytest.raises(ValueError, match="prompt must not be empty"):
        generate_answer("   ")


def test_rejects_empty_model_answer():
    # create a fake response containing no answer
    fake_client = MagicMock()
    fake_completion = MagicMock()
    fake_completion.choices[0].message.content = None
    fake_client.chat.completions.create.return_value = fake_completion

    with pytest.raises(ValueError, match="the model returned an empty answer"):
        generate_answer("Answer this question.", client=fake_client)