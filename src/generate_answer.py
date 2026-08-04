import os

from dotenv import load_dotenv
from groq import Groq


# use one model name so it is easy to replace later
DEFAULT_MODEL = "llama-3.3-70b-versatile"


def generate_answer(
    prompt,
    model_name=DEFAULT_MODEL,
    client=None,
    json_mode=False,
):
    # reject a prompt that contains no useful text
    if not prompt.strip():
        raise ValueError("prompt must not be empty")

    # create a real Groq client when one was not supplied
    if client is None:
        # load values stored in the local .env file
        load_dotenv()

        # read the secret API key without placing it directly in the code
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError("GROQ_API_KEY was not found")

        client = Groq(api_key=api_key)

    # store the options that will be sent to Groq
    request_options = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "temperature": 0.1,
    }

    # require a JSON response when structured analysis is requested
    if json_mode:
        request_options["response_format"] = {
            "type": "json_object"
        }

    # send the grounded prompt to the language model
    completion = client.chat.completions.create(
        **request_options
    )

    # retrieve the text from the model's first response
    answer = completion.choices[0].message.content

    if not answer:
        raise ValueError("the model returned an empty answer")

    return answer.strip()
