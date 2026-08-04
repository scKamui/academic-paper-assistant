import os

from dotenv import load_dotenv
from groq import Groq


# use one model name so it is easy to replace later
DEFAULT_MODEL = "llama-3.3-70b-versatile"


def generate_answer(prompt, model_name=DEFAULT_MODEL):
    # reject a prompt that contains no useful text
    if not prompt.strip():
        raise ValueError("prompt must not be empty")

    # load values stored in the local .env file
    load_dotenv()

    # read the secret API key without placing it directly in the code
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY was not found")

    # create a client that can communicate with Groq
    client = Groq(api_key=api_key)

    # send the grounded prompt to the hosted language model
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0.1,
    )

    # retrieve the text from the model's first response
    answer = completion.choices[0].message.content

    if not answer:
        raise ValueError("the model returned an empty answer")

    return answer.strip()
