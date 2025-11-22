# utils/llm.py
import json
import os
from dotenv import load_dotenv
from openai import OpenAI

# load .env so OPENAI_API_KEY is available
load_dotenv()

# If OPENAI_API_KEY is set, OpenAI() will pick it up automatically
client = OpenAI()


def call_llm(prompt: str, json_expected: bool = False) -> str:
    """
    Call an OpenAI chat model.
    - prompt: full text prompt (we already include system-style text inside it)
    - json_expected: if True, ask the model to return a single JSON object
    """
    try:
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",   
            messages=[
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"} if json_expected else None,
            temperature=0.2,
        )

        content = resp.choices[0].message.content or ""

        if json_expected:
            # Try to ensure we return valid JSON string
            try:
                json.loads(content)
            except Exception:
                # crude cleanup if model added text around JSON
                start = content.find("{")
                end = content.rfind("}")
                if start != -1 and end != -1 and end > start:
                    content = content[start : end + 1]
                else:
                    content = "{}"

        return content

    except Exception as e:
        print("LLM call failed:", e)
        # return a JSON error string so pydantic doesn't completely explode
        if json_expected:
            return json.dumps({"error": str(e)})
        return f"ERROR: {e}"
