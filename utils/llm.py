# utils/llm.py
import json
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
print(bool(os.getenv("OPENAI_API_KEY")))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def call_llm(prompt: str, json_expected: bool = False) -> str:
    """
    Calls an OpenAI model. If json_expected=True, request structured JSON output.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # or "gpt-4o" if you have access
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"} if json_expected else None,
            temperature=0.2,
        )

        content = response.choices[0].message.content or ""

        # Optional: if expecting JSON, validate
        if json_expected:
            try:
                json.loads(content)
            except Exception:
                start = content.find("{")
                end = content.rfind("}")
                content = content[start:end+1] if start != -1 and end != -1 else "{}"

        return content

    except Exception as e:
        print("LLM call failed:", e)
        return json.dumps({"error": str(e)})
    
