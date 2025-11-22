import os 
import json
from pathlib import Path
import re
import time
from openai import OpenAI 

client = OpenAI()

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "extracted_cases"
OUT_DIR = BASE_DIR / "data" / "cases_structured"

OUT_DIR.mkdir(exist_ok=True)

SYSTEM_PROMPT = """You are helping extract structured information from an industrial incident investigation report.

You MUST:
- Use ONLY the information in the provided text.
- NOT invent or assume facts that are not in the text.
- Copy or lightly compress sentences from the report; do not hallucinate.

Return a single JSON object with this exact schema:

{
  "case_id": "",
  "title": "",
  "executive_summary": "",
  "incident_description": "",
  "technical_analysis": "",
  "safety_issues": "",
  "recommendations": "",
  "key_lessons": ""
}

Rules:
- If a field is not clearly present in the text, set it to an empty string "".
- Use full sentences, not bullet labels like "Executive Summary:" in the values.
- Respond with VALID JSON ONLY, no extra comments, explanations, or markdown.
"""



USER_PROMPT_TEMPLATE = """Extract the fields from the following incident report text.

--- BEGIN REPORT TEXT ---
{report_text}
--- END REPORT TEXT ---
"""


def clean_text(text: str) -> str:
    # remove non-printable characters
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E]", " ", text)
    # collapse multiple blank lines
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text


def call_llm_for_case(text: str, case_id: str):
    """Call the LLM and return the parsed JSON dict"""

    text = clean_text(text)

    MAX_CHARS = 4000
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]

    user_prompt = USER_PROMPT_TEMPLATE.format(report_text=text)

    for attempt in range(3):
        try:
            response = client.responses.create(
                model="gpt-4.1-mini",
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_output_tokens=1200,
            )
            raw_output = response.output[0].content[0].text
            data = json.loads(raw_output)
            data["case_id"] = case_id
            return data

        except Exception as e:
            print(f"[Attempt {attempt+1}] Error: {e}")
            if attempt < 2:
                print("Retrying in 2 seconds…")
                time.sleep(2)
            else:
                raise RuntimeError(f"Failed after 3 attempts for case {case_id}") from e

def main():
    all_cases = []

    for txt_path in RAW_DIR.glob("*.txt"):
        print(f"Processing {txt_path.name}...")
        text = txt_path.read_text(encoding="utf-8", errors="ignore")

        case_id = txt_path.stem.replace("_raw", "")

        try:
            case_data = call_llm_for_case(text, case_id)
        except Exception as e:
            print(f"[ERROR] Skipping {case_id} due to repeated failures: {e} ")
            # creating a placeholder 
            case_data = {
            "case_id": case_id,
            "title": case_id,
            "executive_summary": "",
            "incident_description": "",
            "technical_analysis": "",
            "safety_issues": "",
            "recommendations": "",
            "key_lessons": ""
        }
        
        out_path = OUT_DIR / f"{case_id}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(case_data, f, ensure_ascii=False, indent=2)

        all_cases.append(case_data)


    # save combine cases in one JSON
    compined_path = OUT_DIR / "cases.json"
    with compined_path.open("w", encoding="utf-8") as f:
        json.dump(all_cases, f, ensure_ascii=False, indent=2)

        print(f"Saved {len(all_cases)} cases to {compined_path}")

if __name__ == "__main__":
    main()

        

