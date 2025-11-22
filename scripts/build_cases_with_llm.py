import os
import re
import json
import time
from pathlib import Path

from openai import OpenAI

# ---------------- CONFIG ----------------

MODEL_NAME = "gpt-4-turbo-mini"  # You can change this later if you want

# Project-relative paths (portable)
BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "extracted_cases"
OUT_DIR = BASE_DIR / "data" / "cases_structured_llm"
OUT_DIR.mkdir(exist_ok=True)

# Set your API key in the environment before running:
#   setx OPENAI_API_KEY "your-key-here"  (Windows)
client = OpenAI()

# Canonical section keys and possible headings (in UPPERCASE)
SECTION_HEADINGS = {
    "executive_summary": [
        r"EXECUTIVE SUMMARY",
        r"SUMMARY",
        r"ABSTRACT",
    ],
    "incident_description": [
        r"INCIDENT DESCRIPTION",
        r"DESCRIPTION OF INCIDENT",
        r"DESCRIPTION OF THE INCIDENT",
        r"ACCIDENT DESCRIPTION",
        r"BACKGROUND OF THE INCIDENT",
    ],
    "technical_analysis": [
        r"TECHNICAL ANALYSIS",
        r"TECHNICAL DISCUSSION",
        r"CAUSE ANALYSIS",
        r"ANALYSIS",
    ],
    "safety_issues": [
        r"SAFETY ISSUES",
        r"SAFETY MANAGEMENT",
        r"CONTRIBUTING FACTORS",
        r"CAUSAL FACTORS",
        r"ROOT CAUSES",
    ],
    "recommendations": [
        r"RECOMMENDATIONS",
        r"SAFETY RECOMMENDATIONS",
        r"CORRECTIVE ACTIONS",
        r"PREVENTIVE ACTIONS",
        r"PREVENTION MEASURES",
    ],
    "key_lessons": [
        r"KEY LESSONS",
        r"LESSONS LEARNED",
        r"KEY LESSONS FOR THE INDUSTRY",
        r"KEY LESSONS FOR INDUSTRY",
        r"KEY LESSONS FOR THE PETROLEUM INDUSTRY",
    ],
}

# How we describe each section to the LLM
SECTION_STYLES = {
    "executive_summary": "a clear, concise paragraph summarizing the incident and its outcome",
    "incident_description": "a chronological description of what happened",
    "technical_analysis": "a technical explanation of causes, mechanisms, and contributing factors",
    "safety_issues": "a short paragraph or plain-text bullet list of safety issues and failures",
    "recommendations": "a plain-text bullet list of recommended actions",
    "key_lessons": "a plain-text bullet list of key lessons learned",
}

SYSTEM_PROMPT = (
    "You are helping extract structured information from confidential industrial "
    "incident investigation reports. You must:\n"
    "- Use ONLY the information in the provided section text.\n"
    "- Do NOT invent or assume facts that are not clearly present.\n"
    "- Rewrite the text in clear professional English.\n"
    "- Preserve technical accuracy and important details.\n"
    "- Return only the rewritten text, with no extra commentary or headings."
)

USER_PROMPT_TEMPLATE = (
    "Here is the {section_name} section from an incident investigation report.\n\n"
    "Task: Rewrite it as {style}. Preserve all key facts, causes, and recommendations "
    "that appear in the text, but remove formatting noise and repetitions. "
    "Do NOT add any information that is not in the text.\n\n"
    "--- BEGIN SECTION TEXT ---\n"
    "{section_text}\n"
    "--- END SECTION TEXT ---"
)

# ------------- TEXT UTILITIES -------------


def clean_text(text: str) -> str:
    """Remove weird characters and collapse extra blank lines."""
    # Remove non-printable characters
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E]", " ", text)
    # Convert Windows newlines if needed
    text = text.replace("\r\n", "\n")
    # Collapse multiple blank lines
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def find_section_positions(text: str):
    """
    Find approximate positions of key headings in the full report text.

    Returns a dict:
      { section_key: (start_index_in_original_text, heading_string_found) }
    """
    upper = text.upper()
    positions = {}

    for key, patterns in SECTION_HEADINGS.items():
        for pat in patterns:
            m = re.search(pat, upper)
            if m:
                # keep the earliest match for this key
                if key not in positions or m.start() < positions[key][0]:
                    positions[key] = (m.start(), m.group(0))
                break  # stop at first pattern that matches for this key

    return positions


def slice_sections(text: str, section_positions: dict):
    """
    Given full text and the heading positions, slice it into sections.

    Returns a dict { section_key: section_text }.
    Missing sections get "".
    """
    sections = {key: "" for key in SECTION_HEADINGS.keys()}

    if not section_positions:
        return sections

    # Build sorted list: (start_index, section_key, heading_text)
    items = sorted(
        [(pos, key, heading) for key, (pos, heading) in section_positions.items()],
        key=lambda x: x[0],
    )

    for i, (start, key, heading) in enumerate(items):
        # Section ends at next heading or end of text
        end = items[i + 1][0] if i + 1 < len(items) else len(text)

        # Find the end of the heading line in original text
        newline_idx = text.find("\n", start)
        if newline_idx == -1:
            heading_end = start + len(heading)
        else:
            heading_end = newline_idx + 1

        raw_section = text[heading_end:end]
        sections[key] = clean_text(raw_section)

    return sections


# ------------- LLM CALLER -------------


def run_llm_section(section_key: str, section_text: str) -> str:
    """
    Call the LLM for a single section and return cleaned text.
    If the section_text is empty or very short, return it as-is.
    """
    section_text = section_text.strip()
    if not section_text:
        return ""

    # Truncate very long sections for stability (roughly 3000 chars)
    MAX_CHARS = 3000
    if len(section_text) > MAX_CHARS:
        section_text = section_text[:MAX_CHARS]

    style = SECTION_STYLES.get(section_key, "a clear, concise summary")
    section_name_nice = section_key.replace("_", " ").title()

    user_prompt = USER_PROMPT_TEMPLATE.format(
        section_name=section_name_nice,
        style=style,
        section_text=section_text,
    )

    # Simple retry logic for transient errors
    for attempt in range(3):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=800,
            )
            content = completion.choices[0].message.content
            return content.strip()
        except Exception as e:
            print(f"[LLM {section_key} attempt {attempt+1}] Error: {e}")
            if attempt < 2:
                print("Retrying in 2 seconds…")
                time.sleep(2)
            else:
                print(f"Giving up on section {section_key}, returning raw text.")
                # Fall back to the raw cleaned section text so we don't lose data
                return section_text


# ------------- MAIN PIPELINE -------------


def process_case_file(txt_path: Path) -> dict:
    """Process a single *_raw.txt file -> structured case dict."""
    raw_text = txt_path.read_text(encoding="utf-8", errors="ignore")
    raw_text = clean_text(raw_text)

    case_id = txt_path.stem.replace("_raw", "")
    title = case_id  # you can manually refine titles later if needed

    print(f"  - Finding sections in {case_id}...")
    positions = find_section_positions(raw_text)
    sections_raw = slice_sections(raw_text, positions)

    # Call LLM per section
    structured_sections = {}
    for key, raw_section_text in sections_raw.items():
        print(f"    > Cleaning section '{key}' with LLM...")
        cleaned = run_llm_section(key, raw_section_text)
        structured_sections[key] = cleaned

    case_data = {
        "case_id": case_id,
        "title": title,
        **structured_sections,
    }

    return case_data


def main():
    all_cases = []

    txt_files = sorted(RAW_DIR.glob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in {RAW_DIR}. Did you run the PDF extractor?")
        return

    print(f"Found {len(txt_files)} raw case files in {RAW_DIR}.")

    for txt_path in txt_files:
        print(f"\nProcessing {txt_path.name} ...")
        try:
            case_data = process_case_file(txt_path)
        except Exception as e:
            print(f"[ERROR] Failed to process {txt_path.name}: {e}")
            # still write a minimal placeholder so the pipeline doesn't break
            case_id = txt_path.stem.replace("_raw", "")
            case_data = {
                "case_id": case_id,
                "title": case_id,
                "executive_summary": "",
                "incident_description": "",
                "technical_analysis": "",
                "safety_issues": "",
                "recommendations": "",
                "key_lessons": "",
            }

        # Save per-case JSON
        out_path = OUT_DIR / f"{case_data['case_id']}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(case_data, f, ensure_ascii=False, indent=2)

        all_cases.append(case_data)

    # Save combined cases.json
    combined_path = OUT_DIR / "cases.json"
    with combined_path.open("w", encoding="utf-8") as f:
        json.dump(all_cases, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Saved {len(all_cases)} cases to {combined_path}")


if __name__ == "__main__":
    main()
