from pathlib import Path
import re
from typing import Dict, List

from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "extracted_cases"
OUT_DIR = BASE_DIR / "data" / "cases_faiss"
OUT_DIR.mkdir(exist_ok=True)


embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# SemanticChunker using that embedding model
chunker = SemanticChunker(
    embedding,
    breakpoint_threshold_type="percentile",  
    # breakpoint_threshold_amount=90,    
)


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
    ],
}


def clean_text(text: str) -> str:
    # normalize newlines 
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def find_section_positions(text: str) -> Dict[str, int]:
    """
    Find approximate starting index of each section heading in uppercase text.
    Returns { section_key: start_index }
    """
    upper = text.upper()
    positions = {}

    for key, patterns in SECTION_HEADINGS.items():
        for pat in patterns:
            m = re.search(pat, upper)
            if m:
                if key not in positions or m.start() < positions[key]:
                    positions[key] = m.start()
                break

    return positions


def slice_sections(text: str, positions: Dict[str, int]) -> Dict[str, str]:
    """
    Slice the full text into sections using heading positions.
    Returns { section_key: section_text }.
    """
    sections = {k: "" for k in SECTION_HEADINGS.keys()}
    if not positions:
        return sections

    # Sort by position
    ordered = sorted(positions.items(), key=lambda x: x[1])

    for i, (key, start) in enumerate(ordered):
        end = ordered[i + 1][1] if i + 1 < len(ordered) else len(text)

        # Find end of heading line in original text
        newline_idx = text.find("\n", start)
        if newline_idx == -1:
            heading_end = start
        else:
            heading_end = newline_idx + 1

        raw_section = text[heading_end:end]
        sections[key] = clean_text(raw_section)

    return sections


def chunk_section(case_id: str, file_name: str, section_key: str, section_text: str) -> List[Document]:
    """
    Use SemanticChunker to split a section into semantically coherent chunks.
    Returns a list of LangChain Document objects with metadata.
    """
    section_text = section_text.strip()
    if not section_text:
        return []

    # SemanticChunker expects a single long string (or list), returns docs
    docs = chunker.create_documents([section_text])

    # Attach metadata to each resulting document
    for i, d in enumerate(docs):
        d.metadata["source"] = "case"
        d.metadata["case_id"] = case_id
        d.metadata["file_name"] = file_name
        d.metadata["section"] = section_key
        d.metadata["chunk_index"] = i

    return docs


def ingest_cases_to_faiss():
    all_docs: List[Document] = []

    txt_files = sorted(RAW_DIR.glob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in {RAW_DIR}")
        return

    for txt_path in txt_files:
        print(f"\nProcessing {txt_path.name}...")
        raw_text = txt_path.read_text(encoding="utf-8", errors="ignore")
        raw_text = clean_text(raw_text)

        case_id = txt_path.stem.replace("_raw", "")
        file_name = txt_path.name

        positions = find_section_positions(raw_text)
        sections = slice_sections(raw_text, positions)

        for section_key, section_text in sections.items():
            print(f"  - Chunking section '{section_key}' for case {case_id}...")
            section_docs = chunk_section(case_id, file_name, section_key, section_text)
            all_docs.extend(section_docs)

    print(f"\nTotal case chunks created: {len(all_docs)}")

    if not all_docs:
        print("No documents created, nothing to index.")
        return

    # Build FAISS index from these documents
    vectorstore = FAISS.from_documents(all_docs, embedding)

    # Save FAISS index to disk
    index_path = OUT_DIR / "cases_faiss_index"
    vectorstore.save_local(str(index_path))

    print(f"FAISS index saved to {index_path}")


if __name__ == "__main__":
    ingest_cases_to_faiss()
