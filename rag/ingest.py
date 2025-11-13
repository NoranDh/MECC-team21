# rag/ingest.py
from pathlib import Path
import pdfplumber, json, re
from tqdm import tqdm

DATA_DIR = Path("data")
CASES_DIR = DATA_DIR / "cases"
HB_DIR    = DATA_DIR / "handbook"
OUT_JSONL = DATA_DIR / "rag_corpus.jsonl"

def read_pdf(path: Path) -> str:
    parts = []
    with pdfplumber.open(str(path)) as pdf:
        for p in pdf.pages:
            txt = p.extract_text() or ""
            parts.append(txt)
    return "\n".join(parts)

def chunk(text: str, max_chars: int = 800):
    # split on double newlines then pack paragraphs
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks = []
    cur = ""
    for p in paras:
        if len(cur) + len(p) + 2 <= max_chars:
            cur = (cur + "\n\n" + p) if cur else p
        else:
            if cur:
                chunks.append(cur)
            cur = p
    if cur:
        chunks.append(cur)
    return chunks

def ingest_dir(dir_path: Path, kind: str, rows: list):
    for pdf in tqdm(sorted(dir_path.glob("*.pdf")), desc=f"{kind} PDFs"):
        full_text = read_pdf(pdf)
        chs = chunk(full_text, max_chars=800)
        title = pdf.stem
        for i, ch in enumerate(chs):
            rows.append({
                "id": f"{kind}-{title}-{i+1}",
                "kind": kind,               # "case" or "hb"
                "title": title,
                "text": ch
            })

def main():
    rows = []
    CASES_DIR.mkdir(parents=True, exist_ok=True)
    HB_DIR.mkdir(parents=True, exist_ok=True)

    ingest_dir(CASES_DIR, "case", rows)
    ingest_dir(HB_DIR,   "hb",   rows)

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} chunks → {OUT_JSONL}")

if __name__ == "__main__":
    main()
