import pdfplumber
from pathlib import Path

CASES_DIR = Path(r"C:\Users\Noran\Desktop\MECC_21\data\cases")
OUT_DIR = Path(r"C:\Users\Noran\Desktop\MECC_21\data\extracted_cases")

OUT_DIR.mkdir(exist_ok=True)

for pdf_path in CASES_DIR.glob("*.pdf"):
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text() or ""
            text_parts.append(txt)
    full_text = "\n\n".join(text_parts)

    out_txt = OUT_DIR / (pdf_path.stem + "_raw.txt")
    out_txt.write_text(full_text, encoding="utf-8")
    print("Saved", out_txt)