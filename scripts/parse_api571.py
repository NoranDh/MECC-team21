from pathlib import Path
import pdfplumber
import re, json


# 1-based page number where Section 3 starts in your PDF
START_PAGE = 16


def read_api_text(pdf_path: Path) -> str:
    """Read API-571 text starting from page 16 (skip TOC, preface, etc.)."""
    pages = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        # pdfplumber is 0-based, so START_PAGE-1
        for p in pdf.pages[START_PAGE - 1 :]:
            pages.append(p.extract_text() or "")
    return "\n".join(pages)


# Match any heading 3.x or 3.x.y at the start of a line
_SECTION_RE = re.compile(r"(?m)^(3\.\d+(?:\.\d+)?)\s+(.+)$")


def find_sections(text: str):
    """
    Return list of sections: each is {id, title, body}.

    id    -> '3.1' or '3.1.4'
    title -> heading text
    body  -> text until the next heading.
    """
    matches = list(_SECTION_RE.finditer(text))
    sections = []
    for i, m in enumerate(matches):
        sec_id = m.group(1)
        title = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        sections.append({"id": sec_id, "title": title, "body": body})
    return sections


def _field_for_sub(subnum: int):
    """
    Map 3.x.N -> field name on the mechanism.
    True  = treat body as list of bullets.
    False = treat body as plain text.
    """
    mapping = {
        1: ("description", False),
        2: ("affected_materials", True),
        3: ("critical_factors", True),
        4: ("affected_units_equipment", True),
        5: ("appearance", True),
        6: ("prevention_mitigation", True),
        7: ("inspection_monitoring", True),
        8: ("related_mechanisms", True),
        9: ("references", True),
    }
    return mapping.get(subnum, (None, False))


def _body_to_list(body: str):
    # turn multi-line text into clean bullet list
    lines = [ln.strip(" •-\t") for ln in body.splitlines()]
    return [ln for ln in lines if ln]


def build_mechanisms(sections):
    """
    Group 3.1, 3.1.1, 3.1.2… under a single mechanism dict.
    Returns a list[dict].
    """
    mechanisms = {}
    order = []  # keep order of mechanisms as in the document

    for sec in sections:
        sid = sec["id"]
        parts = sid.split(".")
        body = sec["body"]

        # Top-level damage mechanism: 3.1, 3.2, 3.3...
        if len(parts) == 2:
            top_id = sid
            if top_id not in mechanisms:
                mechanisms[top_id] = {
                    "id": top_id,
                    "name": sec["title"],
                    "section": top_id,
                    "summary": "",
                    "description": "",
                    "aliases": [],
                    "affected_materials": [],
                    "critical_factors": [],
                    "affected_units_equipment": [],
                    "appearance": [],
                    "prevention_mitigation": [],
                    "inspection_monitoring": [],
                    "related_mechanisms": [],
                    "references": [],
                }
                order.append(top_id)
            else:
                mechanisms[top_id]["name"] = sec["title"]

        # Sub-sections: 3.1.1, 3.1.2, ...
        elif len(parts) == 3:
            top_id = ".".join(parts[:2])
            subnum = int(parts[2])
            field, as_list = _field_for_sub(subnum)
            if not field:
                continue  # ignore anything else

            mech = mechanisms.setdefault(
                top_id,
                {
                    "id": top_id,
                    "name": "",
                    "section": top_id,
                    "summary": "",
                    "description": "",
                    "aliases": [],
                    "affected_materials": [],
                    "critical_factors": [],
                    "affected_units_equipment": [],
                    "appearance": [],
                    "prevention_mitigation": [],
                    "inspection_monitoring": [],
                    "related_mechanisms": [],
                    "references": [],
                },
            )

            if as_list:
                mech[field] = _body_to_list(body)
            else:
                txt = " ".join(body.split())
                mech[field] = txt
                if field == "description":
                    # short summary for the UI / ranking
                    mech["summary"] = txt[:300]

    # keep original order
    return [mechanisms[k] for k in order]


def parse_api571(pdf_path: str | Path, out_path: str | Path):
    pdf_path = Path(pdf_path)
    out_path = Path(out_path)

    text = read_api_text(pdf_path)
    sections = find_sections(text)
    mechs = build_mechanisms(sections)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(mechs, f, indent=2, ensure_ascii=False)

    print(f"Parsed {len(mechs)} mechanisms from {pdf_path.name}")
    print(f"Wrote JSON → {out_path}")


if __name__ == "__main__":
    # adjust these paths to match your project
    parse_api571(
        pdf_path=r"C:\Users\Noran\Desktop\MECC\data\api571\API_571_2020.pdf",
        out_path=r"C:\Users\Noran\Desktop\MECC\data\api571_mechanisms.json"
    )
