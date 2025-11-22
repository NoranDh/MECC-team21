# app.py
from flask import Flask, request, jsonify
from pathlib import Path 
import sys 
from typing import List, Dict
from flask_cors import CORS

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"

if str(SCRIPTS) not in sys.path:
    sys.path.append(str(SCRIPTS))

from rag_faiss_client import get_rag_evidence
from api571_loader import get_mechanism_entry, get_mechanism_name

from agents import Incident, SimilarCase
from agents.reasoner import reasoner
from agents.recommender import recommender

app = Flask(__name__)
CORS(app)

def docs_to_handbook_snips(hb_docs) -> List[Dict]:
    snips = []
    for i, d in enumerate(hb_docs):
        md = d.metadata or {}
        snips.append({
            "id": f"hb-{i}",
            "source": md.get("file_name", "handbook"),
            "score": md.get("score"),
            "text": d.page_content,
        })
    return snips 

def docs_to_similar_cases(case_docs) -> List[SimilarCase]:
    sims: List[SimilarCase] = []
    for i, d in enumerate(case_docs):
        md = d.metadata or {}
        snippet = d.page_content[:800]
        sim = SimilarCase(
            id=md.get("case_id", f"case-{i}"),
            title=md.get("file_name", "unknown_case"),
            snippet=snippet,
            mechanism=md.get("section", ""),
            similarity=1.0 / (1.0 + md.get("score", 0.0)),
        )
        sims.append(sim)
    return sims

def add_api571_snip(handbook_snips: List[Dict], mech_id: str) -> List[Dict]:
    entry = get_mechanism_entry(mech_id)
    if not entry:
        return handbook_snips

    lines = []
    name = entry.get("name", mech_id)
    lines.append(f"API 571 mechanism {mech_id}: {name}")

    def add_block(label, key):
        val = entry.get(key)
        if not val:
            return
        if isinstance(val, list):
            val = "\n".join(val)
        lines.append(f"{label}: {val}")

    add_block("Description of damage", "description_of_damage")
    add_block("Affected materials", "affected_materials")
    add_block("Critical factors", "critical_factors")
    add_block("Affected units / equipment", "affected_units_equipment")
    add_block("Appearance / morphology", "appearance")
    add_block("Prevention / mitigation", "prevention_mitigation")
    add_block("Inspection / monitoring", "inspection_monitoring")

    text = "\n".join(lines)

    api_snip = {
        "id": f"api571-{mech_id}",
        "source": "api571",
        "score": 0.0,
        "text": text,
    }

    return [api_snip] + handbook_snips



# API endpoints 

@app.post("/api/analyze")
def analyze():
    """
    JSON in:
    {
      "description": "...",          # required
      "mechanism_id": "3.2",         # required (from dropdown in UI)
      "material": "Carbon steel",    # optional
      "environment": "Wet CO2 ...",  # optional
      "time_in_service": "5 years"   # optional
    }
    """
    data = request.get_json(force=True) or {}

    description = (data.get("description") or "").strip()
    if not description:
        return jsonify({"error": "description is required"}), 400

    mech_id = str(data.get("mechanism_id") or "3.2")
    mech_name = get_mechanism_name(mech_id)

    # Build Incident – we keep extra fields optional
    incident = Incident(
        material=data.get("material") or "",
        environment=data.get("environment") or "",
        observed_damage=description,
        time_in_service=data.get("time_in_service") or "",
        description=description,
    )

    # Build RAG query
    query = (
        f"Failure mechanism: {mech_name} (API571 {mech_id}). "
        f"Observed damage: {incident.observed_damage}. "
        f"Environment: {incident.environment}."
    )

    hb_docs, case_docs = get_rag_evidence(query, k=8)
    handbook_snips = docs_to_handbook_snips(hb_docs)
    similar_cases = docs_to_similar_cases(case_docs)
    handbook_snips = add_api571_snip(handbook_snips, mech_id)

    mechs_out = reasoner(
        case=incident,
        similar_cases=similar_cases,
        handbook_snips=handbook_snips,
    )

    recs_out = recommender(
        case=incident,
        mechanisms=mechs_out,
        handbook_snips=handbook_snips,
    )

    # Serialize to plain JSON
    mechanisms_json = [m.model_dump() for m in mechs_out.mechanisms]
    recs_json = recs_out.model_dump()

    return jsonify({
        "mechanisms": mechanisms_json,
        "recommendations": recs_json,
        "mechanism_label": mech_name,
        "mechanism_id": mech_id,
    })


if __name__ == "__main__":
    # Run dev server
    app.run(host="127.0.0.1", port=5000, debug=True)
