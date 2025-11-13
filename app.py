# app.py
from flask import Flask, request, jsonify

from agents import Incident, SimilarCase
from agents.reasoner import reasoner
from agents.recommender import recommender
from rag.store import retrieve_chunks    # <- RAG helper

app = Flask(__name__)


@app.route("/api/analyze_failure", methods=["POST"])
def analyze_failure():
    # 1) Read JSON payload
    payload = request.get_json(force=True) or {}

    # ----- INCIDENT ---------------------------------------------------------
    incident_data = payload.get("incident") or {}
    case = Incident(**incident_data)

    # 2) Build a text query from the incident fields
    query_parts = [
        case.asset_type or "",
        case.component or "",
        case.material or "",
        case.service or "",
        case.environment or "",
        case.temperature or "",
        case.pressure or "",
        case.observed_damage or "",
        case.location or "",
        case.time_in_service or "",
        case.notes or "",
        case.lab_summary or "",
    ]
    query_text = " | ".join([p for p in query_parts if p])

    # ----- RAG RETRIEVAL ----------------------------------------------------
    # 3) Get relevant chunks from corpus (case studies + handbook)
    chunks = retrieve_chunks(query_text, top_k=8)

    case_hits = [c for c in chunks if c.get("kind") == "case"]
    hb_hits   = [c for c in chunks if c.get("kind") == "hb"]

    # Convert top case hits -> SimilarCase models
    similar_cases = []
    for c in case_hits[:3]:
        similar_cases.append(
            SimilarCase(
                id=c["id"],
                title=c.get("title", ""),
                snippet=c.get("text", "")[:400],
                mechanism=None,
                similarity=0.0,     
                ref=c["id"],
            )
        )

    # Handbook snippets stay as simple dicts
    handbook_snips = []
    for h in hb_hits[:5]:
        handbook_snips.append(
            {
                "id": h["id"],
                "text": h["text"],
                "source": h.get("title"),
            }
        )

    # ----- AGENTS -----------------------------------------------------------
    # 4) Run reasoner (mechanisms)
    mechs = reasoner(case, similar_cases, handbook_snips)

    # 5) Run recommender (actions)
    recs = recommender(case, mechs, handbook_snips)

    # 6) Return JSON-friendly dicts
    return jsonify(
        {
            "incident": case.model_dump(),
            "rag_query": query_text,
            "rag_chunks": chunks,  # optional: remove if too big
            "mechanisms": mechs.model_dump(),
            "recommendations": recs.model_dump(),
        }
    )


@app.route("/", methods=["GET"])
def health():
    return "OK", 200


if __name__ == "__main__":
    print("Routes:", app.url_map)
    app.run(port=8000, debug=True)
