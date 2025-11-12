from flask import Flask, request, jsonify

# --- dummy imports so this file runs even if agents aren't ready ---
try:
    from agents import Incident, SimilarCase
    from agents.reasoner import reasoner
    from agents.recommender import recommender
except Exception:
    # minimal fallbacks so the route exists
    from pydantic import BaseModel
    from typing import Optional, List

    class Incident(BaseModel):
        asset_type: Optional[str] = None
        component: Optional[str] = None
        material: Optional[str] = None
        service: Optional[str] = None
        environment: Optional[str] = None
        temperature: Optional[str] = None
        pressure: Optional[str] = None
        observed_damage: Optional[str] = None
        location: Optional[str] = None
        time_in_service: Optional[str] = None
        notes: Optional[str] = None
        lab_summary: Optional[str] = None

    class SimilarCase(BaseModel):
        id: str
        title: str
        snippet: str
        mechanism: Optional[str] = None
        similarity: float
        ref: Optional[str] = None

    def reasoner(case, similar, hb):
        return {"mechanisms": [{"name":"CO2 corrosion","confidence":0.8,"reasoning":"demo","evidence":["HB11-2.3"]}]}

    def recommender(case, mechs, hb):
        return {"immediate":["Isolate section"],"medium_term":["Verify inhibitor residuals"],
                "long_term":["Upgrade elbows"],"monitoring":["Coupons @ 90d"],"gaps":[]}

app = Flask(__name__)

@app.route("/api/analyze_failure", methods=["POST","GET"])
def analyze_failure():
    if request.method == "GET":
        return "analyze_failure endpoint is alive (send POST JSON)", 200

    payload = request.get_json(silent=True) or {}
    case = Incident(**(payload.get("incident") or {}))
    similar = [SimilarCase(**c) for c in (payload.get("similar_cases") or [])]
    handbook = payload.get("handbook_snippets") or []

    mechs = reasoner(case, similar, handbook)
    recs  = recommender(case, mechs, handbook)

    return jsonify({"mechanisms": mechs.model_dump(), "recommendations": recs.model_dump()})


@app.route("/", methods=["GET"])
def health():
    return "OK", 200


@app.route("/api/_envcheck", methods=["GET"])
def envcheck():
    import os
    from utils.llm import API_KEY, MODEL
    return {
        "cwd": os.getcwd(),
        "has_api_key": bool(API_KEY),
        "model": MODEL,
    }, 200

if __name__ == "__main__":
    print("Routes:\n", app.url_map)
    app.run(port=8000, debug=True)
