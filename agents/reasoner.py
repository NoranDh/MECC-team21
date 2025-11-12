from typing import List
from agents import Incident, SimilarCase, MechanismsOut, Mechanism
from utils.llm import call_llm
from utils.prompts import REASONER_SYS
import json

MECH_KEYWORDS = [
    "co2 corrosion","mic","scc","hic","ssc","erosion-corrosion",
    "pitting","galvanic","crevice","chloride scc","caustic scc",
    "abrasive wear","adhesive wear","fatigue","embrittlement"
]

def _candidate_list(handbook_snips: List[dict],
                    similar_cases: List[SimilarCase],
                    max_k: int = 6) -> List[str]:
    """Build a short, deduped mechanism shortlist from handbook + similar cases."""
    text = " ".join(s.get("text", "") for s in handbook_snips).lower()
    found = set()

    # 1) from handbook text
    for kw in MECH_KEYWORDS:
        if kw in text:
            found.add(kw.upper() if kw.endswith(" scc") else kw.title() if "scc" not in kw else kw.upper())

    # 2) from similar cases (if they carry a mechanism tag)
    for c in similar_cases or []:
        if getattr(c, "mechanism", None):
            found.add(c.mechanism)

    # 3) fallback if empty
    if not found:
        found = {"CO2 corrosion", "MIC", "erosion-corrosion"}

    return list(found)[:max_k]

def reasoner(case: Incident,
             similar_cases: List[SimilarCase],
             handbook_snips: List[dict]) -> MechanismsOut:

    candidates = _candidate_list(handbook_snips, similar_cases)

    prompt = f"""{REASONER_SYS}

New case JSON:
{case.model_dump()}

Candidate mechanisms: {candidates}

Similar cases (id/title/snippet/mechanism/similarity):
{[c.model_dump() for c in similar_cases]}

Handbook excerpts (ids/sources only):
{[{"id": s.get("id"), "source": s.get("source") or s.get("metadata", {}).get("source")} for s in handbook_snips]}

Choose 1–3 mechanisms with confidence and reasoning. Cite evidence by case id or handbook id.
Return JSON only."""
    raw = call_llm(prompt, json_expected=True)

    # Robust JSON handling
    try:
        data = json.loads(raw)
        items = data.get("mechanisms", [])
    except Exception:
        # Safe fallback so the API never crashes
        items = [{"name": "CO2 corrosion", "confidence": 0.5,
                  "reasoning": "Fallback response; model returned invalid JSON.",
                  "evidence": []}]

    mechs = [Mechanism(**m) for m in items]
    return MechanismsOut(mechanisms=mechs)