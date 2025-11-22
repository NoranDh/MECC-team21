from typing import List
from agents import Incident, MechanismsOut, RecsOut
from utils.llm import call_llm
from utils.prompts import RECS_SYS
import json
from dotenv import load_dotenv
load_dotenv()

def _gaps(case: Incident) -> List[str]:
    must = ["material", "environment", "observed_damage", "time_in_service"]
    return [f"Missing {k.replace('_', ' ')}" for k in must if not getattr(case, k)]

def recommender(case: Incident,
                mechanisms: MechanismsOut,
                handbook_snips: List[dict]) -> RecsOut:
    
    # safely build handbook refs list
    hb_refs = [
        {
            "id": s.get("id"),
            "source": s.get("source") or (s.get("metadata") or {}).get("source")
        }
        for s in handbook_snips
    ]

    prompt = f"""{RECS_SYS}

Incident:
{case.model_dump()}

Mechanisms (selected):
{[m.model_dump() for m in mechanisms.mechanisms]}

Handbook snippets (ids/sources only):
{hb_refs}

Return valid JSON with keys: immediate, medium_term, long_term, monitoring, gaps.
No prose, no markdown — only valid JSON.
"""

    raw = call_llm(prompt, json_expected=True)
    data = json.loads(raw)
    data["gaps"] = sorted(set((data.get("gaps") or []) + _gaps(case)))
    return RecsOut(**data)