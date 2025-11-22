from pathlib import Path
from typing import List, Dict

import sys

# Add project root to Python path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


from rag_faiss_client import get_rag_evidence
from api571_loader import get_mechanism_entry, get_mechanism_name

from agents.reasoner import reasoner
from agents.recommender import recommender
from agents import Incident, SimilarCase

BASE_DIR = Path(__file__).resolve().parents[1]


# convert FAISS indec to agents expected input 

def docs_to_handbook_snips(hb_docs) -> List[Dict]:
    """
    Convert handbook Documents from FAISS into simple dicts.
    Reasoner/Recommender already expect handbook_snips as List[dict].
   
    """

    snips = []
    for i,d in enumerate(hb_docs):
        md = d.metadata or {}
        snips.append({
             "id": f"hb-{i}",
            "source": md.get("file_name", "handbook"),
            "score": md.get("score"),
            "text": d.page_content,
        })
        return snips 
    
def docs_to_similar_cases(case_docs) -> List[SimilarCase]:
     
     """
    Convert case Documents into SimilarCase objects.
    Adjust field names if your SimilarCase model is different.
    """
     sims: List[SimilarCase] = []
     for i, d in enumerate(case_docs):
        md = d.metadata or {}
        snippet = d.page_content[:800]

        sim = SimilarCase(
            id=md.get("case_id", f"case-{i}"),
            title=md.get("file_name", "unknown_case"),
            snippet=snippet,
            mechanism=md.get("section", ""),      # if you have section per chunk
            similarity=1.0 / (1.0 + md.get("score", 0.0)),  # simple transform
        )
        sims.append(sim)
     return sims

def add_api571_snip(handbook_snips: List[Dict], mech_id: str) -> List[Dict]:
    """
    Wrap API571 mechanism entry as an extra 'handbook snippet',
    so we don't need to change the agents' signatures.
    """
    entry = get_mechanism_entry(mech_id)
    if not entry:
        return handbook_snips

    # Build a compact text block from API571 fields
    lines = []
    name = entry.get("name", mech_id)
    lines.append(f"API 571 mechanism {mech_id}: {name}")
    desc = entry.get("description_of_damage") or entry.get("description") or ""
    if desc:
        lines.append(f"Description of damage: {desc}")
    mat = entry.get("affected_materials") or ""
    if mat:
        lines.append(f"Affected materials: {mat}")
    crit = entry.get("critical_factors") or ""
    if crit:
        lines.append(f"Critical factors: {crit}")
    morph = entry.get("appearance_or_morphology_of_damage") or entry.get("appearance") or ""
    if morph:
        lines.append(f"Appearance / morphology: {morph}")
    prev = entry.get("prevention_mitigation") or ""
    if prev:
        lines.append(f"Prevention / mitigation: {prev}")
    insp = entry.get("inspection_monitoring") or ""
    if insp:
        lines.append(f"Inspection / monitoring: {insp}")

    text = "\n".join(lines)

    api_snip = {
        "id": f"api571-{mech_id}",
        "source": "api571",
        "score": 0.0,
        "text": text,
    }

    # Put API571 at the front so LLM clearly sees it
    return [api_snip] + handbook_snips


# Build a dummy Incident for testing 

def build_test_incident() -> Incident:
    """
    Build a test incident. Make sure these fields match your Incident model.
    If your Incident class has different field names, adjust here.
    """
    inc = Incident(
        material="Carbon steel",
        environment="Wet CO2 amine service in carbon steel piping",
        observed_damage="Internal pitting and general wall loss at the bottom of a horizontal carbon steel line.",
        time_in_service="5 years",
        description="Suspected CO2 corrosion in an amine unit transfer line.",
    )
    return inc


def main():
    # 1) Build test incident (this will later come from API payload)
    incident = build_test_incident()

    # 2) Assume the mechanism id from classifier or user selection
    mech_id = "3.2"  # example
    mech_name = get_mechanism_name(mech_id)

    # 3) Build RAG query from mechanism + incident description
    query = (
        f"Failure mechanism: {mech_name} (API571 {mech_id}). "
        f"Observed damage: {incident.observed_damage}. "
        f"Environment: {incident.environment}."
    )

    print(">>> RAG query:")
    print(query, "\n")

    # 4) Retrieve handbook + case chunks from FAISS
    hb_docs, case_docs = get_rag_evidence(query, k=8)
    print(f"Retrieved {len(hb_docs)} handbook chunks and {len(case_docs)} case chunks.\n")

    handbook_snips = docs_to_handbook_snips(hb_docs)
    similar_cases = docs_to_similar_cases(case_docs)

    # 5) Inject API571 as one extra handbook snippet
    handbook_snips_with_api = add_api571_snip(handbook_snips, mech_id)

    # 6) Call reasoner -> MechanismsOut
    mechs_out = reasoner(
        case=incident,
        similar_cases=similar_cases,
        handbook_snips=handbook_snips_with_api,
    )

    print("=== MechanismsOut ===")
    print(mechs_out, "\n")

    # 7) Call recommender -> RecsOut
    recs_out = recommender(
        case=incident,
        mechanisms=mechs_out,
        handbook_snips=handbook_snips_with_api,
    )

    print("=== RecsOut ===")
    print(recs_out)


if __name__ == "__main__":
    main()
