import json
from pathlib import Path
from functools import lru_cache

BASE_DIR = Path(__file__).resolve().parents[1]

API571_PATH = BASE_DIR / "data" / "api571_mechanisms_clean.json"


@lru_cache(maxsize=1)
def _load_index():
    """
    Load API571 mechanisms as a dict indexed by 'id'.
    The raw JSON is a list of entries.
    """
    with open(API571_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # If it's already a dict, just return it
    if isinstance(raw, dict):
        return raw

    # Otherwise assume it's a list[dict] with an 'id' field
    index = {}
    if isinstance(raw, list):
        for entry in raw:
            mech_id = entry.get("id") or entry.get("section")
            if mech_id:
                index[str(mech_id)] = entry
    return index


def get_mechanism_entry(mech_id: str):
    """
    Return the API571 entry for a mechanism id like '3.2'.
    If not found, returns None.
    """
    data = _load_index()
    return data.get(str(mech_id))


def get_mechanism_name(mech_id: str) -> str:
    entry = get_mechanism_entry(mech_id)
    if not entry:
        return str(mech_id)
    return entry.get("name", str(mech_id))