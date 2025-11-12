from pydantic import BaseModel, Field 
from typing import List, Optional

class Incident(BaseModel):
    asset_type: Optional[str] = None
    component: Optional[str] = None
    material: Optional[str] = None
    service : Optional[str] = None
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

class Mechanism(BaseModel):
    name: str
    confidence: float = Field(ge=0, le=1)
    reasoning: str
    evidence: List[str] = []

class MechanismsOut(BaseModel):
    mechanisms: List[Mechanism]

class RecsOut(BaseModel):
    immediate: List[str]
    medium_term: List[str]
    long_term: List[str]
    monitoring: List[str]
    gaps: List[str] = []
