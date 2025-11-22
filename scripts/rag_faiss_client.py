from pathlib import Path
from typing import List, Tuple

from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document



BASE_DIR = Path(__file__).resolve().parents[1]
INDEX_DIR = BASE_DIR / "data" / "rag_faiss_index"

EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_embedding = HuggingFaceEmbeddings(model_name = EMB_MODEL_NAME)
_vectorstore = FAISS.load_local(
    str(INDEX_DIR),
    embeddings=_embedding,
    allow_dangerous_deserialization=True,

)

def get_rag_evidence(query: str, k: int = 8) -> Tuple[List[Document], List[Document]]:
    """
    Run semantic search over the unified FAISS index.
    Returns (hb_docs, case_docs).
    """

    docs_scores = _vectorstore.similarity_search_with_score(query, k=k)

    hb_docs: List[Document] = []
    case_docs: List[Document] = []

    for doc, score in docs_scores:
        md = doc.metadata or {}

        md["score"] = float(score)
        doc.metadata = md

        src = md.get("source")
        if src == "hb":
            hb_docs.append(doc)
        elif src == "case":
            case_docs.append(doc)

    return hb_docs, case_docs

