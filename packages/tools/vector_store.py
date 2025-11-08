import numpy as np
import hashlib
import os
import json
from typing import List, Tuple

STORE_PATH = "packages/tools/_vector_store.json"

def _text_to_embedding(text: str, dim: int = 128) -> np.ndarray:
    """
    Deterministic pseudo-embedding: hash text -> seed -> rand vector
    Not a real semantic embedder, but stable and reproducible for tests/demos.
    """
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    seed = int(h[:8], 16)  # take first 8 hex chars
    rng = np.random.RandomState(seed)
    vec = rng.normal(size=(dim,))
    # normalize
    vec = vec / np.linalg.norm(vec)
    return vec

class MockVectorStore:
    def __init__(self, dim: int = 128):
        self.dim = dim
        self._index = {}  # id -> {"text":..., "vec": list}
        if os.path.exists(STORE_PATH):
            try:
                with open(STORE_PATH, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                # convert vec lists back to numpy arrays
                for k,v in raw.items():
                    v["vec"] = np.array(v["vec"])
                self._index = raw
            except Exception:
                self._index = {}

    def add(self, doc_id: str, text: str):
        vec = _text_to_embedding(text, dim=self.dim)
        self._index[doc_id] = {"text": text, "vec": vec.tolist()}
        self._persist()

    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        qvec = _text_to_embedding(query, dim=self.dim)
        scores = []
        for doc_id, v in self._index.items():
            vec = np.array(v["vec"])
            # cosine similarity
            sim = float(np.dot(qvec, vec) / (np.linalg.norm(qvec) * np.linalg.norm(vec)))
            scores.append((doc_id, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def _persist(self):
        # convert np arrays to lists for JSON
        to_save = {k: {"text": v["text"], "vec": v["vec"]} for k,v in self._index.items()}
        with open(STORE_PATH, "w", encoding="utf-8") as f:
            json.dump(to_save, f, indent=2)
