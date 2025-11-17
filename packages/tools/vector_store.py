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
        self._index = {}  # id -> {"text":..., "vec": list} - always keep as list for JSON serialization
        if os.path.exists(STORE_PATH):
            try:
                with open(STORE_PATH, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                # Keep vectors as lists (not numpy arrays) to avoid JSON serialization issues
                # We'll convert to numpy arrays on-the-fly when needed for computation
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
            # Convert list to numpy array on-the-fly for computation
            vec_list = v.get("vec", [])
            if isinstance(vec_list, list):
                vec = np.array(vec_list)
            else:
                # Handle case where it might already be a numpy array (backward compatibility)
                vec = np.array(vec_list) if not isinstance(vec_list, np.ndarray) else vec_list
            # cosine similarity
            sim = float(np.dot(qvec, vec) / (np.linalg.norm(qvec) * np.linalg.norm(vec)))
            scores.append((doc_id, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def _persist(self):
        # Ensure all vectors are lists (not numpy arrays) for JSON serialization
        to_save = {}
        for k, v in self._index.items():
            vec = v.get("vec", [])
            # Convert numpy array to list if needed
            if isinstance(vec, np.ndarray):
                vec = vec.tolist()
            elif not isinstance(vec, list):
                vec = list(vec) if hasattr(vec, '__iter__') else []
            to_save[k] = {"text": v.get("text", ""), "vec": vec}
        with open(STORE_PATH, "w", encoding="utf-8") as f:
            json.dump(to_save, f, indent=2)
