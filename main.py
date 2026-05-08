from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
import numpy as np

app = FastAPI(title="Embedding Space 3D")

# Model loaded lazily on first request
_model = None

def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return _model

# In-memory state for the session
_state: List[dict] = []

GROUP_COLORS = {
    "animais":    "#5DCAA5",
    "tecnologia": "#9d8ff5",
    "culinária":  "#EF9F27",
    "emoções":    "#e06fa0",
    "custom":     "#c8f064",
}


class Term(BaseModel):
    word: str
    group: str = "custom"

class AddRequest(BaseModel):
    terms: List[Term]


def _reduce_3d(vecs: np.ndarray) -> np.ndarray:
    n = len(vecs)
    if n == 1:
        return np.zeros((1, 3))
    if n == 2:
        return np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
    if n < 6:
        from sklearn.decomposition import PCA
        coords = PCA(n_components=3).fit_transform(vecs)
    else:
        import umap
        coords = umap.UMAP(
            n_components=3,
            n_neighbors=min(15, n - 1),
            min_dist=0.15,
            metric="cosine",
            random_state=42,
        ).fit_transform(vecs)

    # Normalise to roughly [-1.5, 1.5] so it fits the canvas projection
    for i in range(3):
        col = coords[:, i]
        span = col.max() - col.min()
        if span > 0:
            coords[:, i] = (col - col.mean()) / (span * 0.4)
    return coords


def _build_response() -> dict:
    if not _state:
        return {"points": []}
    vecs = np.array([s["vec"] for s in _state])
    coords = _reduce_3d(vecs)
    return {
        "points": [
            {
                "word":  s["word"],
                "group": s["group"],
                "color": s["color"],
                "x": float(coords[i, 0]),
                "y": float(coords[i, 1]),
                "z": float(coords[i, 2]),
                "vec":   s["vec"],
            }
            for i, s in enumerate(_state)
        ]
    }


@app.post("/api/embed")
def add_terms(req: AddRequest):
    existing = {s["word"] for s in _state}
    new_terms = [t for t in req.terms if t.word not in existing]
    if new_terms:
        model = get_model()
        vecs = model.encode([t.word for t in new_terms], normalize_embeddings=True)
        for t, vec in zip(new_terms, vecs):
            _state.append({
                "word":  t.word,
                "group": t.group,
                "color": GROUP_COLORS.get(t.group, "#c8f064"),
                "vec":   vec.tolist(),
            })
    return _build_response()


@app.delete("/api/term/{word}")
def remove_term(word: str):
    _state[:] = [s for s in _state if s["word"] != word]
    return _build_response()


@app.delete("/api/state")
def clear_state():
    _state.clear()
    return {"points": []}


app.mount("/", StaticFiles(directory="static", html=True), name="static")
