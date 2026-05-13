from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict
import numpy as np

app = FastAPI(title="Embedding Space 3D")

MODELS: Dict[str, Dict] = {
    "multilingual": {
        "id":    "paraphrase-multilingual-MiniLM-L12-v2",
        "label": "Multilingual MiniLM",
        "hint":  "Multilingual · paragraphs & long texts",
        "dims":  384,
    },
    "minilm": {
        "id":    "all-MiniLM-L6-v2",
        "label": "MiniLM L6",
        "hint":  "English · terms & short phrases",
        "dims":  384,
    },
    "mpnet": {
        "id":    "all-mpnet-base-v2",
        "label": "MPNet",
        "hint":  "English · highest quality",
        "dims":  768,
    },
}

_model_instances: Dict[str, object] = {}
_states: Dict[str, List[dict]] = {k: [] for k in MODELS}


class Term(BaseModel):
    word: str

class AddRequest(BaseModel):
    terms: List[Term]
    model: str = "multilingual"


def get_model(key: str):
    if key not in MODELS:
        key = "multilingual"
    if key not in _model_instances:
        from sentence_transformers import SentenceTransformer
        _model_instances[key] = SentenceTransformer(MODELS[key]["id"])
    return _model_instances[key]


def _reduce_3d(vecs: np.ndarray) -> np.ndarray:
    n = len(vecs)
    if n == 1:
        return np.zeros((1, 3))
    if n == 2:
        sim = float(np.dot(vecs[0], vecs[1]))  # vecs are already normalized
        dist = (1.0 - sim) / 2.0
        return np.array([[-dist, 0.0, 0.0], [dist, 0.0, 0.0]])
    if n < 6:
        from sklearn.manifold import MDS
        from sklearn.metrics.pairwise import cosine_distances
        dist_matrix = cosine_distances(vecs)
        coords = MDS(n_components=3, dissimilarity='precomputed', random_state=42).fit_transform(dist_matrix)
    else:
        import umap
        coords = umap.UMAP(
            n_components=3,
            n_neighbors=min(15, n - 1),
            min_dist=0.15,
            metric="cosine",
            random_state=42,
        ).fit_transform(vecs)

    # Uniform scaling across all axes — preserves relative distances between points
    scale = max(coords[:, i].max() - coords[:, i].min() for i in range(3))
    if scale > 0:
        for i in range(3):
            coords[:, i] = (coords[:, i] - coords[:, i].mean()) / (scale * 0.4)
    return coords


def _build_response(model_key: str) -> dict:
    state = _states[model_key]
    if not state:
        return {"points": [], "model": model_key}
    vecs = np.array([s["vec"] for s in state])
    coords = _reduce_3d(vecs)
    return {
        "model": model_key,
        "points": [
            {
                "word":  s["word"],
                "color": s["color"],
                "x": float(coords[i, 0]),
                "y": float(coords[i, 1]),
                "z": float(coords[i, 2]),
                "vec":   s["vec"],
            }
            for i, s in enumerate(state)
        ]
    }


@app.get("/api/models")
def list_models():
    return {
        "models": [
            {"key": k, "label": v["label"], "hint": v["hint"], "dims": v["dims"]}
            for k, v in MODELS.items()
        ]
    }


@app.get("/api/state")
def get_state(model: str = Query("multilingual")):
    if model not in _states:
        model = "multilingual"
    return _build_response(model)


@app.post("/api/embed")
def add_terms(req: AddRequest):
    key = req.model if req.model in _states else "multilingual"
    state = _states[key]
    existing = {s["word"] for s in state}
    new_terms = [t for t in req.terms if t.word not in existing]
    if new_terms:
        model = get_model(key)
        vecs = model.encode([t.word for t in new_terms], normalize_embeddings=True)
        for t, vec in zip(new_terms, vecs):
            state.append({
                "word":  t.word,
                "color": "#c8f064",
                "vec":   vec.tolist(),
            })
    return _build_response(key)


@app.delete("/api/term/{word}")
def remove_term(word: str, model: str = Query("multilingual")):
    if model not in _states:
        model = "multilingual"
    _states[model][:] = [s for s in _states[model] if s["word"] != word]
    return _build_response(model)


@app.delete("/api/state")
def clear_state(model: str = Query("multilingual")):
    if model not in _states:
        model = "multilingual"
    _states[model].clear()
    return {"points": [], "model": model}


app.mount("/", StaticFiles(directory="static", html=True), name="static")
