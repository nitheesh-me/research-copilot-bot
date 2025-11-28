try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

import numpy as np
import hashlib
from moya.tools.tool import Tool

_model = None

def init_model(name="all-MiniLM-L6-v2"):
    global _model
    if _model is None and HAS_SENTENCE_TRANSFORMERS:
        _model = SentenceTransformer(name)
    return _model

def embed_texts(texts: list) -> list:
    """
    Compute embeddings for a list of texts.

    :param texts: List of strings to embed.
    :return: List of embeddings (as lists of floats).
    """
    if HAS_SENTENCE_TRANSFORMERS:
        try:
            m = init_model()
            # batch_size is hardcoded for now, could be a param
            embs = m.encode(texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
            return embs.tolist() # Convert to list for JSON serialization if needed
        except Exception as e:
            print(f"Embedding failed with model, falling back to dummy: {e}")
            pass

    # FALLBACK: Deterministic dummy embedding for prototype stability
    # This avoids heavy dependency issues (tensorflow/torch) during demo

    dim = 384 # Standard dimension for MiniLM
    embeddings = []
    for text in texts:
        # Create a deterministic vector from the text hash
        seed = int(hashlib.sha256(text.encode('utf-8')).hexdigest(), 16) % (2**32)
        rng = np.random.RandomState(seed)
        vec = rng.randn(dim)
        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        embeddings.append(vec.tolist())

    return embeddings

embedder_tool = Tool(
    name="embed_texts",
    description="Compute embeddings for a list of texts.",
    function=embed_texts
)
