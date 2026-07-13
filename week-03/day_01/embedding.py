from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-large-en-v1.5"


def load_embedding_model():
    return SentenceTransformer(MODEL_NAME)

def embed_chunks(model, chunks):
    """
    chunks: list of strings (or Document objects with .page_content)
    Returns: list of vectors (numpy arrays)
    """
    texts = [c.page_content if hasattr(c, "page_content") else c for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings