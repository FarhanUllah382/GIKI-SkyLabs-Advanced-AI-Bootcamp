"""
embeddings.py

Loads and manages the embedding model used for semantic memory.

Default Model:
    sentence-transformers/all-MiniLM-L6-v2

Output Dimension:
    384
"""

from sentence_transformers import SentenceTransformer


class EmbeddingManager:

    def __init__(
        self,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    ):

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    # ----------------------------------------------------
    # Embed Single Text
    # ----------------------------------------------------

    def embed(self, text: str):

        embedding = self.model.encode(
            text,
            normalize_embeddings=True
        )

        return embedding.tolist()

    # ----------------------------------------------------
    # Embed Multiple Texts
    # ----------------------------------------------------

    def embed_batch(self, texts):

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True
        )

        return embeddings.tolist()

    # ----------------------------------------------------
    # Model Information
    # ----------------------------------------------------

    @property
    def dimension(self):

        return self.model.get_sentence_embedding_dimension()

    @property
    def name(self):

        return self.model_name