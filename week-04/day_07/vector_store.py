"""
vector_store.py

Persistent ChromaDB wrapper.

Responsibilities
----------------
✓ Store memories
✓ Retrieve relevant memories
✓ Delete memories
✓ Update memories
✓ Count memories
✓ Filter by metadata
"""

from datetime import datetime
import uuid

import chromadb
from chromadb.config import Settings

from embeddings import EmbeddingManager


class VectorStore:

    def __init__(

        self,

        persist_directory="vector_db",

        collection_name="user_memory",

    ):

        self.embedding = EmbeddingManager()

        self.client = chromadb.PersistentClient(

            path=persist_directory,

            settings=Settings(

                anonymized_telemetry=False

            )

        )

        self.collection = self.client.get_or_create_collection(

            name=collection_name,

            metadata={

                "description": "Semantic memory"

            }

        )

    # -------------------------------------------------------
    # Add Memory
    # -------------------------------------------------------

    def add_memory(

        self,

        text,

        thread_id,

        importance=1.0,

        category="general",

    ):

        embedding = self.embedding.embed(text)

        memory_id = str(uuid.uuid4())

        metadata = {

            "thread_id": thread_id,

            "importance": float(importance),

            "category": category,

            "timestamp": datetime.now().isoformat(),

        }

        self.collection.add(

            ids=[memory_id],

            embeddings=[embedding],

            documents=[text],

            metadatas=[metadata],

        )

        return memory_id

    # -------------------------------------------------------
    # Batch Insert
    # -------------------------------------------------------

    def add_memories(

        self,

        texts,

        thread_id,

        category="general",

    ):

        embeddings = self.embedding.embed_batch(texts)

        ids = [

            str(uuid.uuid4())

            for _ in texts

        ]

        metadata = []

        for _ in texts:

            metadata.append(

                {

                    "thread_id": thread_id,

                    "importance": 1.0,

                    "category": category,

                    "timestamp": datetime.now().isoformat(),

                }

            )

        self.collection.add(

            ids=ids,

            embeddings=embeddings,

            documents=texts,

            metadatas=metadata,

        )

    # -------------------------------------------------------
    # Semantic Search
    # -------------------------------------------------------

    def search(

        self,

        query,

        top_k=5,

    ):

        embedding = self.embedding.embed(query)

        results = self.collection.query(

            query_embeddings=[embedding],

            n_results=top_k,

        )

        memories = []

        docs = results.get(

            "documents",

            [[]]

        )[0]

        metadata = results.get(

            "metadatas",

            [[]]

        )[0]

        distance = results.get(

            "distances",

            [[]]

        )[0]

        ids = results.get(

            "ids",

            [[]]

        )[0]

        for doc, meta, dist, memory_id in zip(

            docs,

            metadata,

            distance,

            ids,

        ):

            memories.append(

                {

                    "id": memory_id,

                    "text": doc,

                    "metadata": meta,

                    "distance": dist,

                }

            )

        return memories

    # -------------------------------------------------------
    # Search By Category
    # -------------------------------------------------------

    def search_category(

        self,

        query,

        category,

        top_k=5,

    ):

        embedding = self.embedding.embed(query)

        results = self.collection.query(

            query_embeddings=[embedding],

            n_results=top_k,

            where={

                "category": category

            }

        )

        return results

    # -------------------------------------------------------
    # Delete
    # -------------------------------------------------------

    def delete(

        self,

        memory_id,

    ):

        self.collection.delete(

            ids=[memory_id]

        )

    # -------------------------------------------------------
    # Delete Thread
    # -------------------------------------------------------

    def delete_thread(

        self,

        thread_id,

    ):

        self.collection.delete(

            where={

                "thread_id": thread_id

            }

        )

    # -------------------------------------------------------
    # Count
    # -------------------------------------------------------

    def count(self):

        return self.collection.count()

    # -------------------------------------------------------
    # Get All Memories
    # -------------------------------------------------------

    def all_memories(self):

        return self.collection.get()

    # -------------------------------------------------------
    # Reset
    # -------------------------------------------------------

    def reset(self):

        self.client.delete_collection(

            "user_memory"

        )

        self.collection = self.client.get_or_create_collection(

            "user_memory"

        )