"""
Embedding and Vector Store Client.

Provides a unified interface for generating embeddings and storing/retrieving
them using various backends (Memory, Milvus).
"""

import json
import logging
import os
import random
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)

# --- Vector Store Abstractions ---


class VectorStore(ABC):
    @abstractmethod
    def add(self, id: str, vector: list[float], metadata: dict[str, Any]):
        pass

    @abstractmethod
    def search(
        self, vector: list[float], limit: int = 10, filters: str = None
    ) -> list[dict[str, Any]]:
        pass


class InMemoryVectorStore(VectorStore):
    def __init__(self):
        self.data = {}  # id -> {vector, metadata}

    def add(self, id: str, vector: list[float], metadata: dict[str, Any]):
        self.data[id] = {"vector": vector, "metadata": metadata}

    def search(
        self, vector: list[float], limit: int = 10, filters: str = None
    ) -> list[dict[str, Any]]:
        import numpy as np

        results = []
        q_vec = np.array(vector)
        q_norm = np.linalg.norm(q_vec)

        for id, item in self.data.items():
            # Simple filter (exact match on 'island')
            if filters:
                try:
                    # Very basic filter parsing "island == 1"
                    # In a real impl, use a proper parser
                    key, val = filters.replace(" ", "").split("==")
                    meta_val = item["metadata"].get(key)
                    if str(meta_val) != str(val):
                        continue
                except Exception:
                    pass  # Ignore complex filters in memory for now

            d_vec = np.array(item["vector"])
            d_norm = np.linalg.norm(d_vec)

            if q_norm > 0 and d_norm > 0:
                sim = np.dot(q_vec, d_vec) / (q_norm * d_norm)
                results.append({"id": id, "score": float(sim), "metadata": item["metadata"]})

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]


class MilvusVectorStore(VectorStore):
    """Wrapper for Milvus."""

    def __init__(self, host="localhost", port="19530", collection_name="openevolve_programs"):
        try:
            import importlib.util

            if importlib.util.find_spec("pymilvus") is None:
                raise ImportError
            self.pymilvus = importlib.import_module("pymilvus")
        except ImportError:
            logger.error("pymilvus not installed. Run: pip install pymilvus")
            raise

        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.connected = False
        self._connect()

    def _connect(self):
        try:
            from pymilvus import connections

            connections.connect("default", host=self.host, port=self.port)
            logger.info(f"Connected to Milvus at {self.host}:{self.port}")
            self.connected = True
            self._init_collection()
        except Exception as e:
            logger.warning(f"Failed to connect to Milvus: {e}")
            self.connected = False

    def _init_collection(self):
        if not self.connected:
            return
        from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, utility

        if not utility.has_collection(self.collection_name):
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1536),  # Default dim
                FieldSchema(name="metadata_json", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="island", dtype=DataType.INT64),  # Scalar field for filtering
            ]
            schema = CollectionSchema(fields, "OpenEvolve Program Embeddings")
            self.collection = Collection(self.collection_name, schema)

            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128},
            }
            self.collection.create_index(field_name="vector", index_params=index_params)
            logger.info(f"Created Milvus collection: {self.collection_name}")
        else:
            self.collection = Collection(self.collection_name)
            self.collection.load()

    def add(self, id: str, vector: list[float], metadata: dict[str, Any]):
        if not self.connected:
            return

        island = int(metadata.get("island", -1))

        data = [[id], [vector], [json.dumps(metadata)], [island]]
        self.collection.insert(data)
        # Flush is expensive, relying on auto-flush or manual flush elsewhere is better
        # But for correctness in this loop, we might need it visible
        # self.collection.flush()

    def search(
        self, vector: list[float], limit: int = 10, filters: str = None
    ) -> list[dict[str, Any]]:
        if not self.connected:
            return []

        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}

        # Convert simple "island == X" filter to Milvus expression
        expr = None
        if filters:
            try:
                # Basic parsing for "island == 1"
                # Milvus supports "island == 1" directly if island is a scalar field
                expr = filters
            except:
                pass

        results = self.collection.search(
            data=[vector],
            anns_field="vector",
            param=search_params,
            limit=limit,
            expr=expr,
            output_fields=["metadata_json"],
        )

        out = []
        for hits in results:
            for hit in hits:
                out.append(
                    {
                        "id": hit.id,
                        "score": hit.score,
                        "metadata": json.loads(hit.entity.get("metadata_json")),
                    }
                )
        return out


def get_vector_store(type_name="memory", **kwargs):
    if type_name == "milvus":
        return MilvusVectorStore(**kwargs)
    return InMemoryVectorStore()


# --- Embedding Client (Legacy + New) ---


class EmbeddingClient:
    """
    Client for generating embeddings and interfacing with the vector store.
    """

    def __init__(
        self, model: str = "text-embedding-3-small", vector_store_type: str = "memory", **kwargs
    ):
        self.model = model
        self.store = get_vector_store(vector_store_type, **kwargs)
        self.client = None
        self._setup_client()

    def _setup_client(self):
        try:
            import openai

            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key:
                self.client = openai.OpenAI(api_key=api_key)
        except ImportError:
            logger.debug("OpenAI client not found. Embeddings will be random.")

    def get_embedding(self, text: str) -> list[float]:
        """Generate embedding for text."""
        if not self.client:
            # Mock mode
            return [random.random() for _ in range(1536)]

        try:
            text = text.replace("\n", " ")
            # Truncate if too long (rough check)
            if len(text) > 30000:
                text = text[:30000]

            return self.client.embeddings.create(input=[text], model=self.model).data[0].embedding
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return [0.0] * 1536

    def add_program(self, program: Any):
        """Add program to vector store."""
        vec = self.get_embedding(program.code)

        # Store essential metadata
        meta = {
            "island": program.metadata.get("island", 0),
            "generation": program.generation,
            "score": program.metrics.get("combined_score", 0.0),
        }

        self.store.add(program.id, vec, meta)

        # Legacy compatibility: attach to program object
        program.embedding = vec

    def find_similar(self, text: str, limit: int = 1, filters: str = None) -> list[dict[str, Any]]:
        """Search for similar programs."""
        vec = self.get_embedding(text)
        return self.store.search(vec, limit=limit, filters=filters)
