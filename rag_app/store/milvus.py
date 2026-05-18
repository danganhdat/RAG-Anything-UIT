from __future__ import annotations

import logging
import time
from typing import Any

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from rag_app.core.config import Settings
from rag_app.core.exceptions import VectorStoreError

log = logging.getLogger(__name__)


class MilvusVectorStore:
    """Wrapper around Milvus Lite for vector CRUD operations."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._collection: Collection | None = None
        self._connect()

    def _connect(self) -> None:
        uri = self._settings.milvus_db_path
        name = self._settings.milvus_collection
        dim = self._settings.embed_dim

        try:
            connections.connect("default", uri=uri, timeout=30)
            log.info("Connected to Milvus at %s", uri)
        except Exception as exc:
            raise VectorStoreError(f"Cannot connect to Milvus at {uri}") from exc

        if utility.has_collection(name):
            self._collection = Collection(name)
            log.info("Using existing collection '%s'", name)
        else:
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=512, is_primary=True),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="content_type", dtype=DataType.VARCHAR, max_length=32),
                FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=1024),
                FieldSchema(name="ts", dtype=DataType.INT64),
            ]
            schema = CollectionSchema(fields, "Multimodal RAG collection")
            self._collection = Collection(name, schema)
            self._collection.create_index(
                "vector",
                {"metric_type": "COSINE", "index_type": "AUTOINDEX", "params": {}},
            )
            log.info("Created collection '%s' with AUTOINDEX", name)

        self._collection.load()

    @property
    def collection(self) -> Collection:
        if self._collection is None:
            raise VectorStoreError("Collection not initialized")
        return self._collection

    def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        contents: list[str],
        content_types: list[str],
        sources: list[str],
    ) -> None:
        n = len(ids)
        if not (len(vectors) == len(contents) == len(content_types) == len(sources) == n):
            raise ValueError(
                f"All lists must have equal length (got ids={n}, vectors={len(vectors)}, "
                f"contents={len(contents)}, types={len(content_types)}, sources={len(sources)})"
            )

        data = [ids, vectors, contents, content_types, sources, [int(time.time() * 1000)] * n]
        self.collection.upsert(data)
        self.collection.flush()
        log.info("Upserted %d records into '%s'", n, self._settings.milvus_collection)

    def search(
        self,
        query_vectors: list[list[float]],
        top_k: int = 5,
        content_type: str | None = None,
    ) -> list[list[dict[str, Any]]]:
        expr = f'content_type == "{content_type}"' if content_type else None
        params = {"metric_type": "COSINE", "params": {"nprobe": 16}}

        results = self.collection.search(
            data=query_vectors,
            anns_field="vector",
            param=params,
            limit=top_k,
            expr=expr,
            output_fields=["id", "content", "content_type", "source", "ts"],
        )

        return [
            [
                {
                    "id": hit.entity.get("id"),
                    "content": hit.entity.get("content"),
                    "content_type": hit.entity.get("content_type"),
                    "source": hit.entity.get("source"),
                    "score": hit.score,
                }
                for hit in hits
            ]
            for hits in results
        ]
