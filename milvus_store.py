import json
import time
from typing import List, Dict, Any, Optional
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
from config import MILVUS_DB_PATH, MILVUS_COLLECTION, EMBED_DIM

class MilvusVectorStore:
    def __init__(self, uri: str = MILVUS_DB_PATH, collection_name: str = MILVUS_COLLECTION, dim: int = EMBED_DIM):
        self.uri = uri
        self.collection_name = collection_name
        self.dim = dim
        self.collection: Optional[Collection] = None
        self._connect_and_prepare()

    def _connect_and_prepare(self):
        connections.connect("default", uri=self.uri)
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
        else:
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=512, is_primary=True),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="content_type", dtype=DataType.VARCHAR, max_length=32),
                FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=1024),
                FieldSchema(name="ts", dtype=DataType.INT64),
            ]
            schema = CollectionSchema(fields, "Minimal multimodal collection")
            self.collection = Collection(self.collection_name, schema)
            self.collection.create_index("vector", {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            })
        self.collection.load()

    def upsert(self, ids: List[str], vectors: List[List[float]], contents: List[str],
               content_types: List[str], sources: List[str]) -> None:
        data = [
            ids,
            vectors,
            contents,
            content_types,
            sources,
            [int(time.time() * 1000)] * len(ids)
        ]
        self.collection.upsert(data)
        self.collection.flush()

    def search(self, query_vectors: List[List[float]], top_k: int = 5, content_type: Optional[str] = None):
        expr = f'content_type == "{content_type}"' if content_type else None
        params = {"metric_type": "COSINE", "params": {"nprobe": 16}}
        results = self.collection.search(
            data=query_vectors,
            anns_field="vector",
            param=params,
            limit=top_k,
            expr=expr,
            output_fields=["id", "content", "content_type", "source", "ts"]
        )
        out = []
        for hits in results:
            out.append([{
                "id": h.entity.get("id"),
                "content": h.entity.get("content"),
                "content_type": h.entity.get("content_type"),
                "source": h.entity.get("source"),
                "score": h.score
            } for h in hits])
        return out
