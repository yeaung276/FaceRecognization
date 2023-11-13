import uuid
from typing import List, List, Generator
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    Hit
)
from api_schema.requests import Vector

import env

fields = [
    FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=128),
    FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=16)
]

schema = CollectionSchema(fields, "user_schema")
index_params = {
  "metric_type":"L2",
  "index_type":"IVF_FLAT",
  "params":{"nlist":128}
}
search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

# should only be run at app start up to setup the vector db
def setup():
    connections.connect(
            host=env.MILVUS_DATABASE_URL,
            port=env.MILVUS_DATABASE_PORT
        )
    collection = Collection("user_embeddings", schema)
    collection.flush()
    if not collection.has_index(index_name="main_idx"):
        collection.create_index(
            index_name="main_idx",
            field_name='embeddings',
            index_params=index_params
        )

class VectorDB:
    def __init__(self):
        connections.connect(
            host=env.MILVUS_DATABASE_URL,
            port=env.MILVUS_DATABASE_PORT
        )
        self.collection = Collection("user_embeddings", schema)

    def insert(self, vector: List[float], user_id: uuid.UUID) -> None:
        self.collection.insert([[user_id.hex], [vector]])
        self.collection.flush()

    def query(self, vector: List[float]) -> List[Hit]:
        self.collection.load()
        results = self.collection.search(
            data=[vector], 
            anns_field="embeddings", 
            param=search_params, 
            limit=10, 
            expr=None,
            consistency_level="Strong",
            output_fields=['user_id']
        )
        return results[0]

    def __del__(self):
        connections.disconnect(alias='')
        
    
        
    