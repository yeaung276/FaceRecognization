from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
from api_schema.requests import Vector

import env

fields = [
    FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="user_id", dtype=DataType.BINARY_VECTOR),
    FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=16)
]

schema = CollectionSchema(fields, "user_schema")
index_params = {
  "metric_type":"L2",
  "index_type":"IVF_FLAT",
  "params":{"nlist":1024}
}

class VectorDB:
    def __init__(self):
        connections.connect(
            host=env.MILVUS_DATABASE_URL,
            port=env.MILVUS_DATABASE_PORT
        )
        self.collection = Collection("user_embeddings", schema)
        if not self.collection.has_index():
            self.collection.create_index(
                field_name='embeddings',
                index_params=index_params
            )

    def insert(self, vector: Vector):
        pass

    def query(self, vector: Vector):
        pass

    def __del__(self):
        connections.disconnect()
        
    
        
    