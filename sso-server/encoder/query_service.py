import logging
import uuid
from fastapi import Depends
import tensorflow as tf
import numpy as np

from repository.milvus import VectorDB, Vector
from encoder.executors import encoder

L2_THRESHOLD = 5
class VectorQuery:
    def __init__(self, vector_db:VectorDB = Depends(VectorDB)):
        self.vector_db = vector_db
        
    @staticmethod
    def predict(vector: Vector) -> np.ndarray:
        tensor = tf.constant(vector.vector)
        tensors = tf.reshape(tensor, (1, len(vector.vector)))
        logging.info(f'Converted enbedding to tensor with shape={tensor.shape}')
        return encoder.predict(tensors)[0]

    def authenticate(self, vector: Vector) -> uuid.UUID:
        encodings = self.predict(vector)
        search_result = self.vector_db.query(encodings.tolist())
        candedicate = max(search_result, key=lambda x: x.distance)
        if candedicate.distance > L2_THRESHOLD:
            return None
        return uuid.UUID(hex=candedicate.entity.get('user_id'))