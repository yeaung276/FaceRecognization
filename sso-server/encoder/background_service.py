import uuid
import asyncio
import logging

from fastapi import Depends, BackgroundTasks
import tensorflow as tf
import numpy as np

from repository.milvus import VectorDB
from api_schema.requests import Vector
from encoder.executors import encoder, executor

class BackgroundService:
    def __init__(self, 
                 background_task: BackgroundTasks, 
                 vector_db:VectorDB=Depends(VectorDB)) -> None:
        self.vector_db = vector_db
        self.background_task = background_task
        
    @staticmethod
    def predict(vector: Vector) -> np.ndarray:
        tensor = tf.constant(vector.vector)
        tensors = tf.reshape(tensor, (1, len(vector.vector)))
        logging.info(f'Converted enbedding to tensor with shape={tensor.shape}')
        return encoder.predict(tensors)[0]
        
    async def encode_vector(self, vector: Vector, user_id: uuid.UUID):
        loop = asyncio.get_event_loop()
        try:
            logging.info(f'Encoding process started. id={user_id}')
            encoding = await loop.run_in_executor(executor, self.predict, vector)
            logging.info(f'Saving encoding to the database. id={user_id}')

            self.vector_db.insert(encoding.tolist(), user_id)
            
        except Exception as e:
            logging.info(f'Error saving encoding. id={user_id}, error={e}')
            raise e
    
    def start_add_vector_task(self, vector: Vector, user_id: uuid.UUID):
        self.background_task.add_task(self.encode_vector, vector, user_id)
        