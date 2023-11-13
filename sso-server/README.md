## Dependencies

- FastAPI
- Redis
- Milvus
- TensorFlow

### Summary

The SSO-Server leverages FastAPI for serving. The app creates dedicated processes for handling model predictions using Python concurrency package. Embeddings sent by the frontend are re-encoded with a custom-trained model to reduce dimensionality. The resulting encodings are saved in the Milvus database.

### Authentication Process

1. When a user wants to authenticate, the server searches the encoding using L2 distance search in the vector database.
2. If the user is found, an SSO code is generated and stored in the Redis server.
3. The SSO code is then sent back to the user.
4. Users can use this code to obtain a JWT token for accessing the rest of the system securely.

## How to Use
1. run `pip install -r requirements.txt` to install dependencies.
2. run `alembic upgrade head` to do sql migration.
3. run `setup` function in `repository.milvus` to do vector db migration.
4. run `python main.py` to start the server.

Go to `localhost:8080/docs` to view openApi documentation.