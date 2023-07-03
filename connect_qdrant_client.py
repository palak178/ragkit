import os
from qdrant_client import QdrantClient
from qdrant_client import models

qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_KEY"),
    timeout=60
)

