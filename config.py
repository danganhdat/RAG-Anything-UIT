import os
from dotenv import load_dotenv
load_dotenv()

MILVUS_DB_PATH = os.getenv("MILVUS_DB_PATH", "./milvus_lite.db")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "rag_multimodal_collection")
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))
# Basic runtime parameters
TIMEOUT = 60
MAX_RETRIES = 2