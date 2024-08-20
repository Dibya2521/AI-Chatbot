from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from settings import Settings
from timings import time_it, logger
import numpy as np

class Embedder(Embeddings):
    def __init__(self):
        try:
            self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            logger.info("Embedding model loaded")
        except Exception as e:
            logger.error(f"Error loading embedder: {str(e)}")
            raise

    @time_it
    def embed_documents(self, texts):
        try:
            embeddings = self.model.encode(texts)
            logger.info(f"Embedded {len(texts)} docs")
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error embedding docs: {str(e)}")
            raise

    @time_it
    def embed_query(self, text):
        try:
            embedding = self.model.encode([text])[0]
            logger.info("Query embedded")
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error embedding query: {str(e)}")
            raise

    def embed_many(self, texts):
        return self.embed_documents(texts)
