from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from settings import Settings
from timings import logger, time_it
import pickle
import os
import json

class DBManager:
    def __init__(self, embedder):
        self.embedder = embedder
        self.db_file = os.path.join(Settings.DB_FOLDER, "faiss_index.pkl")
        self.db = self._load_or_create_db()

    def _load_or_create_db(self):
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, "rb") as f:
                    db = pickle.load(f)
                logger.info("Loaded existing DB")
                return db
            except Exception as e:
                logger.error(f"Error loading DB: {str(e)}")
        
        logger.info("Creating new DB")
        return self._create_db()

    def _create_db(self):
        dummy_texts = ["Dummy text for init"]
        db = FAISS.from_texts(dummy_texts, self.embedder)
        logger.info("Created new DB")
        return db

    @time_it
    def add_docs(self, documents):
        if not documents:
            logger.warning("No docs to add")
            return
        
        try:
            for doc in documents:
                if isinstance(doc.page_content, list):
                    text = json.dumps(doc.page_content)
                else:
                    text = doc.page_content
                self.db.add_texts([text], metadatas=[doc.metadata])
            self._save_db()
            logger.info(f"Added {len(documents)} docs to DB")
        except Exception as e:
            logger.error(f"Error adding docs to DB: {str(e)}")
            raise

    @time_it
    def search(self, query, k=4):
        try:
            results = self.db.similarity_search(query, k=k)
            logger.info(f"Searched for: '{query}'")
            return results
        except Exception as e:
            logger.error(f"Error searching DB: {str(e)}")
            return []

    def _save_db(self):
        os.makedirs(os.path.dirname(self.db_file), exist_ok=True)
        try:
            with open(self.db_file, "wb") as f:
                pickle.dump(self.db, f)
            logger.info("DB saved")
        except Exception as e:
            logger.error(f"Error saving DB: {str(e)}")
            raise
