from settings import Settings
from pdf_reader import PDFReader
from embedder import Embedder
from db_manager import DBManager
from query_handler import QueryHandler
from responder import Responder
from timings import logger
import time

class MemoryContext:
    def __init__(self, max_history=5):
        self.history = []
        self.max_history = max_history

    def add(self, question, answer):
        self.history.append((question, answer))
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def get_context(self):
        return "\n\n".join([f"Q: {q}\nA: {a}" for q, a in self.history])

def setup():
    try:
        Settings.check()
        
        pdfs = PDFReader.read_all_pdfs()
        
        embedder = Embedder()
        db = DBManager(embedder)
        db.add_docs(pdfs)
        
        query_handler = QueryHandler(db)
        responder = Responder()
        memory = MemoryContext()
        
        logger.info("Setup complete")
        return query_handler, responder, memory
    except Exception as e:
        logger.error(f"Setup error: {str(e)}")
        raise

def run():
    try:
        query_handler, responder, memory = setup()
        
        while True:
            query = input("Ask a question (or 'quit' to exit): ")
            if query.lower() == 'quit':
                break
            
            context = query_handler.handle(query)
            memory_context = memory.get_context()
            
            full_context = f"{memory_context}\n\n{context}"
            
            answer = responder.respond(query, full_context)
            print(f"Answer: {answer}")
            
            if "technical difficulties" not in answer and "unable to provide an answer" not in answer:
                memory.add(query, answer)
            else:
                print("Apologies for the inconvenience. Please wait a moment before asking another question.")
                time.sleep(10)  
    except Exception as e:
        logger.error(f"Runtime error: {str(e)}")

if __name__ == "__main__":
    run()