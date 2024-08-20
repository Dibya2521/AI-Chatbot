from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from settings import Settings
from timings import time_it, logger
import pandas as pd
import json

def clean_doc(doc):
    if doc.metadata.get("tables"):
        clean_data = []
        for table in doc.metadata["tables"]:
            clean_data.append(table.to_json(orient="records"))
        return [Document(page_content=json.dumps(clean_data), metadata=doc.metadata)]
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=Settings.CHUNK_SIZE,
            chunk_overlap=Settings.CHUNK_OVERLAP,
            length_function=len,
        )
        return splitter.split_documents([doc])

@time_it
def clean_docs(documents):
    chunks = []
    for doc in documents:
        chunks.extend(clean_doc(doc))
    
    logger.info(f"Split {len(documents)} docs into {len(chunks)} chunks")
    return chunks
