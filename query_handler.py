from timings import time_it, logger
import json
import pandas as pd

class QueryHandler:
    def __init__(self, db_manager):
        self.db = db_manager

    @time_it
    def handle(self, query):
        try:
            docs = self.db.search(query)
            context = self._format_context(docs)
            logger.info(f"Handled query: '{query}' with {len(docs)} relevant docs")
            return context
        except Exception as e:
            logger.error(f"Error handling query: {str(e)}")
            return []

    def _format_context(self, documents):
        context = []
        for doc in documents:
            if doc.page_content.startswith('['):
                tables = json.loads(doc.page_content)
                for table in tables:
                    df = pd.read_json(table)
                    context.append(f"Table:\n{df.to_string(index=False)}")
            else:
                context.append(f"Text: {doc.page_content}")
        return "\n\n".join(context)
