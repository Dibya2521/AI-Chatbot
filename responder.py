from timings import time_it, logger
from groq import Groq
from settings import Settings
import tiktoken
import pandas as pd
import json
import time

class Responder:
    def __init__(self):
        self.client = Groq(api_key=Settings.GROQ_KEY)
        self.model = Settings.GROQ_MODEL
        self.encoder = tiktoken.get_encoding("cl100k_base")
        self.max_tokens = 32768

    def _trim_context(self, context, query, max_tokens):
        sys_prompt = "You are a helpful assistant specializing in budget analysis. Provide accurate, detailed, and comprehensive information based on the given context, including specific figures when available. When analyzing tables, consider all columns and rows, and treat empty or missing values as zero. Use the conversation history to provide more informative answers."
        query_prompt = f"Question: {query}\n\nPlease provide a comprehensive and detailed answer based on the given context and conversation history. If specific budget figures are mentioned, include them in your response. If the information is not available, please say so. When analyzing tables, consider all columns and look for the most relevant data. Provide additional relevant information and context when possible."
        
        fixed_tokens = len(self.encoder.encode(sys_prompt + query_prompt))
        available_tokens = max_tokens - fixed_tokens - 8000 

        parts = context.split("\n\n")
        trimmed = []
        current_tokens = 0
        
        for part in parts:
            part_tokens = len(self.encoder.encode(part))
            if current_tokens + part_tokens <= available_tokens:
                trimmed.append(part)
                current_tokens += part_tokens
            else:
                break
        
        return "\n\n".join(trimmed)

    @time_it
    def respond(self, query, context):
        max_retries = 3
        retry_delay = 10  # seconds

        for attempt in range(max_retries):
            try:
                trimmed_context = self._trim_context(context, query, self.max_tokens)
                
                prompt = f"""
                Conversation history and context:
                {trimmed_context}

                Current question: {query}

                Please provide a comprehensive, detailed, and informative answer based on the given context and conversation history. If specific budget figures are mentioned, include them in your response. If the information is not available, please say so. When analyzing tables, consider all columns and look for the most relevant data. Treat empty or missing values as zero. Provide additional relevant information and context when possible. Your response should be thorough and at least 150 words long.
                """

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant specializing in budget analysis. Provide accurate, detailed, and comprehensive information based on the given context, including specific figures when available. When analyzing tables, consider all columns and rows, and treat empty or missing values as zero. Use the conversation history to provide more informative answers."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    max_tokens=8000 
                )

                answer = response.choices[0].message.content
                logger.info(f"Generated response for: '{query}'")
                return answer.strip()
            except Exception as e:
                if "rate_limit_exceeded" in str(e) and attempt < max_retries - 1:
                    logger.warning(f"Rate limit exceeded. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Error generating response: {str(e)}")
                    return "I apologize, but I'm currently experiencing technical difficulties. Please try again later or rephrase your question."

        return "I'm sorry, but I'm unable to provide an answer at this time due to technical issues. Please try again later."
