import fitz
import pandas as pd
from langchain.schema import Document
from settings import Settings
from timings import logger, time_it
import os
import re

class PDFReader:
    @staticmethod
    @time_it
    def read_pdf(path):
        try:
            doc = fitz.open(path)
            text = ""
            tables = []
            for page in doc:
                page_text = page.get_text()
                text += page_text
                tables.extend(PDFReader._extract_table_data(page_text))
            
            clean_tables = PDFReader._clean_tables(tables)
            return text, clean_tables
        except Exception as e:
            logger.error(f"Error reading PDF {path}: {str(e)}")
            return "", []

    @staticmethod
    def _extract_table_data(text):
        lines = text.split('\n')
        table_data = []
        current_table = []
        in_table = False
        
        for line in lines:
            if re.match(r'^(FY\s+\d{4}|Appropriation)', line):
                if in_table and current_table:
                    table_data.append(current_table)
                    current_table = []
                in_table = True
            
            if in_table:
                columns = re.split(r'\s{2,}', line.strip())
                if len(columns) >= 2:
                    current_table.append(columns)
        
        if current_table:
            table_data.append(current_table)
        
        return table_data

    @staticmethod
    def _clean_tables(tables):
        clean = []
        for table in tables:
            if len(table) > 1: 
                header = table[0]
                data = table[1:]
                num_columns = max(len(row) for row in data)
                if len(header) < num_columns:
                    header.extend([''] * (num_columns - len(header)))
                data = [row + [''] * (num_columns - len(row)) for row in data]
                
                df = pd.DataFrame(data, columns=header)
                df = df.apply(lambda x: x.str.replace(',', '').str.strip())
                
                for col in df.columns:
                    try:
                        df[col] = pd.to_numeric(df[col])
                    except ValueError:
                        pass
                
                clean.append(df)
        return clean

    @staticmethod
    @time_it
    def read_all_pdfs():
        docs = []
        pdf_folder = Settings.PDF_FOLDER
        if not os.path.exists(pdf_folder):
            logger.error(f"PDF folder not found: {pdf_folder}")
            return docs

        for file in os.listdir(pdf_folder):
            if file.endswith(".pdf"):
                pdf_path = os.path.join(pdf_folder, file)
                logger.info(f"Reading PDF: {file}")
                text, tables = PDFReader.read_pdf(pdf_path)
                if text or tables:
                    doc = Document(page_content=text, metadata={"source": file, "tables": tables})
                    docs.append(doc)
                else:
                    logger.warning(f"Skipping {file} - empty content")
        
        logger.info(f"Read {len(docs)} PDF documents")
        return docs