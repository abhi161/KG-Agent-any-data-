# unified_kg/core/data_processor.py
import pandas as pd
import fitz  # PyMuPDF
import pdfplumber
from typing import List, Dict, Any, Tuple, Optional
import logging
import json
import os

logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Processor for handling both structured (CSV) and unstructured (PDF) data
    """
    
    def __init__(self, llm, chunk_size=3000, chunk_overlap=200):
        self.llm = llm
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def process_csv(self, file_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Process a CSV file and identify potential entity types
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Tuple of (DataFrame, metadata)
        """
        logger.info(f"Processing CSV file: {file_path}")
        
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Remove empty rows and columns
        df = df.dropna(how='all').fillna('')
        
        # Extract metadata
        metadata = {
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": list(df.columns)
        }
        
        # Infer column entity types
        column_mappings = self._infer_column_types(df)
        metadata["column_mappings"] = column_mappings
        
        return df, metadata
    
    def _infer_column_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Use LLM to infer entity types from column names and data
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary mapping column names to entity types
        """
        # Sample data to help inference
        sample_data = df.head(5).to_dict('records')
        
        prompt = f"""
        Analyze these CSV columns and data samples:
        
        Column names: {list(df.columns)}
        
        Sample data:
        {json.dumps(sample_data, indent=2)}
        
        For each column, determine if it represents a distinct entity type.
        Consider columns like IDs, names, titles, etc. as potential entities.
        Ignore columns that are just attributes or measurements.
        
        Return a JSON object mapping column names to entity types.
        Use format: {{"column_name": "EntityType", ...}}
        Only include columns that represent entities.
        """
        
        response = self.llm.predict(prompt)
        
        # Extract JSON from response
        try:
            # Find JSON pattern in the response
            import re
            json_match = re.search(r'\{.*\}', response.replace('\n', ''), re.DOTALL)
            if json_match:
                mappings = json.loads(json_match.group(0))
                return mappings
            return {}
        except Exception as e:
            logger.warning(f"Error parsing column type inference: {e}")
            return {}
    
    def process_pdf(self, file_path: str) -> Tuple[List[str], Dict[str, Any]]:
        """
        Process a PDF file and extract content as text chunks
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Tuple of (text chunks, metadata)
        """
        logger.info(f"Processing PDF file: {file_path}")
        
        # Extract text from PDF
        text = self._extract_text_from_pdf(file_path)
        
        # Split into chunks
        chunks = self._split_text(text)
        
        # Extract metadata
        metadata = {
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "chunk_count": len(chunks),
            "total_chars": len(text)
        }
        
        return chunks, metadata
    
    def _extract_text_from_pdf(self, file_path: str) -> str:
        """
        Extract text from PDF using PyMuPDF
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text as string
        """
        try:
            text_content = []
            
            # Try extracting text directly with PyMuPDF
            doc = fitz.open(file_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                text_content.append(text)
            
            # If extracted text is too short, try with pdfplumber
            full_text = "\n".join(text_content)
            if len(full_text.strip()) < 100:
                logger.info(f"Text extraction with PyMuPDF yielded short text, trying pdfplumber")
                text_content = []
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        text = page.extract_text() or ""
                        text_content.append(text)
            
            return "\n".join(text_content)
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return ""
    
    def _split_text(self, text: str) -> List[str]:
        """
        Split text into chunks of specified size with overlap
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
            
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            
            # Try to find a sentence or paragraph break
            if end < len(text):
                # Look for paragraph breaks
                paragraph_break = text.rfind('\n\n', start, end)
                if paragraph_break != -1 and paragraph_break > start + self.chunk_size // 2:
                    end = paragraph_break + 2
                else:
                    # Look for line breaks
                    line_break = text.rfind('\n', start, end)
                    if line_break != -1 and line_break > start + self.chunk_size // 2:
                        end = line_break + 1
                    else:
                        # Look for sentence breaks
                        sentence_break = max(
                            text.rfind('. ', start, end),
                            text.rfind('? ', start, end),
                            text.rfind('! ', start, end)
                        )
                        if sentence_break != -1 and sentence_break > start + self.chunk_size // 2:
                            end = sentence_break + 2
            
            chunks.append(text[start:end])
            start = max(start + self.chunk_size - self.chunk_overlap, end - self.chunk_overlap)
        
        return chunks
    
    def extract_entities_from_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract entities from text using LLM
        
        Args:
            text: Text to process
            
        Returns:
            List of extracted entities
        """
        prompt = f"""
        Extract key entities from the following text. For each entity, identify:
        1. The entity name
        2. The entity type (e.g., Person, Organization, Location, Product, etc.)
        3. Any important attributes mentioned about the entity
        
        Text:
        {text[:4000]}  # Limit text to 4000 chars to avoid token limits
        
        Return entities in the following JSON format:
        [
            {{
                "name": "entity name",
                "type": "EntityType",
                "attributes": {{"attribute1": "value1", "attribute2": "value2"}}
            }}
        ]
        
        Only include clear entities mentioned in the text.
        """
        
        response = self.llm.predict(prompt)
        
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\[.*\]', response.replace('\n', ''), re.DOTALL)
            if json_match:
                entities = json.loads(json_match.group(0))
                return entities
            return []
        except Exception as e:
            logger.warning(f"Error parsing entity extraction: {e}")
            return []
    
    def extract_relations_from_text(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract relationships between entities in text
        
        Args:
            text: Text to process
            entities: List of entities found in the text
            
        Returns:
            List of extracted relationships
        """
        if len(entities) < 2:
            return []
            
        entity_list = [f"{e['name']} ({e['type']})" for e in entities]
        
        prompt = f"""
        Identify relationships between the following entities mentioned in the text:
        
        Entities:
        {json.dumps(entity_list, indent=2)}
        
        Text:
        {text[:4000]}  # Limit text to 4000 chars to avoid token limits
        
        For each relationship you find, specify:
        1. The source entity
        2. The relationship type (e.g., WORKS_FOR, LOCATED_IN, PRODUCES, etc.)
        3. The target entity
        
        Return relationships in the following JSON format:
        [
            {{
                "source": "source entity name",
                "relation": "RELATIONSHIP_TYPE",
                "target": "target entity name"
            }}
        ]
        
        Only include clear relationships explicitly mentioned in the text.
        """
        
        response = self.llm.predict(prompt)
        
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\[.*\]', response.replace('\n', ''), re.DOTALL)
            if json_match:
                relations = json.loads(json_match.group(0))
                return relations
            return []
        except Exception as e:
            logger.warning(f"Error parsing relation extraction: {e}")
            return []