from typing import List, Dict, Any, Tuple, Optional
import logging
import json
import os
import pandas as pd
import re

# LangChain imports
from langchain.document_loaders import PyPDFLoader, CSVLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain.chains import LLMChain
from langchain.embeddings.base import Embeddings

logger = logging.getLogger(__name__)

# Pydantic models for structured output parsing
class Entity(BaseModel):
    name: str = Field(description="The name of the entity")
    type: str = Field(description="The type of entity (e.g., Person, Organization, Location)")
    attributes: Dict[str, Any] = Field(default_factory=dict, description="Additional attributes of the entity")
    context: Optional[str] = Field(default=None, description="Surrounding context where the entity was mentioned")

class Relation(BaseModel):
    source: str = Field(description="The name of the source entity")
    relation: str = Field(description="The type of relationship between the entities")
    target: str = Field(description="The name of the target entity")
    context: Optional[str] = Field(default=None, description="Surrounding context where the relationship was mentioned")

class EntitiesOutput(BaseModel):
    entities: List[Entity] = Field(description="List of entities extracted from the text")

class RelationsOutput(BaseModel):
    relations: List[Relation] = Field(description="List of relationships extracted from the text")

class ColumnMappingsOutput(BaseModel):
    column_mappings: Dict[str, str] = Field(description="Mapping of column names to entity types")

class DataProcessor:
    """
    Processor for handling both structured (CSV) and unstructured (PDF) data
    using LangChain components with vector embedding support
    """
    
    def __init__(self, llm, embeddings: Embeddings = None, chunk_size=3000, chunk_overlap=200):
        self.llm = llm
        self.embeddings = embeddings
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False
        )
        
        # Set up the entity extraction chain with context capturing
        entity_parser = PydanticOutputParser(pydantic_object=EntitiesOutput)
        entity_prompt = PromptTemplate(
            template="""
            Extract key entities from the following text. For each entity, identify:
            1. The entity name
            2. The entity type (e.g., Person, Organization, Location, Product, etc.)
            3. Any important attributes mentioned about the entity
            4. The immediate context around where the entity is mentioned (capture a sentence or phrase where the entity appears)

            Text:
            {text}

            {format_instructions}
            """,
            input_variables=["text"],
            partial_variables={"format_instructions": entity_parser.get_format_instructions()}
        )
        self.entity_chain = LLMChain(llm=llm, prompt=entity_prompt, output_parser=entity_parser)
        
        # Set up the relation extraction chain with context capturing
        relation_parser = PydanticOutputParser(pydantic_object=RelationsOutput)
        relation_prompt = PromptTemplate(
            template="""
            Identify relationships between entities in the following text.
            
            Text:
            {text}
            
            Entities found:
            {entities}
            
            For each relationship you find, specify:
            1. The source entity name
            2. The relationship type (e.g., WORKS_FOR, LOCATED_IN, PRODUCES, etc.)
            3. The target entity name
            4. The immediate context around where the relationship is mentioned (capture a sentence or phrase that shows the relationship)
            
            {format_instructions}
            """,
            input_variables=["text", "entities"],
            partial_variables={"format_instructions": relation_parser.get_format_instructions()}
        )
        self.relation_chain = LLMChain(llm=llm, prompt=relation_prompt, output_parser=relation_parser)
        
        # Set up the column mapping chain
        column_parser = PydanticOutputParser(pydantic_object=ColumnMappingsOutput)
        column_prompt = PromptTemplate(
            template="""
            Analyze these CSV columns and data samples:
            
            Column names: {column_names}
            
            Sample data:
            {sample_data}
            
            For each column, determine if it represents a distinct entity type.
            Consider columns like IDs, names, titles, etc. as potential entities.
            Ignore columns that are just attributes or measurements.
            
            {format_instructions}
            """,
            input_variables=["column_names", "sample_data"],
            partial_variables={"format_instructions": column_parser.get_format_instructions()}
        )
        self.column_chain = LLMChain(llm=llm, prompt=column_prompt, output_parser=column_parser)
        
        # Store metadata about processed documents
        self.document_metadata = {}
        
        # Document embeddings cache
        self.chunk_embeddings = {}
    
    def process_csv(self, file_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Process a CSV file and identify potential entity types
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Tuple of (DataFrame, metadata)
        """
        logger.info(f"Processing CSV file: {file_path}")
        
        # Load CSV using LangChain's loader
        try:
            loader = CSVLoader(file_path=file_path)
            documents = loader.load()
            
            # Convert back to pandas DataFrame for easier processing
            df = pd.read_csv(file_path)
            
            # Remove empty rows and columns
            df = df.dropna(how='all').fillna('')
            
            # Generate embeddings for each row if embeddings model is available
            if self.embeddings:
                row_embeddings = []
                for _, row in df.iterrows():
                    # Create a string representation of the row
                    row_text = " ".join([f"{col}: {val}" for col, val in row.items() if val])
                    embedding = self.embeddings.embed_query(row_text)
                    row_embeddings.append(embedding)
                
                # Store row embeddings in metadata
                row_embeddings_dict = {i: emb for i, emb in enumerate(row_embeddings)}
                
            # Extract metadata
            metadata = {
                "file_path": file_path,
                "file_name": os.path.basename(file_path),
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": list(df.columns)
            }
            
            # Infer column entity types using LangChain
            column_mappings = self._infer_column_types(df)
            metadata["column_mappings"] = column_mappings
            
            return df, metadata
        except Exception as e:
            logger.error(f"Error processing CSV file: {e}")
            raise
    
    def _infer_column_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Use LLM to infer entity types from column names and data using LangChain
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary mapping column names to entity types
        """
        # Sample data to help inference
        sample_data = df.head(5).to_dict('records')
        
        try:
            response = self.column_chain.run(
                column_names=list(df.columns),
                sample_data=json.dumps(sample_data, indent=2)
            )
            
            return response.column_mappings
        except Exception as e:
            logger.warning(f"Error parsing column type inference: {e}")
            return {}
    
    def process_pdf(self, file_path: str) -> Tuple[List[str], Dict[str, Any]]:
        """
        Process a PDF file and extract content as text chunks with embeddings
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Tuple of (text chunks, metadata)
        """
        logger.info(f"Processing PDF file: {file_path}")
        
        try:
            # Use LangChain's PyPDFLoader
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # If the documents are empty or very short, try TextLoader as fallback
            total_content = sum(len(doc.page_content) for doc in documents)
            if total_content < 100 and os.path.exists(file_path + '.txt'):
                logger.info(f"PDF content too short, trying TextLoader as fallback")
                loader = TextLoader(file_path + '.txt')
                documents = loader.load()
            
            # Split documents into smaller chunks using LangChain's splitter
            split_docs = self.text_splitter.split_documents(documents)
            
            # Generate embeddings for each chunk if embeddings model is available
            if self.embeddings:
                for i, doc in enumerate(split_docs):
                    embedding = self.embeddings.embed_query(doc.page_content)
                    self.chunk_embeddings[i] = embedding
            
            # Extract just the text content to maintain compatibility with existing code
            text_chunks = [doc.page_content for doc in split_docs]
            
            # Store the document metadata for reference
            self.document_metadata = {
                i: doc.metadata for i, doc in enumerate(split_docs)
            }
            
            # Extract metadata
            metadata = {
                "file_path": file_path,
                "file_name": os.path.basename(file_path),
                "document_count": len(documents),
                "chunk_count": len(split_docs),
                "total_chars": sum(len(doc.page_content) for doc in documents)
            }
            
            return text_chunks, metadata
        except Exception as e:
            logger.error(f"Error processing PDF file: {e}")
            raise
    
    def extract_entities_from_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract entities from text using LangChain with context capturing
        
        Args:
            text: Text to process
            
        Returns:
            List of extracted entities in dict format for compatibility
        """
        try:
            # Truncate text if too long to avoid token limits
            truncated_text = text[:12000]  # Adjust based on your model's context limit
            
            result = self.entity_chain.run(text=truncated_text)
            
            # Extract entities with their context
            extracted_entities = []
            for entity in result.entities:
                # For each entity, find its occurrences in the text to enhance context
                entity_dict = {
                    "name": entity.name,
                    "type": entity.type,
                    "attributes": entity.attributes
                }
                
                # Include context if available
                if entity.context:
                    entity_dict["context"] = entity.context
                else:
                    # Try to find context programmatically
                    context = self._extract_entity_context(truncated_text, entity.name)
                    if context:
                        entity_dict["context"] = context
                
                # Generate and store entity-specific embedding if embeddings model is available
                if self.embeddings and "context" in entity_dict:
                    # Create rich context for embedding
                    embedding_text = f"Entity: {entity.name}\nType: {entity.type}\nContext: {entity_dict['context']}"
                    entity_dict["embedding"] = self.embeddings.embed_query(embedding_text)
                
                extracted_entities.append(entity_dict)
            
            return extracted_entities
        except Exception as e:
            logger.warning(f"Error extracting entities: {e}")
            return []
    
    def _extract_entity_context(self, text: str, entity_name: str) -> Optional[str]:
        """
        Extract context around an entity mention
        
        Args:
            text: Full text
            entity_name: Name of the entity
            
        Returns:
            Context string or None
        """
        # Find the entity in the text
        # Use regex to handle case variations and partial matches
        pattern = re.compile(r'[^.!?]*\b' + re.escape(entity_name) + r'\b[^.!?]*[.!?]', re.IGNORECASE)
        matches = pattern.findall(text)
        
        if matches:
            # Return the first match (most relevant context)
            return matches[0].strip()
        
        # Fallback to a simple window around the entity
        index = text.lower().find(entity_name.lower())
        if index != -1:
            start = max(0, index - 100)
            end = min(len(text), index + len(entity_name) + 100)
            return text[start:end].strip()
        
        return None
    
    def extract_relations_from_text(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract relationships between entities in text using LangChain with context capturing
        
        Args:
            text: Text to process
            entities: List of entities found in the text
            
        Returns:
            List of extracted relationships in dict format for compatibility
        """
        if len(entities) < 2:
            return []
            
        try:
            # Format entities for the prompt
            entity_list = [f"{e['name']} ({e['type']})" for e in entities]
            
            # Truncate text if too long
            truncated_text = text[:12000]  # Adjust based on your model's context limit
            
            result = self.relation_chain.run(
                text=truncated_text,
                entities=json.dumps(entity_list)
            )
            
            # Extract relations with their context
            extracted_relations = []
            for relation in result.relations:
                relation_dict = {
                    "source": relation.source,
                    "relation": relation.relation,
                    "target": relation.target
                }
                
                # Include context if available
                if relation.context:
                    relation_dict["context"] = relation.context
                else:
                    # Try to find context programmatically
                    context = self._extract_relation_context(truncated_text, relation.source, relation.target)
                    if context:
                        relation_dict["context"] = context
                
                # Generate relation embedding if embeddings model is available
                if self.embeddings and "context" in relation_dict:
                    # Create rich context for embedding
                    embedding_text = f"Relation: {relation.source} {relation.relation} {relation.target}\nContext: {relation_dict['context']}"
                    relation_dict["embedding"] = self.embeddings.embed_query(embedding_text)
                
                extracted_relations.append(relation_dict)
            
            return extracted_relations
        except Exception as e:
            logger.warning(f"Error extracting relations: {e}")
            return []
    
    def _extract_relation_context(self, text: str, source_entity: str, target_entity: str) -> Optional[str]:
        """
        Extract context around a relationship mention
        
        Args:
            text: Full text
            source_entity: Source entity name
            target_entity: Target entity name
            
        Returns:
            Context string or None
        """
        # Find sentences containing both entities
        sentences = re.split(r'[.!?]', text)
        
        for sentence in sentences:
            if source_entity.lower() in sentence.lower() and target_entity.lower() in sentence.lower():
                return sentence.strip()
        
        # Find paragraphs containing both entities if no single sentence found
        paragraphs = re.split(r'\n\n', text)
        
        for paragraph in paragraphs:
            if source_entity.lower() in paragraph.lower() and target_entity.lower() in paragraph.lower():
                # Limit paragraph size
                if len(paragraph) > 500:
                    return paragraph[:500] + "..."
                return paragraph.strip()
        
        return None
    
    def get_chunk_embedding(self, chunk_idx: int) -> Optional[List[float]]:
        """
        Get the embedding for a specific chunk
        
        Args:
            chunk_idx: Index of the chunk
            
        Returns:
            Embedding vector or None
        """
        return self.chunk_embeddings.get(chunk_idx)