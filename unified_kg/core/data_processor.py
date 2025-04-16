import os
import re
import logging
import json
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
import datetime
# LangChain imports
from langchain.document_loaders import PyPDFLoader, CSVLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from pydantic import BaseModel, Field, validator
from langchain.chains import LLMChain
from langchain.llms.base import BaseLLM # Using base class for type hint
from langchain.embeddings.base import Embeddings # Using base class for type hint


logger = logging.getLogger(__name__)

# --- Pydantic Models ---
class Entity(BaseModel):
    name: str = Field(description="The specific name of the entity mentioned in the text")
    type: str = Field(description="A concise entity type (e.g., Person, Organization, Location, Drug, Disease, Chemical Compound, Medical Device)")
    attributes: Dict[str, Any] = Field(default_factory=dict, description="Key attributes or properties of the entity mentioned (e.g., dosage for a drug, symptom for a disease)")
    context: Optional[str] = Field(default=None, description="The sentence or short phrase where the entity was mentioned, providing context")

    @validator('name', 'type')
    def must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Name and type must not be empty")
        return v.strip()

class Relation(BaseModel):
    source: str = Field(description="The name of the source entity in the relationship")
    relation: str = Field(description="The type of relationship (verb phrase, e.g., TREATS, CAUSES, INTERACTS_WITH, MANUFACTURED_BY)")
    target: str = Field(description="The name of the target entity in the relationship")
    context: Optional[str] = Field(default=None, description="The sentence or short phrase where the relationship was mentioned")

    @validator('source', 'relation', 'target')
    def must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Source, relation, and target must not be empty")
        return v.strip()

class EntitiesOutput(BaseModel):
    entities: List[Entity] = Field(description="List of entities extracted from the text")

class RelationsOutput(BaseModel):
    relations: List[Relation] = Field(description="List of relationships extracted from the text")

class ColumnMapping(BaseModel):
     column_name: str = Field(description="The original name of the CSV column.")
     entity_type: str = Field(description="The primary entity type this column represents (e.g., Patient, Drug, Company, Location). Use 'Attribute' if it's just a property and not a distinct entity.")

class ColumnMappingsOutput(BaseModel):
    # Changed structure slightly for potentially better parsing
    column_mappings: List[ColumnMapping] = Field(description="Mapping of relevant column names to their primary entity types or 'Attribute'.")


# --- DataProcessor Class ---
class DataProcessor:
    """ Processor for structured (CSV) and unstructured (PDF/Text) data. """

    def __init__(self, llm: BaseLLM, embeddings: Optional[Embeddings] = None, chunk_size=1000, chunk_overlap=150):
        self.llm = llm
        self.embeddings = embeddings # Can be None
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            add_start_index=True, # Helpful for context mapping
        )

        # --- LLM Chains ---
        # Entity Extraction Chain
        entity_parser = PydanticOutputParser(pydantic_object=EntitiesOutput)
        entity_prompt = PromptTemplate(
            template="""Extract key entities from the text below. Focus on specific names and classify them into relevant types like Person, Organization, Location, Drug, Disease, Brand Name, Chemical Compound, etc. For each entity, provide its name, type, any key attributes mentioned directly with it, and the sentence where it appears.

Text:
{text}

{format_instructions}""",
            input_variables=["text"],
            partial_variables={"format_instructions": entity_parser.get_format_instructions()}
        )
        self.entity_chain = LLMChain(llm=llm, prompt=entity_prompt) # Output parser applied later if needed
        self.entity_parser = entity_parser
        # Add OutputFixingParser for resilience
        self.entity_fixing_parser = OutputFixingParser.from_llm(parser=entity_parser, llm=llm)


        # Relation Extraction Chain
        relation_parser = PydanticOutputParser(pydantic_object=RelationsOutput)
        relation_prompt = PromptTemplate(
            template="""Identify relationships between the entities listed below, based *only* on the provided text context. Relationships should be directional (source -> relation -> target). Use clear, concise relationship types (e.g., TREATS, CAUSES, INTERACTS_WITH, MANUFACTURED_BY, IS_A, PART_OF). Provide the source entity name, relation type, target entity name, and the sentence showing the relationship.

Text Context:
{text}

Entities Found (Name - Type):
{entities_list_str}

{format_instructions}""",
            input_variables=["text", "entities_list_str"],
            partial_variables={"format_instructions": relation_parser.get_format_instructions()}
        )
        self.relation_chain = LLMChain(llm=llm, prompt=relation_prompt)
        self.relation_parser = relation_parser
        self.relation_fixing_parser = OutputFixingParser.from_llm(parser=relation_parser, llm=llm)


        # Column Type Inference Chain
        column_parser = PydanticOutputParser(pydantic_object=ColumnMappingsOutput)
        column_prompt = PromptTemplate(
            template="""Analyze the following CSV column names and sample data to determine the primary *entity type* each column represents. If a column seems to represent a distinct real-world concept (like a person, product, company, location, medical record), assign an appropriate entity type (e.g., Patient, Drug, Manufacturer, Hospital, Prescription). If the column represents a simple attribute, measurement, date, or identifier that doesn't stand alone as an entity, classify it as 'Attribute'.

Column names: {column_names}

Sample data (first 5 rows):
{sample_data}

{format_instructions}""",
            input_variables=["column_names", "sample_data"],
            partial_variables={"format_instructions": column_parser.get_format_instructions()}
        )
        self.column_chain = LLMChain(llm=llm, prompt=column_prompt)
        self.column_parser = column_parser
        self.column_fixing_parser = OutputFixingParser.from_llm(parser=column_parser, llm=llm)

        # Store metadata about processed chunks (page numbers, etc.)
        self.document_metadata: Dict[int, Dict] = {} # Map chunk index to metadata
        # Document embeddings cache (optional, if needed elsewhere)
        # self.chunk_embeddings: Dict[int, List[float]] = {}

        logger.info("DataProcessor initialized.")

    def process_csv(self, file_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """ Process CSV: Load data, infer column types. """
        logger.info(f"Processing CSV file: {file_path}")
        try:
            # Use pandas directly for more control over reading options
            # Consider adding options like encoding, delimiter detection if needed
            df = pd.read_csv(file_path, keep_default_na=True, low_memory=False) # Keep NaN initially

            # Basic cleaning
            df.columns = df.columns.str.strip() # Strip whitespace from headers
            df = df.dropna(how='all') # Drop rows where all values are NaN
            # Replace NaN with empty string AFTER dropping all-NaN rows
            df = df.fillna('')

            logger.info(f"Loaded CSV '{os.path.basename(file_path)}': {len(df)} rows, {len(df.columns)} columns.")

            metadata = {
                "file_path": file_path,
                "file_name": os.path.basename(file_path),
                "row_count": len(df),
                "column_count": len(df.columns),
                "columns": list(df.columns)
            }

            # Infer column entity types using LLM chain
            column_mappings = self._infer_column_types(df) # Returns Dict[str, str]
            metadata["column_mappings"] = column_mappings
            logger.info(f"Inferred column mappings: {column_mappings}")

            # Row embeddings generation removed for simplicity, focus on KG structure first.
            # If needed, embeddings could be generated here per row.

            return df, metadata

        except FileNotFoundError:
             logger.error(f"CSV file not found: {file_path}")
             raise
        except pd.errors.EmptyDataError:
             logger.warning(f"CSV file is empty: {file_path}")
             # Return empty DataFrame and basic metadata
             return pd.DataFrame(columns=[]), {"file_path": file_path, "file_name": os.path.basename(file_path), "row_count": 0, "column_count": 0, "columns": [], "column_mappings": {}}
        except Exception as e:
            logger.error(f"Error processing CSV file {file_path}: {e}", exc_info=True)
            raise # Re-raise the exception

    def _infer_column_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """ Use LLM to infer entity types from column names and data. """
        if df.empty:
             return {}

        # Prepare inputs for the LLM chain
        column_names = list(df.columns)
        # Sample data needs careful handling for large strings or complex types
        try:
             sample_data = df.head(5).to_dict('records')
             # Sanitize sample data for JSON serialization (e.g., handle dates, large strings)
             def sanitize_for_json(obj):
                  if isinstance(obj, (datetime, pd.Timestamp)):
                       return obj.isoformat()
                  # Add other type handlers if needed
                  try:
                      # Attempt to stringify, truncate if long
                      s = str(obj)
                      return s[:200] + '...' if len(s) > 200 else s
                  except Exception:
                      return "<Unserializable Data>"

             sanitized_sample = [[{k: sanitize_for_json(v) for k, v in row.items()} for row in sample_data]]
             sample_data_str = json.dumps(sanitized_sample, indent=2)
        except Exception as json_err:
             logger.warning(f"Could not serialize sample data for LLM inference: {json_err}. Proceeding with column names only.")
             sample_data_str = "[Sample data could not be displayed]"


        # Limit prompt size if necessary
        max_prompt_chars = 8000 # Adjust based on LLM limits
        input_text = f"Columns: {column_names}, Sample: {sample_data_str}"
        if len(input_text) > max_prompt_chars:
             # Truncate sample data primarily
             sample_data_str = sample_data_str[:max_prompt_chars - len(f"Columns: {column_names}, Sample: ")] + "...]"
             logger.warning("Sample data truncated for LLM column type inference due to length limits.")


        try:
            raw_response = self.column_chain.invoke({
                 "column_names": json.dumps(column_names),
                 "sample_data": sample_data_str
            })
            # Assuming raw_response is a dict with the key containing the LLM output string
            # Adjust key based on actual LLMChain output, often 'text'
            llm_output_text = raw_response.get('text', '')

            if not llm_output_text:
                 logger.error("LLM call for column inference returned empty response.")
                 return {}

            # Try parsing with Pydantic parser first
            try:
                 parsed_result = self.column_parser.parse(llm_output_text)
                 # Convert List[ColumnMapping] back to Dict[str, str], filtering out 'Attribute'
                 mappings = {mapping.column_name: mapping.entity_type
                             for mapping in parsed_result.column_mappings
                             if mapping.entity_type != 'Attribute' and mapping.column_name in df.columns} # Ensure column exists
                 return mappings
            except Exception as parse_error:
                 logger.warning(f"Failed to parse initial LLM response for column types: {parse_error}. Attempting to fix...")
                 # Use OutputFixingParser
                 fixed_result = self.column_fixing_parser.parse(llm_output_text)
                 mappings = {mapping.column_name: mapping.entity_type
                              for mapping in fixed_result.column_mappings
                              if mapping.entity_type != 'Attribute' and mapping.column_name in df.columns}
                 return mappings

        except Exception as e:
            logger.error(f"LLM chain failed during column type inference: {e}", exc_info=True)
            return {} # Return empty on failure

    def process_pdf(self, file_path: str) -> Tuple[List[str], Dict[str, Any]]:
        """ Process PDF: Load, chunk text. """
        logger.info(f"Processing PDF file: {file_path}")
        try:
            loader = PyPDFLoader(file_path, extract_images=False) # Disable image extraction
            documents = loader.load()
            logger.info(f"Loaded PDF '{os.path.basename(file_path)}': {len(documents)} pages.")

            # Combine page contents before splitting for better context across pages
            full_text = "\n\n".join([doc.page_content for doc in documents])

            if not full_text.strip():
                 logger.warning(f"PDF file {file_path} seems empty or contains no extractable text.")
                 return [], {"file_path": file_path, "file_name": os.path.basename(file_path), "document_count": len(documents), "chunk_count": 0, "total_chars": 0}


            # Split the combined text
            split_docs = self.text_splitter.create_documents([full_text]) # create_documents handles metadata

            # Store metadata and extract text
            text_chunks = []
            self.document_metadata.clear() # Clear previous doc metadata
            for i, doc in enumerate(split_docs):
                 text_chunks.append(doc.page_content)
                 # Store metadata (like start_index) - page number might be lost here
                 # If page number is crucial, split per page first, then chunk pages
                 self.document_metadata[i] = doc.metadata
                 # Add original filename to metadata
                 self.document_metadata[i]['source_file'] = os.path.basename(file_path)


            logger.info(f"Split PDF into {len(split_docs)} chunks.")

            metadata = {
                "file_path": file_path,
                "file_name": os.path.basename(file_path),
                "document_count": len(documents), # Original page/doc count
                "chunk_count": len(split_docs),
                "total_chars": len(full_text)
            }

            # Chunk embedding generation removed for simplicity, focus on KG structure.

            return text_chunks, metadata

        except FileNotFoundError:
             logger.error(f"PDF file not found: {file_path}")
             raise
        except Exception as e:
            # Catch specific PDF parsing errors if possible (e.g., from PyPDF2)
            logger.error(f"Error processing PDF file {file_path}: {e}", exc_info=True)
            raise


    def extract_entities_from_text(self, text: str) -> List[Dict[str, Any]]:
        """ Extract entities using LLMChain and Pydantic parsing. """
        if not text or not text.strip():
             return []

        # Limit input text size
        max_len = 12000 # Adjust based on model context window and typical chunk size
        truncated_text = text[:max_len]
        if len(text) > max_len:
            logger.warning("Text truncated for entity extraction due to length.")

        try:
            raw_response = self.entity_chain.invoke({"text": truncated_text})
            llm_output_text = raw_response.get('text', '')

            if not llm_output_text:
                 logger.error("LLM call for entity extraction returned empty response.")
                 return []

            # Try parsing with Pydantic parser
            try:
                 parsed_result = self.entity_parser.parse(llm_output_text)
            except Exception as parse_error:
                 logger.warning(f"Failed to parse initial LLM response for entities: {parse_error}. Attempting to fix...")
                 parsed_result = self.entity_fixing_parser.parse(llm_output_text)

            # Convert Pydantic models to simple dicts for KG builder compatibility
            entities_list = [entity.dict() for entity in parsed_result.entities]
            logger.debug(f"Extracted {len(entities_list)} entities.")
            return entities_list

        except Exception as e:
            logger.error(f"LLM chain failed during entity extraction: {e}", exc_info=True)
            return []


    def extract_relations_from_text(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """ Extract relationships using LLMChain and Pydantic parsing. """
        if len(entities) < 2 or not text or not text.strip():
            return []

        # Format entity list for the prompt
        entities_list_str = "\n".join([f"- {e.get('name', 'N/A')} ({e.get('type', 'N/A')})" for e in entities])

        # Limit input text size
        max_len = 12000
        truncated_text = text[:max_len]
        if len(text) > max_len:
             logger.warning("Text truncated for relation extraction due to length.")


        try:
            raw_response = self.relation_chain.invoke({
                "text": truncated_text,
                "entities_list_str": entities_list_str
            })
            llm_output_text = raw_response.get('text', '')

            if not llm_output_text:
                 logger.error("LLM call for relation extraction returned empty response.")
                 return []

            # Try parsing
            try:
                 parsed_result = self.relation_parser.parse(llm_output_text)
            except Exception as parse_error:
                 logger.warning(f"Failed to parse initial LLM response for relations: {parse_error}. Attempting to fix...")
                 parsed_result = self.relation_fixing_parser.parse(llm_output_text)

            # Convert to dicts
            relations_list = [relation.dict() for relation in parsed_result.relations]
            logger.debug(f"Extracted {len(relations_list)} relationships.")
            return relations_list

        except Exception as e:
            logger.error(f"LLM chain failed during relation extraction: {e}", exc_info=True)
            return []