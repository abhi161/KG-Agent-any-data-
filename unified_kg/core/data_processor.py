# --- core/data_processor.py ---
import os
import re
import logging
import json
import pandas as pd
from datetime import datetime 
from typing import List, Dict, Any, Tuple, Optional


from langchain_community.document_loaders import PyPDFLoader, CSVLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from pydantic import BaseModel, Field, field_validator
from langchain.chains import LLMChain
from langchain.llms.base import BaseLLM 
from langchain.embeddings.base import Embeddings 
from .schema_manager import SchemaManager


logger = logging.getLogger(__name__)



class Entity(BaseModel):
    name: str = Field(description="The specific name of the entity mentioned in the text")
    type: str = Field(description="A concise entity type (e.g., Person, Organization, Location, Drug, Disease, Chemical Compound, Medical Device)")
    attributes: Dict[str, Any] = Field(default_factory=dict, description="Key attributes or properties of the entity mentioned (e.g., dosage for a drug, symptom for a disease)")
    context: Optional[str] = Field(default=None, description="The sentence or short phrase where the entity was mentioned, providing context")

    @field_validator('name', 'type')
    def must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Name and type must not be empty")
        return v.strip()

class Relation(BaseModel):
    source: str = Field(description="The name of the source entity in the relationship")
    relation: str = Field(description="The type of relationship (verb phrase, e.g., TREATS, CAUSES, INTERACTS_WITH, MANUFACTURED_BY)")
    target: str = Field(description="The name of the target entity in the relationship")
    context: Optional[str] = Field(default=None, description="The sentence or short phrase where the relationship was mentioned")

    @field_validator('source', 'relation', 'target')
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
    column_mappings: List[ColumnMapping] = Field(description="Mapping of relevant column names to their primary entity types or 'Attribute'.")




class DataProcessor:
    """ Processor for structured (CSV) and unstructured (PDF/Text) data. """

    def __init__(self, llm: BaseLLM,schema_manager: SchemaManager,  embeddings: Optional[Embeddings] = None, chunk_size=1000, chunk_overlap=150):
        self.llm = llm
        self.embeddings = embeddings
        self.schema_manager = schema_manager
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            add_start_index=True,
        )

        # Entity Extraction Chain----------------------------------------------------------------------
        entity_parser = PydanticOutputParser(pydantic_object=EntitiesOutput)
        entity_prompt = PromptTemplate(
            template="""Extract key entities from the text below. For each entity, provide its name and any key attributes mentioned.
            **Crucially, map the entity to the MOST specific existing entity type from the schema provided below.**
            If no existing type accurately represents the entity, you may propose a concise, new type.
            Also provide the sentence where the entity appears as context.

            Available Schema Entity Types:
            {schema_definition}

            Text:
            {text}

            {format_instructions}""",
            input_variables=["text", "schema_definition"], 
            partial_variables={"format_instructions": entity_parser.get_format_instructions()}
        )
        self.entity_chain = LLMChain(llm=llm, prompt=entity_prompt)
        self.entity_parser = entity_parser
        self.entity_fixing_parser = OutputFixingParser.from_llm(parser=entity_parser, llm=llm)



        # Relation Extraction Chain----------------------------------------------------------------------------------
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




        # Column Type Inference Chain-------------------------------------------------------------------------------
        column_parser = PydanticOutputParser(pydantic_object=ColumnMappingsOutput)
        column_prompt = PromptTemplate(
            template="""
            Task: Infer Entity Type Mappings for CSV Columns

            Analyze the provided CSV column names and sample data. Your goal is to map **relevant columns** to their **primary entity type** based on the provided Knowledge Graph Schema Definition.

            **Knowledge Graph Schema Definition:**
            {schema_definition}

            **Instructions:**

            1.  **Identify Identifier Columns:** Pay **critical attention** to columns whose names **exactly match** an `identifier_property` defined in the schema (e.g., `patient_id`, `doctor_id`, `med_id`, `prescription_id`, or `name` if the type is Hospital, Manufacturer, Disease, etc.). These columns are **strong indicators** of the entity type. Map these columns to their corresponding entity type from the schema.
            2.  **Identify Other Entity Columns:** Map other columns that clearly represent instances of an entity type defined in the schema (even if not the identifier column) to that entity type. Use the exact type name from the schema.
            3.  **Use 'Attribute' Sparingly:** If a column represents a simple property, measurement, date, description, flag, or characteristic that **does not** correspond to a distinct entity type defined in the schema, classify it as 'Attribute'. **Do NOT** classify identifier columns (like `patient_id`) as 'Attribute'.
            4.  **Avoid New Types:** Do NOT propose new entity types unless absolutely necessary and the concept is clearly distinct from all existing schema types. Prefer mapping to existing types or 'Attribute'.
            5.  **Focus on Primary Meaning:** Consider the primary meaning of the column based on its name and data.
            6.  **Output Format:** Provide the mapping only for columns identified as representing an entity type (not 'Attribute'). Use the specified JSON format.

            **CSV Data:**

            Column names:
            {column_names}

            Sample data (first 5 rows):
            {sample_data}

            **Required Output:**

            {format_instructions}
            """,
            input_variables=["column_names", "sample_data", "schema_definition"], # Added schema_definition
            partial_variables={"format_instructions": column_parser.get_format_instructions()}
        )
        self.column_chain = LLMChain(llm=llm, prompt=column_prompt)
        self.column_parser = column_parser
        self.column_fixing_parser = OutputFixingParser.from_llm(parser=column_parser, llm=llm)

        self.document_metadata: Dict[int, Dict] = {}
        logger.info("DataProcessor initialized.")



    # Helper function to sanitize values for JSON 
    def _sanitize_value_for_json(self, value: Any) -> Any:
        """Sanitizes individual values for safe JSON serialization."""

        if isinstance(value, (datetime, pd.Timestamp)):
            try:
                return value.isoformat()
            except Exception:
                return str(value) 
        elif pd.isna(value): 
             return None 
        elif isinstance(value, (int, float, bool, str)):
            if isinstance(value, str) and len(value) > 200:
                return value[:197] + '...'
            return value
        else:
            try:
                s = str(value)
                if len(s) > 200:
                    return s[:197] + '...'
                return s
            except Exception:
                return "<Unserializable Data>"


    # --- CSV Processing ---
    def process_csv(self, file_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
      
        logger.info(f"Processing CSV file: {file_path}")
        try:
            df = pd.read_csv(file_path, low_memory=False)
            df.columns = df.columns.str.strip()
            df = df.dropna(how='all')

            logger.info(f"Loaded CSV '{os.path.basename(file_path)}': {len(df)} rows, {len(df.columns)} columns.")
            metadata = {
                "file_path": file_path, "file_name": os.path.basename(file_path),
                "row_count": len(df), "column_count": len(df.columns),
                "columns": list(df.columns)
            }
            column_mappings = self._infer_column_types(df)
            metadata["column_mappings"] = column_mappings
            logger.info(f"Inferred column mappings: {column_mappings}")
            
            df = df.fillna('')
            return df, metadata
        
        except FileNotFoundError: logger.error(f"CSV file not found: {file_path}"); raise
        except pd.errors.EmptyDataError:
             logger.warning(f"CSV file is empty: {file_path}")
             return pd.DataFrame(columns=[]), {"file_path": file_path, "file_name": os.path.basename(file_path), "row_count": 0, "column_count": 0, "columns": [], "column_mappings": {}}
        except Exception as e: logger.error(f"Error processing CSV file {file_path}: {e}", exc_info=True); raise



    def _infer_column_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """ Use LLM to infer entity types from column names and data. """
        if df.empty:
             return {}

        column_names = list(df.columns)
        sample_data_str = "[Sample data unavailable]" # Default

        try:
            sample_data_raw = df.head(5).to_dict('records')
            
            sanitized_sample_data = [
                {k: self._sanitize_value_for_json(v) for k, v in row.items()}
                for row in sample_data_raw
            ]
           
            sample_data_str = json.dumps(sanitized_sample_data, indent=2)
            logger.debug(f"Sample data for LLM inference:\n{sample_data_str}") 

        except Exception as json_err:
             logger.warning(f"Could not serialize sample data for LLM inference: {json_err}. Proceeding with column names only.", exc_info=True)

        current_schema_prompt_str = self.schema_manager.get_current_schema_definition(format_for_prompt=True)

        max_prompt_chars = 8000
        input_text = f"Columns: {column_names}, Sample: {sample_data_str}"
        if len(input_text) > max_prompt_chars:
             estimated_col_len = len(json.dumps(column_names)) + 30
             allowed_sample_len = max_prompt_chars - estimated_col_len
             sample_data_str = sample_data_str[:allowed_sample_len] + "...]" 
             logger.warning("Sample data string truncated for LLM column type inference due to length limits.")

        try:
            raw_response = self.column_chain.invoke({
                 "column_names": json.dumps(column_names),
                 "sample_data": sample_data_str,
                 "schema_definition": current_schema_prompt_str 
            })
            llm_output_text = raw_response.get('text', '')
            if not llm_output_text:
                 logger.error("LLM call for column inference returned empty response.")
                 return {}

            try:
                 parsed_result = self.column_parser.parse(llm_output_text)
                 mappings = {mapping.column_name: mapping.entity_type
                             for mapping in parsed_result.column_mappings
                             if mapping.entity_type != 'Attribute' and mapping.column_name in df.columns}
                 return mappings
            except Exception as parse_error:
                 logger.warning(f"Failed to parse initial LLM response for column types: {parse_error}. Attempting to fix...")
                 fixed_result = self.column_fixing_parser.parse(llm_output_text)
                 mappings = {mapping.column_name: mapping.entity_type
                              for mapping in fixed_result.column_mappings
                              if mapping.entity_type != 'Attribute' and mapping.column_name in df.columns}
                 return mappings

        except Exception as e:
            logger.error(f"LLM chain failed during column type inference: {e}", exc_info=True)
            return {}


    # --- PDF Processing (process_pdf unchanged) ---
    def process_pdf(self, file_path: str) -> Tuple[List[str], Dict[str, Any]]:
        # (Implementation remains the same as previous correct version)
        logger.info(f"Processing PDF file: {file_path}")
        try:
            loader = PyPDFLoader(file_path, extract_images=False)
            documents = loader.load()
            logger.info(f"Loaded PDF '{os.path.basename(file_path)}': {len(documents)} pages.")
            full_text = "\n\n".join([doc.page_content for doc in documents])
            if not full_text.strip():
                 logger.warning(f"PDF file {file_path} seems empty or contains no extractable text.")
                 return [], {"file_path": file_path, "file_name": os.path.basename(file_path), "document_count": len(documents), "chunk_count": 0, "total_chars": 0}
            split_docs = self.text_splitter.create_documents([full_text])
            text_chunks = []
            self.document_metadata.clear()
            for i, doc in enumerate(split_docs):
                 text_chunks.append(doc.page_content)
                 self.document_metadata[i] = doc.metadata
                 self.document_metadata[i]['source_file'] = os.path.basename(file_path)
            logger.info(f"Split PDF into {len(split_docs)} chunks.")
            metadata = {
                "file_path": file_path, "file_name": os.path.basename(file_path),
                "document_count": len(documents), "chunk_count": len(split_docs),
                "total_chars": len(full_text)
            }
            return text_chunks, metadata
        except FileNotFoundError: logger.error(f"PDF file not found: {file_path}"); raise
        except Exception as e: logger.error(f"Error processing PDF file {file_path}: {e}", exc_info=True); raise

    # --- Entity Extraction (extract_entities_from_text unchanged) ---
    def extract_entities_from_text(self, text: str) -> List[Dict[str, Any]]:
        # (Implementation remains the same as previous correct version)
        if not text or not text.strip(): return []
        max_len = 12000
        truncated_text = text[:max_len]
        if len(text) > max_len: logger.warning("Text truncated for entity extraction.")

        current_schema_prompt_str = self.schema_manager.get_current_schema_definition(format_for_prompt=True)

        try:
            raw_response = self.entity_chain.invoke({"text": truncated_text,"schema_definition": current_schema_prompt_str})
            llm_output_text = raw_response.get('text', '')
            if not llm_output_text: logger.error("LLM entity extraction empty."); return []
            try: parsed_result = self.entity_parser.parse(llm_output_text)
            except Exception as parse_error:
                 logger.warning(f"Failed parse entities: {parse_error}. Fixing...")
                 parsed_result = self.entity_fixing_parser.parse(llm_output_text)
            entities_list = [entity.dict() for entity in parsed_result.entities]
            logger.debug(f"Extracted {len(entities_list)} entities.")
            return entities_list
        except Exception as e: logger.error(f"LLM chain failed entity extraction: {e}", exc_info=True); return []

    # --- Relation Extraction (extract_relations_from_text unchanged) ---
    def extract_relations_from_text(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # (Implementation remains the same as previous correct version)
        if len(entities) < 2 or not text or not text.strip(): return []
        entities_list_str = "\n".join([f"- {e.get('name', 'N/A')} ({e.get('type', 'N/A')})" for e in entities])
        max_len = 12000
        truncated_text = text[:max_len]
        if len(text) > max_len: logger.warning("Text truncated relation extraction.")
        try:
            raw_response = self.relation_chain.invoke({"text": truncated_text, "entities_list_str": entities_list_str})
            llm_output_text = raw_response.get('text', '')
            if not llm_output_text: logger.error("LLM relation extraction empty."); return []
            try: parsed_result = self.relation_parser.parse(llm_output_text)
            except Exception as parse_error:
                 logger.warning(f"Failed parse relations: {parse_error}. Fixing...")
                 parsed_result = self.relation_fixing_parser.parse(llm_output_text)
            relations_list = [relation.dict() for relation in parsed_result.relations]
            logger.debug(f"Extracted {len(relations_list)} relationships.")
            return relations_list
        except Exception as e: logger.error(f"LLM chain failed relation extraction: {e}", exc_info=True); return []