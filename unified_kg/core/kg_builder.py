import logging
import pandas as pd
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import os 
import regex as re
# Assuming these are in the same directory or package
from .entity_resolution import EntityResolution
from .data_processor import DataProcessor
from .schema_manager import SchemaManager # Assuming SchemaManager exists

logger = logging.getLogger(__name__)

class LLMEnhancedKnowledgeGraph:
    """
    Main class for building a unified knowledge graph from structured and unstructured data
    with vector embeddings support.
    Refactored relationship creation for structured data.
    """

    def __init__(self, llm, graph_db, embeddings, initial_schema: Optional[Dict[str, Any]] = None,
                 config: Optional[Dict[str, Any]] = None):
        self.llm = llm
        self.graph_db = graph_db
        self.embeddings = embeddings # Can be None if vector_enabled is False
        self.config = config or {}
        self.config['initial_schema'] = initial_schema
        self.vector_enabled = self.config.get("vector_enabled", False) and self.embeddings is not None
        self.vector_similarity_threshold = self.config.get("vector_similarity_threshold", 0.85)

        logger.info(f"KG Builder initialized. Vector support: {'Enabled' if self.vector_enabled else 'Disabled'}")

        # Initialize components
        self.entity_resolution = EntityResolution(
            llm=llm,
            graph_db=graph_db,
            embeddings=self.embeddings if self.vector_enabled else None, # Pass embeddings only if enabled
            config=self.config # Pass config for threshold etc.
        )
        self.data_processor = DataProcessor(
            llm=llm,
            embeddings=self.embeddings if self.vector_enabled else None,
            chunk_size=self.config.get("chunk_size", 1000),
            chunk_overlap=self.config.get("chunk_overlap", 150)
        )
        # Assuming SchemaManager is defined elsewhere and handles schema logic
        self.schema_manager = SchemaManager(llm, graph_db, initial_schema)

        # Initialize schema (assuming SchemaManager handles this)
        logger.info("Initializing schema...")
        self.schema_manager.initialize_schema()
        logger.info("Schema initialized.")
        self.schema_relation_map = self._build_schema_relation_map(initial_schema)


        # Statistics
        self.stats = {
            "start_time": datetime.now().isoformat(),
            "csv_files_processed": 0,
            "pdf_files_processed": 0,
            "entities_created": 0, # Tracked via entity_resolution stats
            "entities_merged": 0, # Tracked via entity_resolution stats
            "relationships_created": 0,
            "csv": {"rows_processed": 0, "relationships_created": 0},
            "pdf": {"chunks_processed": 0, "entities_extracted": 0, "relationships_extracted": 0, "relationships_created": 0},
            "cross_ref": {"name_based_rels_created": 0, "vector_based_rels_created": 0}
        }

    def _build_schema_relation_map(self, schema: Optional[Dict[str, Any]]) -> Dict[Tuple[str, str], str]:
        """Helper to create a map for quick lookup of schema-defined relationships."""
        rel_map = {}
        if schema and "relation_types" in schema:
            for rel in schema["relation_types"]:
                name = rel.get("name")
                # Handle multiple source/target types if defined
                src_types = rel.get("source_types", [])
                tgt_types = rel.get("target_types", [])
                if name and src_types and tgt_types:
                    for src in src_types:
                        for tgt in tgt_types:
                             # Store sanitized labels
                            rel_map[(self.sanitize_label(src), self.sanitize_label(tgt))] = name
        logger.debug(f"Built schema relationship map: {rel_map}")
        return rel_map

    @staticmethod
    def sanitize_label(label):
        """Convert entity type names to valid Neo4j labels (alphanumeric + underscore)."""
        if not isinstance(label, str):
             logger.warning(f"Attempted to sanitize non-string label: {label}. Returning 'UnknownType'.")
             return "UnknownType"
        # Replace spaces and invalid characters with underscore, ensure starts with letter
        sanitized = ''.join(c if c.isalnum() else '_' for c in label)
        if not sanitized or not sanitized[0].isalpha():
            sanitized = "Type_" + sanitized
        # Remove consecutive underscores
        sanitized = '_'.join(filter(None, sanitized.split('_')))
        return sanitized

    def process_csv_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process a CSV file, add entities, and create relationships based on structure.
        """
        logger.info(f"Processing CSV file: {file_path}")
        file_basename = os.path.basename(file_path)
        processing_stats = {
            "file_path": file_path,
            "rows_processed": 0,
            "entities_processed": 0,
            "relationships_created": 0,
            "column_mappings": {}
        }

        try:
            # 1. Process the CSV using DataProcessor
            df, metadata = self.data_processor.process_csv(file_path)
            processing_stats["column_mappings"] = metadata.get("column_mappings", {})
            logger.info(f"CSV loaded: {len(df)} rows, Columns: {metadata.get('columns')}")
            logger.info(f"Inferred column mappings: {processing_stats['column_mappings']}")

            # 2. Process entities row by row
            entities_by_row = self._process_csv_entities(df, processing_stats["column_mappings"], file_basename)
            processing_stats["rows_processed"] = len(entities_by_row)
            processing_stats["entities_processed"] = sum(len(row_entities) for row_entities in entities_by_row)
            logger.info(f"Processed {processing_stats['entities_processed']} potential entities from {processing_stats['rows_processed']} rows.")


            # 3. Create relationships based on CSV structure and schema
            rels_created_count = self._create_csv_structural_relationships(
                entities_by_row,
                processing_stats["column_mappings"],
                file_basename
            )
            processing_stats["relationships_created"] = rels_created_count
            logger.info(f"Created {rels_created_count} structural relationships from CSV.")

            # Update global stats
            self.stats["csv_files_processed"] += 1
            self.stats["csv"]["rows_processed"] += processing_stats["rows_processed"]
            self.stats["csv"]["relationships_created"] += processing_stats["relationships_created"]


        except Exception as e:
            logger.error(f"Failed to process CSV file {file_path}: {e}", exc_info=True)
            processing_stats["error"] = str(e)
            # Optionally re-raise or handle differently
            # raise

        return processing_stats


    def _process_csv_entities(self, df: pd.DataFrame, column_mappings: Dict[str, str],
                          source_file: str) -> List[Dict[str, Any]]:
        """
        Process entities from a CSV DataFrame. Identifies entities based on column_mappings.
        Ensures identifier properties are included in the properties dict passed for resolution.
        """
        entities_by_row = []
        batch_size = self.config.get("batch_size", 100)
        total_entities_processed = 0

        valid_columns = {col: type for col, type in column_mappings.items() if col in df.columns}
        if len(valid_columns) != len(column_mappings):
            logger.warning(f"Some columns in mappings not found in DataFrame: {set(column_mappings.keys()) - set(df.columns)}")
        if not valid_columns:
            logger.warning(f"No valid columns mapped to entity types for {source_file}. Skipping entity processing.")
            return []

        # Pre-fetch identifier property names for mapped types for efficiency
        identifier_map = self.entity_resolution.identifier_properties_map # Access map from ER instance

        for start_idx in range(0, len(df), batch_size):
            end_idx = min(start_idx + batch_size, len(df))
            batch_df = df.iloc[start_idx:end_idx]
            logger.debug(f"Processing batch: rows {start_idx} to {end_idx-1}")

            batch_row_entities = []
            for idx, row in batch_df.iterrows():
                row_entities = {}
                # Base properties includes all raw row data initially
                base_properties = {
                    "source": source_file,
                    "row_index": idx
                }
                for prop_col, prop_val in row.items():
                    # Sanitize property name from column header
                    prop_name = self.sanitize_label(prop_col).lower()
                    # Store raw value first, handle NaN/empty later if needed
                    base_properties[prop_name] = prop_val


                # Process each column identified as an entity type
                for column, entity_type in valid_columns.items():
                    entity_value = row[column]
                    if pd.notna(entity_value) and entity_value != '':
                        entity_name = str(entity_value)
                        sanitized_entity_type = self.sanitize_label(entity_type)

                        # Prepare properties specifically for this entity
                        entity_properties = {
                            "source": source_file,
                            "row_index": idx
                            # DO NOT copy all base_properties blindly here initially
                        }

                        # Add all OTHER columns from the row as properties
                        for prop_col, prop_val in row.items():
                            # Only add if it's not the column defining this entity's name/value
                            if prop_col != column and pd.notna(prop_val) and prop_val != '':
                                prop_name_sanitized = self.sanitize_label(prop_col).lower()
                                # Basic type preservation attempt
                                try:
                                    entity_properties[prop_name_sanitized] = pd.to_numeric(prop_val, errors='ignore')
                                except Exception:
                                    entity_properties[prop_name_sanitized] = str(prop_val)


                        # *** CRUCIAL FIX START ***
                        # Check if this entity type has a defined identifier property
                        identifier_prop_name = identifier_map.get(sanitized_entity_type)

                        if identifier_prop_name:
                            # Find the original column name that CORRESPONDS to this identifier property
                            # This requires reversing the column_mappings or searching
                            # Assume the column used to get entity_name IS the identifier column if not found elsewhere
                            id_column_name = None
                            # A) If the identifier prop name is directly in the row's base props (already sanitized)
                            if identifier_prop_name in base_properties:
                                id_value_from_row = base_properties[identifier_prop_name]
                                if pd.notna(id_value_from_row) and id_value_from_row != '':
                                    entity_properties[identifier_prop_name] = id_value_from_row
                                    logger.debug(f"Added identifier '{identifier_prop_name}' = '{id_value_from_row}' from base_properties for {entity_name}")

                            # B) As a fallback: If the current column being processed IS the identifier column
                            # (This happens if patient_id column is directly mapped to Patient entity type)
                            elif identifier_prop_name == self.sanitize_label(column).lower():
                                id_value_from_row = entity_value # The value we are processing IS the ID
                                if pd.notna(id_value_from_row) and id_value_from_row != '':
                                    entity_properties[identifier_prop_name] = id_value_from_row
                                    logger.debug(f"Added identifier '{identifier_prop_name}' = '{id_value_from_row}' using current column value for {entity_name}")

                            else:
                                # Attempt to find the original column name for the identifier property if needed
                                # This logic might be complex depending on how mappings are done
                                logger.warning(f"Could not automatically determine source column for identifier property '{identifier_prop_name}' for entity type {sanitized_entity_type} defined by column '{column}'. Identifier matching may fail.")

                        # *** CRUCIAL FIX END ***


                        # Process entity (add/update/resolve)
                        entity_details = self._add_or_update_entity(
                            name=entity_name, # Pass the original value as name (e.g., 'P003' or 'Robert Chen')
                            entity_type=sanitized_entity_type,
                            properties=entity_properties, # Pass the specifically constructed properties
                            source=source_file
                        )

                        # Store details needed for relationship creation
                        if entity_details: # Check if entity creation/update was successful
                            row_entities[column] = {
                                "id": entity_details["id"],
                                "name": entity_name,
                                "type": sanitized_entity_type,
                                "merged": entity_details["merged"]
                            }
                            total_entities_processed += 1
                        else:
                            logger.error(f"Failed to process entity from column '{column}', value '{entity_name}' in row {idx}. Skipping.")


                batch_row_entities.append(row_entities)

            entities_by_row.extend(batch_row_entities)
            logger.debug(f"Processed entities for batch, total entities processed so far: {total_entities_processed}")

        return entities_by_row

    def _add_or_update_entity(self, name: str, entity_type: str, properties: Dict[str, Any],
                              source: str) -> Dict[str, Any]:
        """
        Add a new entity or update/merge an existing one. Uses EntityResolution component.
        Updates global stats based on resolution outcome. Returns entity details including ID and merge status.
        """
        # Ensure entity type exists in schema (discover if necessary)
        # This check might be better placed within EntityResolution or SchemaManager
        # For simplicity, keeping basic check here. Assumes SchemaManager handles persistence.
        if not self.schema_manager.has_entity_type(entity_type):
             logger.info(f"Discovering new entity type: {entity_type} from source {source}")
             # Confidence might be dynamic or fixed for CSV discovery
             self.schema_manager.discover_entity_type(entity_type, confidence=0.9, source=source)
             # Optionally process pending changes immediately or batch them
             # self.schema_manager.process_pending_changes()

        # Find matching entity using resolution strategies
        match_result = self.entity_resolution.find_matching_entity(name, entity_type, properties)

        entity_id = None
        merged = False

        if match_result:
            # Merge properties into the existing entity
            entity_id = match_result["id"]
            self.entity_resolution.merge_entity_properties(entity_id, properties, source)
            # Update stats for merged entities
            self.stats["entities_merged"] += 1 # Assuming EntityResolution tracks this, or track here
            merged = True
            logger.debug(f"Merged entity '{name}' ({entity_type}) with existing ID {entity_id}")
        else:
            # Create a new entity
            entity_id = self.entity_resolution.create_new_entity(name, entity_type, properties, source)
            # Update stats for new entities
            self.stats["entities_created"] += 1 # Assuming EntityResolution tracks this, or track here
            merged = False
            logger.debug(f"Created new entity '{name}' ({entity_type}) with ID {entity_id}")

        # Update global stats based on EntityResolution internal counters
        resolution_stats = self.entity_resolution.get_stats()
        self.stats["entities_created"] = resolution_stats.get("new_entities", 0)
        self.stats["entities_merged"] = resolution_stats.get("merged_entities", 0) # Ensure ER tracks merges

        return {"id": entity_id, "merged": merged}


    def _create_csv_structural_relationships(self, entities_by_row: List[Dict[str, Any]],
                                             column_mappings: Dict[str, str], source_file: str) -> int:
        """
        Creates relationships between entities found in the *same CSV row* based on
        predefined structural patterns and the initial schema.
        """
        relationships_created = 0
        entity_type_to_column = {self.sanitize_label(v): k for k, v in column_mappings.items()}

        # Define common structural patterns (expand as needed)
        # Tuple format: (SourceEntityType, TargetEntityType, RelationshipName)
        structural_patterns = [
            ("Brand", "Generic", "HAS_GENERIC_NAME"),
            ("Brand", "Manufacturer", "MANUFACTURED_BY"),
            # Add more patterns based on expected CSV structures
        ]

        for row_entities in entities_by_row:
            if not row_entities:
                continue

            # Get entities present in this row, mapped by their SANITIZED type
            present_entities_by_type = {}
            for col, entity_data in row_entities.items():
                 # Use the sanitized type from entity_data
                 sanitized_type = entity_data["type"]
                 present_entities_by_type[sanitized_type] = entity_data # Store full data


            # 1. Apply schema-defined relationships first (more specific)
            schema_rels_found = set() # Avoid duplicate checks
            for (src_type, tgt_type), rel_name in self.schema_relation_map.items():
                 if src_type in present_entities_by_type and tgt_type in present_entities_by_type:
                     if (src_type, tgt_type) not in schema_rels_found:
                         source_entity = present_entities_by_type[src_type]
                         target_entity = present_entities_by_type[tgt_type]

                         logger.debug(f"Creating schema-based relationship: ({source_entity['name']}:{src_type})-[{rel_name}]->({target_entity['name']}:{tgt_type})")
                         self._create_relationship(source_entity["id"], target_entity["id"], rel_name, source_file)
                         relationships_created += 1
                         schema_rels_found.add((src_type, tgt_type))


            # 2. Apply general structural patterns (less specific)
            for src_pattern_type, tgt_pattern_type, rel_pattern_name in structural_patterns:
                 src_type_sanitized = self.sanitize_label(src_pattern_type)
                 tgt_type_sanitized = self.sanitize_label(tgt_pattern_type)

                 # Check if this pattern was already handled by the schema
                 if (src_type_sanitized, tgt_type_sanitized) in schema_rels_found:
                      continue

                 if src_type_sanitized in present_entities_by_type and tgt_type_sanitized in present_entities_by_type:
                     source_entity = present_entities_by_type[src_type_sanitized]
                     target_entity = present_entities_by_type[tgt_type_sanitized]

                     # Discover the relationship type if not already known by SchemaManager
                     if not self.schema_manager.has_relation_type(rel_pattern_name):
                          logger.info(f"Discovering new relationship type: {rel_pattern_name} from CSV structure")
                          self.schema_manager.discover_relation_type(rel_pattern_name, 0.9, f"csv_structure:{source_file}")
                          # Optionally process changes immediately: self.schema_manager.process_pending_changes()


                     logger.debug(f"Creating structural relationship: ({source_entity['name']}:{src_type_sanitized})-[{rel_pattern_name}]->({target_entity['name']}:{tgt_type_sanitized})")
                     self._create_relationship(source_entity["id"], target_entity["id"], rel_pattern_name, source_file)
                     relationships_created += 1


        return relationships_created

    def _create_relationship(self, source_id: str, target_id: str, rel_type: str, source: str, properties: Optional[Dict] = None):
        """ Helper function to create a relationship in Neo4j """
        # Basic validation
        if not source_id or not target_id or not rel_type:
             logger.warning(f"Skipping relationship creation due to missing ID or type: {source_id=}, {target_id=}, {rel_type=}")
             return

        # Sanitize relationship type (uppercase, underscores)
        sanitized_rel_type = re.sub(r'\W|^(?=\d)', '_', rel_type.upper()).replace('__', '_').strip('_')
        if not sanitized_rel_type:
             sanitized_rel_type = "RELATED_TO"
             logger.warning(f"Relationship type '{rel_type}' sanitized to default 'RELATED_TO'")


        query = f"""
        MATCH (a), (b)
        WHERE elementId(a) = $source_id AND elementId(b) = $target_id
        MERGE (a)-[r:`{sanitized_rel_type}`]->(b)
        ON CREATE SET r.source = $source, r.created_at = timestamp()
        ON MATCH SET r.source = CASE WHEN $source IN coalesce(r.sources, []) THEN r.source ELSE coalesce(r.source, '') + '|' + $source END // Append source on match maybe? Or just update timestamp?
        // Add other properties
        """
        params = {
            "source_id": source_id,
            "target_id": target_id,
            "source": source
        }

        # Add optional properties
        if properties:
             prop_set_clauses = []
             for key, value in properties.items():
                  # Sanitize property keys just in case
                  prop_key_sanitized = re.sub(r'\W|^(?=\d)', '_', key).lstrip('_')
                  if prop_key_sanitized:
                       params[f"prop_{prop_key_sanitized}"] = value
                       prop_set_clauses.append(f"r.{prop_key_sanitized} = $prop_{prop_key_sanitized}")

             if prop_set_clauses:
                  # Modify query to include SET clauses for properties
                  set_clause_str = ", ".join(prop_set_clauses)
                  # Be careful inserting SET clauses correctly
                  if "ON CREATE SET" in query:
                      query = query.replace("ON CREATE SET", f"ON CREATE SET {set_clause_str},", 1)
                  else:
                      query += f" SET {set_clause_str}" # If no ON CREATE/MATCH
                  if "ON MATCH SET" in query:
                      query = query.replace("ON MATCH SET", f"ON MATCH SET {set_clause_str},", 1)


        try:
            # logger.debug(f"Executing relationship query: {query} with params: {params}") # Be careful logging params with sensitive data
            self.graph_db.query(query, params)
            # self.stats["relationships_created"] += 1 # Increment done in calling functions
        except Exception as e:
            logger.error(f"Error creating relationship '{sanitized_rel_type}' between {source_id} and {target_id}: {e}", exc_info=True)
            logger.error(f"Failed Query: {query}") # Log the failing query
            logger.error(f"Failed Params: {params}") # Log params related to failure


    def process_pdf_file(self, file_path: str) -> Dict[str, Any]:
        """ Process a PDF file, extract entities/relations, add to graph """
        logger.info(f"Processing PDF file: {file_path}")
        file_basename = os.path.basename(file_path)
        pdf_stats = {
            "file_path": file_path,
            "chunks_processed": 0,
            "entities_extracted": 0,
            "entities_created": 0, # In graph
            "relationships_extracted": 0,
            "relationships_created": 0 # In graph
        }

        try:
            chunks, metadata = self.data_processor.process_pdf(file_path)
            pdf_stats["chunks_processed"] = len(chunks)
            logger.info(f"PDF processed into {len(chunks)} chunks.")

            all_chunk_entities = [] # Store entities per chunk for cross-referencing

            for chunk_idx, chunk_text in enumerate(chunks):
                logger.debug(f"Processing chunk {chunk_idx + 1}/{len(chunks)}")

                # 1. Extract entities from chunk text
                # The DataProcessor's method now returns dicts with 'name', 'type', 'attributes', 'context'
                extracted_entities = self.data_processor.extract_entities_from_text(chunk_text)
                pdf_stats["entities_extracted"] += len(extracted_entities)
                logger.debug(f"Extracted {len(extracted_entities)} entities from chunk {chunk_idx}.")

                # 2. Process extracted entities (add/update in graph)
                chunk_graph_entities = {} # Map entity name in chunk to graph ID/details
                for entity_data in extracted_entities:
                    # Ensure type is sanitized
                    sanitized_type = self.sanitize_label(entity_data["type"])
                    # Combine attributes and context into properties
                    properties = entity_data.get("attributes", {})
                    properties["source"] = file_basename
                    properties["chunk_index"] = chunk_idx
                    if "context" in entity_data:
                        properties["context"] = entity_data["context"][:500] # Limit context length
                    # Add other metadata if available from DataProcessor
                    chunk_meta = self.data_processor.document_metadata.get(chunk_idx, {})
                    properties.update({k:v for k,v in chunk_meta.items() if k != 'source'}) # Avoid overwriting file source

                    entity_details = self._add_or_update_entity(
                        name=entity_data["name"],
                        entity_type=sanitized_type,
                        properties=properties,
                        source=file_basename
                    )
                    chunk_graph_entities[entity_data["name"]] = {
                        "id": entity_details["id"],
                        "name": entity_data["name"],
                        "type": sanitized_type
                    }
                    if not entity_details["merged"]:
                         pdf_stats["entities_created"] += 1


                all_chunk_entities.append(chunk_graph_entities) # Add processed entities for this chunk

                # 3. Extract relationships from chunk text (if enough entities)
                if len(extracted_entities) > 1:
                    # Pass the original extracted entity structures (name, type) to the relation extractor
                    extracted_relations = self.data_processor.extract_relations_from_text(chunk_text, extracted_entities)
                    pdf_stats["relationships_extracted"] += len(extracted_relations)
                    logger.debug(f"Extracted {len(extracted_relations)} relationships from chunk {chunk_idx}.")


                    # 4. Process extracted relationships (add to graph)
                    for relation_data in extracted_relations:
                        source_entity = chunk_graph_entities.get(relation_data["source"])
                        target_entity = chunk_graph_entities.get(relation_data["target"])
                        rel_type = relation_data["relation"] # Assume DataProcessor provides clean type

                        if source_entity and target_entity and rel_type:
                             # Check/discover relationship type in schema
                             sanitized_rel_type = self.sanitize_label(rel_type.upper())
                             if not self.schema_manager.has_relation_type(sanitized_rel_type):
                                  logger.info(f"Discovering new relationship type: {sanitized_rel_type} from PDF {file_basename}")
                                  self.schema_manager.discover_relation_type(sanitized_rel_type, 0.7, f"pdf:{file_basename}:chunk{chunk_idx}")
                                  # Optionally process: self.schema_manager.process_pending_changes()

                             rel_props = {"source": file_basename, "chunk_index": chunk_idx}
                             if "context" in relation_data:
                                  rel_props["context"] = relation_data["context"][:500] # Limit context

                             self._create_relationship(
                                 source_entity["id"],
                                 target_entity["id"],
                                 sanitized_rel_type,
                                 f"{file_basename}_chunk_{chunk_idx}", # More specific source
                                 rel_props
                             )
                             pdf_stats["relationships_created"] += 1
                        else:
                             logger.warning(f"Skipping relationship creation in chunk {chunk_idx} due to missing entity mapping or type: {relation_data}")


            # 5. Optional: Cross-reference entities WITHIN the PDF (e.g., co-occurrence)
            # This can be complex; simple co-occurrence example:
            logger.info("Performing intra-PDF entity cross-referencing...")
            intra_pdf_rels = self._cross_reference_pdf_entities(all_chunk_entities, file_basename)
            pdf_stats["relationships_created"] += intra_pdf_rels # Add count of co-occurrence rels
            logger.info(f"Created {intra_pdf_rels} intra-PDF relationships (e.g., co-occurrence).")


            # Update global stats
            self.stats["pdf_files_processed"] += 1
            self.stats["pdf"]["chunks_processed"] += pdf_stats["chunks_processed"]
            self.stats["pdf"]["entities_extracted"] += pdf_stats["entities_extracted"]
            # relationships_created tracks graph additions
            self.stats["pdf"]["relationships_created"] += pdf_stats["relationships_created"]


        except Exception as e:
            logger.error(f"Failed to process PDF file {file_path}: {e}", exc_info=True)
            pdf_stats["error"] = str(e)

        return pdf_stats

    def _cross_reference_pdf_entities(self, entities_by_chunk: List[Dict[str, Dict[str, Any]]],
                                      source_file: str) -> int:
        """
        Simple cross-referencing within a PDF: creates OCCURS_WITH relationships
        between entities found in the same chunk.
        Returns the number of relationships created.
        """
        rels_created = 0
        # Discover OCCURS_WITH type if not present
        rel_type = "OCCURS_WITH"
        if not self.schema_manager.has_relation_type(rel_type):
            self.schema_manager.discover_relation_type(rel_type, 0.9, f"intra_pdf_inference:{source_file}")
            # Optionally process: self.schema_manager.process_pending_changes()

        for chunk_idx, chunk_entities in enumerate(entities_by_chunk):
            entity_list = list(chunk_entities.values())
            if len(entity_list) < 2:
                continue

            # Create relationship between all pairs in the chunk
            for i in range(len(entity_list)):
                for j in range(i + 1, len(entity_list)):
                    id1 = entity_list[i]["id"]
                    id2 = entity_list[j]["id"]
                    props = {"source": source_file, "chunk_index": chunk_idx, "type": "co-occurrence"}
                    # Use helper to create/merge relationship
                    # MERGE handles checking existence
                    self._create_relationship(id1, id2, rel_type, source_file, props)
                    rels_created += 1 # Count each potential merge/create attempt
                    # Note: If MERGE finds existing, it doesn't create new, but we count the intent.
                    # For more accurate "new" count, _create_relationship would need to return status.

        return rels_created


    # --- Cross-Referencing Between Sources ---

    def cross_reference_data_sources(self) -> Dict[str, Any]:
        """
        Cross-reference entities between different data sources (CSV vs PDF).
        Uses name matching and vector similarity (if enabled).
        """
        logger.info("Cross-referencing data sources (CSV vs PDF)...")
        cross_ref_stats = {
            "name_based_evaluations": 0,
            "name_based_rels_created": 0,
            "vector_based_evaluations": 0,
            "vector_based_rels_created": 0,
        }

        try:
            # 1. Name-based cross-referencing (find entities in both types of sources)
             logger.info("Performing name-based cross-referencing...")
             self._cross_reference_by_name(cross_ref_stats)

            # 2. Vector-based cross-referencing (if enabled)
             if self.vector_enabled:
                 logger.info("Performing vector-based cross-referencing...")
                 self._cross_reference_by_vector(cross_ref_stats)
             else:
                 logger.info("Skipping vector-based cross-referencing (disabled or embeddings unavailable).")

        except Exception as e:
             logger.error(f"Error during cross-referencing: {e}", exc_info=True)
             cross_ref_stats["error"] = str(e)

        # Update global stats
        self.stats["cross_ref"]["name_based_rels_created"] = cross_ref_stats["name_based_rels_created"]
        self.stats["cross_ref"]["vector_based_rels_created"] = cross_ref_stats["vector_based_rels_created"]


        return cross_ref_stats

    def _cross_reference_by_name(self, stats: Dict[str, int]) -> None:
        """ Find entities mentioned in both CSV and PDF sources and evaluate links between their neighbors. """
        try:
            # Find entities ('bridge' nodes) that have source properties indicating both CSV and PDF origins
            # This relies on EntityResolution correctly merging nodes and updating sources array.
            query = """
            MATCH (n)
            WHERE size(n.sources) > 1
              AND any(s IN n.sources WHERE s ENDS WITH '.csv')
              AND any(s IN n.sources WHERE s ENDS WITH '.pdf')
            RETURN elementId(n) AS bridge_id, n.name AS bridge_name, labels(n)[0] AS bridge_type
            LIMIT 500 // Limit to avoid excessive processing
            """
            bridge_entities = self.graph_db.query(query)
            logger.info(f"Found {len(bridge_entities)} potential bridge entities for name-based cross-referencing.")

            for bridge in bridge_entities:
                bridge_id = bridge["bridge_id"]
                logger.debug(f"Evaluating bridge entity: {bridge['bridge_name']} ({bridge['bridge_type']})")

                # Find neighbors connected via CSV-originated relationships/properties
                # This query is complex - might need refinement based on how sources are stored
                csv_neighbors_query = """
                MATCH (bridge)-[r]-(neighbor)
                WHERE elementId(bridge) = $bridge_id
                  AND (
                       (r.source IS NOT NULL AND r.source ENDS WITH '.csv') OR
                       (neighbor.source IS NOT NULL AND neighbor.source ENDS WITH '.csv') OR // Check node source if rel source missing
                       any(s IN r.sources WHERE s ENDS WITH '.csv') OR // Check sources list if used
                       any(s IN neighbor.sources WHERE s ENDS WITH '.csv')
                      )
                  AND elementId(neighbor) <> $bridge_id // Ensure neighbor is not the bridge itself
                RETURN DISTINCT elementId(neighbor) AS id, neighbor.name AS name, labels(neighbor)[0] AS type
                LIMIT 50 // Limit neighbors per bridge
                """
                csv_neighbors = self.graph_db.query(csv_neighbors_query, {"bridge_id": bridge_id})

                # Find neighbors connected via PDF-originated relationships/properties
                pdf_neighbors_query = """
                MATCH (bridge)-[r]-(neighbor)
                WHERE elementId(bridge) = $bridge_id
                 AND (
                       (r.source IS NOT NULL AND r.source ENDS WITH '.pdf') OR
                       (neighbor.source IS NOT NULL AND neighbor.source ENDS WITH '.pdf') OR
                       any(s IN r.sources WHERE s ENDS WITH '.pdf') OR
                       any(s IN neighbor.sources WHERE s ENDS WITH '.pdf')
                     )
                 AND elementId(neighbor) <> $bridge_id
                RETURN DISTINCT elementId(neighbor) AS id, neighbor.name AS name, labels(neighbor)[0] AS type
                LIMIT 50
                """
                pdf_neighbors = self.graph_db.query(pdf_neighbors_query, {"bridge_id": bridge_id})

                if not csv_neighbors or not pdf_neighbors:
                    continue # Need neighbors from both sides to cross-reference

                # Evaluate potential connections between CSV neighbors and PDF neighbors
                for csv_entity in csv_neighbors:
                    for pdf_entity in pdf_neighbors:
                        if csv_entity["id"] == pdf_entity["id"]:
                            continue # Don't link entity to itself

                        stats["name_based_evaluations"] += 1

                        # Check if already directly connected
                        existing_rel_query = """
                        MATCH (a)-[r]-(b)
                        WHERE elementId(a) = $id1 AND elementId(b) = $id2
                        RETURN count(r) > 0 AS connected
                        LIMIT 1
                        """
                        already_connected = self.graph_db.query(existing_rel_query, {"id1": csv_entity["id"], "id2": pdf_entity["id"]})

                        if not already_connected or not already_connected[0]["connected"]:
                            # Use LLM to evaluate if a direct connection makes sense
                            relation_type = self._evaluate_cross_source_connection(
                                csv_entity, pdf_entity, bridge
                            )

                            if relation_type:
                                # Create the relationship
                                rel_props = {
                                    "source": "cross_reference_name_based",
                                    "confidence": 0.7, # Indicate inferred nature
                                    "bridge_entity_name": bridge["bridge_name"],
                                    "bridge_entity_type": bridge["bridge_type"]
                                }
                                self._create_relationship(
                                    csv_entity["id"],
                                    pdf_entity["id"],
                                    relation_type,
                                    "cross_reference_name_based",
                                    rel_props
                                )
                                stats["name_based_rels_created"] += 1
                                self.stats["relationships_created"] += 1 # Update global count too
        except Exception as e:
             logger.error(f"Error during name-based cross-referencing query or processing: {e}", exc_info=True)
             # Decide whether to continue or stop cross-referencing

    def _cross_reference_by_vector(self, stats: Dict[str, int]) -> None:
        """ Use vector similarity to find potentially related entities across CSV/PDF sources. """
        if not self.vector_enabled:
            logger.warning("Vector cross-referencing called but vector support is disabled.")
            return

        try:
            # Get a sample of entities originating from CSVs with embeddings
            csv_entities_query = """
            MATCH (n)
            WHERE n.embedding IS NOT NULL
              AND ( (n.source IS NOT NULL AND n.source ENDS WITH '.csv') OR
                    any(s IN n.sources WHERE s ENDS WITH '.csv') )
            RETURN elementId(n) AS id, n.name AS name, labels(n)[0] AS type, n.embedding AS embedding
            LIMIT 500 // Limit the number of source entities to query for
            """
            csv_entities = self.graph_db.query(csv_entities_query)
            logger.info(f"Found {len(csv_entities)} CSV entities with embeddings for vector cross-referencing.")

            for csv_entity in csv_entities:
                 if not csv_entity["embedding"] or not csv_entity["id"]:
                     continue

                 entity_id = csv_entity["id"]
                 entity_embedding = csv_entity["embedding"]

                 # Find similar entities originating from PDFs using the vector index
                 # Assumes 'global_embedding_index' exists and covers all nodes with 'embedding'
                 vector_query = """
                 CALL db.index.vector.queryNodes('global_embedding_index', 10, $embedding) YIELD node, score
                 WHERE elementId(node) <> $entity_id // Don't match self
                   AND score >= $threshold
                   AND ( (node.source IS NOT NULL AND node.source ENDS WITH '.pdf') OR
                         any(s IN node.sources WHERE s ENDS WITH '.pdf') )
                 // Optional: Check if relationship already exists to avoid re-evaluation
                 // AND NOT EXISTS { MATCH (node)-[:CROSS_SOURCE_SIMILAR_TO]-(csv_node) WHERE elementId(csv_node) = $entity_id }
                 RETURN elementId(node) AS id, node.name AS name, labels(node)[0] AS type, score
                 ORDER BY score DESC
                 LIMIT 5 // Limit number of similar matches to evaluate per source entity
                 """

                 similar_pdf_entities = self.graph_db.query(
                     vector_query,
                     {"entity_id": entity_id, "embedding": entity_embedding, "threshold": self.vector_similarity_threshold}
                 )

                 if similar_pdf_entities:
                     logger.debug(f"Found {len(similar_pdf_entities)} potential vector matches from PDF for CSV entity {csv_entity['name']} ({entity_id})")

                 for pdf_entity in similar_pdf_entities:
                     stats["vector_based_evaluations"] += 1

                     # Check if already directly connected (covers various potential rel types)
                     existing_rel_query = """
                        MATCH (a)-[r]-(b)
                        WHERE elementId(a) = $id1 AND elementId(b) = $id2
                        RETURN count(r) > 0 AS connected
                        LIMIT 1
                        """
                     already_connected = self.graph_db.query(existing_rel_query, {"id1": entity_id, "id2": pdf_entity["id"]})

                     if not already_connected or not already_connected[0]["connected"]:
                          # Use LLM to validate the semantic connection based on similarity score
                          should_connect = self._evaluate_vector_similarity_connection(
                              csv_entity, pdf_entity, pdf_entity["score"]
                          )

                          if should_connect:
                              rel_type = "CROSS_SOURCE_SIMILAR_TO" # Specific type for vector links
                              # Discover if needed
                              if not self.schema_manager.has_relation_type(rel_type):
                                   self.schema_manager.discover_relation_type(rel_type, 0.8, "vector_cross_reference")

                              rel_props = {
                                  "source": "cross_reference_vector_based",
                                  "similarity_score": pdf_entity["score"],
                                  "confidence": pdf_entity["score"] * 0.8 # Confidence related to score
                              }
                              self._create_relationship(
                                  entity_id,
                                  pdf_entity["id"],
                                  rel_type,
                                  "cross_reference_vector_based",
                                  rel_props
                              )
                              stats["vector_based_rels_created"] += 1
                              self.stats["relationships_created"] += 1 # Update global count
        except Exception as e:
            # Catch specific Neo4jError if index doesn't exist?
             logger.error(f"Error during vector-based cross-referencing query or processing: {e}", exc_info=True)
             # Decide whether to continue or stop


    def _evaluate_cross_source_connection(self, entity1: Dict[str, Any], entity2: Dict[str, Any],
                                          bridge: Dict[str, Any]) -> Optional[str]:
        """ Uses LLM to suggest a relationship type between two entities linked via a bridge entity. """
        prompt = f"""
        Analyze the potential relationship between two entities identified from different sources (CSV and PDF),
        which are both linked to a common 'bridge' entity.

        Entity 1 (from CSV context):
        - Name: {entity1.get('name', 'N/A')}
        - Type: {entity1.get('type', 'N/A')}

        Entity 2 (from PDF context):
        - Name: {entity2.get('name', 'N/A')}
        - Type: {entity2.get('type', 'N/A')}

        Common Bridge Entity:
        - Name: {bridge.get('bridge_name', 'N/A')}
        - Type: {bridge.get('bridge_type', 'N/A')}

        Based on these entities and their types, is there a likely direct relationship between Entity 1 and Entity 2?
        If YES, suggest a concise, descriptive relationship type in UPPER_SNAKE_CASE representing how Entity 1 relates to Entity 2 (e.g., MANUFACTURES, INTERACTS_WITH, IS_VARIANT_OF, CITES).
        If NO, or if the relationship is too indirect or uncertain, respond with "NO".

        Consider the types: A {entity1.get('type')} and a {entity2.get('type')} related via a {bridge.get('bridge_type')}.

        Response (Relationship type in UPPER_SNAKE_CASE or "NO"):
        """
        try:
            response = self.llm.invoke(prompt).content.strip().upper() # Use invoke for newer LangChain, get content

            if response == "NO" or len(response) < 3 or ' ' in response: # Basic validation
                logger.debug(f"LLM evaluation suggests NO direct relationship between {entity1.get('name')} and {entity2.get('name')}.")
                return None
            else:
                 # Sanitize just in case LLM adds weird characters
                 relation_type = self.sanitize_label(response)
                 logger.debug(f"LLM suggested relationship '{relation_type}' between {entity1.get('name')} and {entity2.get('name')}.")
                 # Discover the type if new
                 if not self.schema_manager.has_relation_type(relation_type):
                      self.schema_manager.discover_relation_type(relation_type, 0.7, "llm_cross_ref_name_based")
                 return relation_type
        except Exception as e:
            logger.warning(f"LLM call failed during cross-source connection evaluation: {e}", exc_info=True)
            return None # Fail safe: don't create relationship if LLM fails

    def _evaluate_vector_similarity_connection(self, entity1: Dict[str, Any], entity2: Dict[str, Any],
                                              similarity: float) -> bool:
        """ Uses LLM to validate if a connection based on vector similarity makes sense. """
        prompt = f"""
        Evaluate if a direct relationship should be created between two entities based on their semantic similarity,
        calculated using vector embeddings.

        Entity 1 (e.g., from CSV):
        - Name: {entity1.get('name', 'N/A')}
        - Type: {entity1.get('type', 'N/A')}

        Entity 2 (e.g., from PDF):
        - Name: {entity2.get('name', 'N/A')}
        - Type: {entity2.get('type', 'N/A')}

        Vector Similarity Score: {similarity:.4f} (Range: 0 to 1, higher means more similar)
        Threshold considered: {self.vector_similarity_threshold}

        Considering the entity names, types, and the high similarity score, does it make semantic sense
        to create a 'CROSS_SOURCE_SIMILAR_TO' relationship between them? They might represent the same concept,
        related concepts, or it could be a coincidental similarity. Use your judgment.

        Respond with YES or NO.

        Response (YES or NO):
        """
        try:
            response = self.llm.invoke(prompt).content.strip().upper()
            decision = "YES" in response
            logger.debug(f"LLM evaluation for vector similarity ({similarity:.4f}) between {entity1.get('name')} and {entity2.get('name')}: {'Connect' if decision else 'Do not connect'}")
            return decision
        except Exception as e:
            logger.warning(f"LLM call failed during vector similarity connection evaluation: {e}", exc_info=True)
            # Fallback to simple threshold check if LLM fails
            return similarity >= self.vector_similarity_threshold

    def get_statistics(self) -> Dict[str, Any]:
        """ Get processing statistics """
        self.stats["end_time"] = datetime.now().isoformat()
        # Get latest stats from components
        resolution_stats = self.entity_resolution.get_stats()
        self.stats["entity_resolution"] = resolution_stats
        self.stats["entities_created"] = resolution_stats.get("new_entities", 0) # Update final counts
        self.stats["entities_merged"] = resolution_stats.get("merged_entities", 0)

        # Get schema stats if available
        if hasattr(self.schema_manager, 'get_pending_changes_count'):
             self.stats["pending_schema_changes"] = self.schema_manager.get_pending_changes_count()
        elif hasattr(self.schema_manager, 'pending_changes'): # Fallback to direct access if method not present
             self.stats["pending_schema_changes"] = {
                 "entity_types": len(self.schema_manager.pending_changes.get("entity_types", [])),
                 "relation_types": len(self.schema_manager.pending_changes.get("relation_types", []))
             }

        # Calculate total relationships created
        self.stats["total_relationships_created"] = (
            self.stats["csv"]["relationships_created"] +
            self.stats["pdf"]["relationships_created"] +
            self.stats["cross_ref"]["name_based_rels_created"] +
            self.stats["cross_ref"]["vector_based_rels_created"]
        )

        # Clean up stats dictionary (optional)
        # del self.stats["csv"]["relationships_created"] # etc. if redundant with total

        return self.stats