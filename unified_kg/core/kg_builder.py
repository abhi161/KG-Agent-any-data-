# unified_kg/core/kg_builder.py

import logging
import pandas as pd
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import os
import regex as re # Using 'regex' as it might offer slightly more features than 're'

# Assuming these are in the same directory or properly installed/imported
from .entity_resolution import EntityResolution
from .data_processor import DataProcessor
from .schema_manager import SchemaManager

logger = logging.getLogger(__name__)

class LLMEnhancedKnowledgeGraph:
    """
    Main class for building a unified knowledge graph from structured and unstructured data
    with vector embeddings support, using a row-centric approach for CSVs.
    """

    def __init__(self, llm, graph_db, embeddings, schema_path, initial_schema: Optional[Dict[str, Any]] = None,
                 config: Optional[Dict[str, Any]] = None):

        self.llm = llm
        self.graph_db = graph_db
        self.embeddings = embeddings
        self.config = config or {}
        self.config['initial_schema'] = initial_schema # Store initial schema in config
        self.vector_enabled = self.config.get("vector_enabled", False) and self.embeddings is not None
        self.vector_similarity_threshold = self.config.get("vector_similarity_threshold", 0.85)

        logger.info(f"KG Builder initialized. Vector support: {'Enabled' if self.vector_enabled else 'Disabled'}")

        # Initialize components
        self.schema_manager = SchemaManager(llm, graph_db, initial_schema, schema_path)
        self.entity_resolution = EntityResolution(
            llm=llm,
            graph_db=graph_db,
            embeddings=self.embeddings if self.vector_enabled else None,
            config=self.config # Pass config, which now contains initial_schema
        )
        self.data_processor = DataProcessor(
            llm=llm,
            schema_manager= self.schema_manager,
            embeddings=self.embeddings if self.vector_enabled else None,
            chunk_size=self.config.get("chunk_size", 1000),
            chunk_overlap=self.config.get("chunk_overlap", 100)
        )

        # Build schema relationship map for efficient lookup during CSV processing
        self.schema_relation_map = self._build_schema_relation_map(self.schema_manager.schema)

        # Statistics
        self.stats = {
            "start_time": datetime.now().isoformat(),
            "csv_files_processed": 0,
            "pdf_files_processed": 0,
            "entities_created": 0, # Tracks new entities created via EntityResolution
            "entities_merged": 0,  # Tracks merges handled by EntityResolution
            "relationships_created": 0, # Total rels from all sources
            "csv": {
                "rows_processed": 0,
                "relationships_created": 0,
                "primary_entities_processed": 0,
                "related_entities_processed": 0,
             },
            "pdf": {
                "chunks_processed": 0,
                "entities_extracted": 0,
                "relationships_extracted": 0,
                "relationships_created": 0 # Relationships added from PDF extraction
             },
            "cross_ref": {
                "name_based_rels_created": 0,
                "vector_based_rels_created": 0
             }
        }
        # Initialize schema constraints/indexes (optional basic step)
        self.schema_manager.initialize_schema()


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
                            rel_map[(self.sanitize_label(src), self.sanitize_label(tgt))] = self.sanitize_label(name.upper()) # Ensure rel name is sanitized/uppercase
        logger.debug(f"Built schema relationship map: {rel_map}")
        return rel_map

    @staticmethod
    def sanitize_label(label):
        """Sanitizes labels for use in Neo4j (Types, Properties, Relationship types)."""
        if not isinstance(label, str):
             logger.warning(f"Attempted to sanitize non-string label: {label}. Returning 'UnknownType'.")
             return "UnknownType"
        # Allow underscores, remove other non-alphanumeric
        sanitized = re.sub(r'[^\w_]', '', label)
        # Ensure starts with a letter
        if not sanitized or not sanitized[0].isalpha():
            sanitized = "Type_" + sanitized
        # Remove consecutive underscores (optional cleanup)
        sanitized = re.sub(r'_+', '_', sanitized)
        return sanitized

    # ========================================================================
    # CSV Processing - Row-Centric Approach
    # ========================================================================

    def process_csv_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process a CSV file row by row, focusing on identifying the primary entity
        and its relationships within each row.
        """
        logger.info(f"Processing CSV file row-centrically: {file_path}")
        file_basename = os.path.basename(file_path)
        processing_stats = {
            "file_path": file_path,
            "rows_processed": 0,
            "primary_entities_processed": 0, # Entities identified as the main subject of a row
            "related_entities_processed": 0, # Entities linked from a row
            "relationships_created": 0,
            "column_mappings": {}
        }

        try:
            # 1. Process the CSV using DataProcessor to get DataFrame and mappings
            df, metadata = self.data_processor.process_csv(file_path)
            processing_stats["column_mappings"] = metadata.get("column_mappings", {})
            column_mappings = processing_stats["column_mappings"] # Use inferred mappings
            logger.info(f"CSV loaded: {len(df)} rows. Mappings: {column_mappings}")

            if df.empty or not column_mappings:
                logger.warning(f"Skipping processing for empty DataFrame or missing mappings in {file_basename}")
                return processing_stats

            # Create reverse mapping for quick lookup: {sanitized_entity_type: [column_names]}
            type_to_columns = {}
            for col, etype in column_mappings.items():
                sanitized_etype = self.sanitize_label(etype)
                if sanitized_etype not in type_to_columns:
                    type_to_columns[sanitized_etype] = []
                type_to_columns[sanitized_etype].append(col)

            # 2. Process Row by Row (in batches)
            batch_size = self.config.get("batch_size", 100)
            for start_idx in range(0, len(df), batch_size):
                end_idx = min(start_idx + batch_size, len(df))
                batch_df = df.iloc[start_idx:end_idx]
                logger.debug(f"Processing batch: rows {start_idx} to {end_idx-1}")

                for idx, row in batch_df.iterrows():
                    # Use original DataFrame index if available, otherwise use batch index
                    original_index = idx if isinstance(idx, int) else start_idx + list(batch_df.index).index(idx)
                    try:
                        row_stats = self._process_csv_row(row, column_mappings, type_to_columns, file_basename, original_index)
                        # Aggregate stats from the row
                        processing_stats["rows_processed"] += 1
                        processing_stats["primary_entities_processed"] += row_stats.get("primary_entities_processed", 0)
                        processing_stats["related_entities_processed"] += row_stats.get("related_entities_processed", 0)
                        processing_stats["relationships_created"] += row_stats.get("relationships_created", 0)
                    except Exception as row_error:
                        logger.error(f"Error processing row {original_index} in {file_basename}: {row_error}", exc_info=True)
                        # Optionally add to error stats

            # Update global stats
            self.stats["csv_files_processed"] += 1
            self.stats["csv"]["rows_processed"] += processing_stats["rows_processed"]
            self.stats["csv"]["relationships_created"] += processing_stats["relationships_created"]
            self.stats["csv"]["primary_entities_processed"] = self.stats["csv"].get("primary_entities_processed", 0) + processing_stats["primary_entities_processed"]
            self.stats["csv"]["related_entities_processed"] = self.stats["csv"].get("related_entities_processed", 0) + processing_stats["related_entities_processed"]

        except Exception as e:
            logger.error(f"Failed to process CSV file {file_path}: {e}", exc_info=True)
            processing_stats["error"] = str(e)

        return processing_stats


    # In class LLMEnhancedKnowledgeGraph

    def _process_csv_row(self, row: pd.Series, column_mappings: Dict[str, str],
                        type_to_columns: Dict[str, List[str]], # Note: type_to_columns might not be strictly needed with this logic
                        source_file: str, row_index: int) -> Dict[str, int]:
        """
        Processes a single CSV row to find the primary entity, its properties,
        related entities, and create relationships based on the schema.
        Correctly handles columns mapped to the same type as the primary entity.
        """
        row_stats = {"primary_entities_processed": 0, "related_entities_processed": 0, "relationships_created": 0}
        primary_properties = {"source": source_file, "row_index": row_index} # Collect potential properties first
        primary_candidates = [] # Stores potential primary entities {type: str, id_prop: str, id_value: any, column: str}
        potential_relations_temp = {} # Stores {col_name: (entity_type_sanitized, value)}

        # --- 1a. Initial Pass: Identify Primary Candidates and Separate Mapped vs Unmapped Columns ---
        for col_name, value in row.items():
            if pd.isna(value) or str(value).strip() == '':
                continue # Skip empty values

            col_sanitized_lower = self.sanitize_label(col_name).lower()

            # Is this column mapped to an entity type in this file's context?
            if col_name in column_mappings:
                entity_type_sanitized = self.sanitize_label(column_mappings[col_name])
                identifier_property = self.entity_resolution.get_identifier_property(entity_type_sanitized)

                # Check if THIS column is the designated identifier for THIS entity type
                is_id_column_for_this_type = (identifier_property and identifier_property == col_sanitized_lower) or \
                                            (identifier_property and identifier_property == 'name' and col_sanitized_lower == 'name')

                if is_id_column_for_this_type:
                    # Found a potential primary/anchor entity based on its ID column
                    primary_candidates.append({
                        "type": entity_type_sanitized,
                        "id_prop": identifier_property,
                        "id_value": value,
                        "column": col_name
                    })
                    logger.debug(f"Row {row_index}: Found potential primary/anchor ID: {entity_type_sanitized} in column '{col_name}'")
                else:
                    # Mapped to an entity type, but NOT its specific ID column.
                    # Store temporarily - decide later if it's related or a property of primary.
                    potential_relations_temp[col_name] = (entity_type_sanitized, value)
                    logger.debug(f"Row {row_index}: Column '{col_name}' ({entity_type_sanitized}) flagged as temporarily related/property.")
            else:
                # This column is NOT mapped to an entity type -> Treat as a potential property immediately
                prop_key = self.sanitize_label(col_name).lower()
                primary_properties[prop_key] = value

        # --- 1b. Decide Primary Entity & Finalize Properties/Relations ---
        identified_primary = None
        identified_primary_type = None
        potential_relations = [] # Final list: (col_name, related_entity_type_sanitized, related_value)

        if len(primary_candidates) == 1:
            # Case A: Standard - One primary type identified
            identified_primary = primary_candidates[0]
            identified_primary_type = identified_primary["type"]
            # Add its ID to the properties list
            primary_properties[identified_primary["id_prop"]] = identified_primary["id_value"]
            logger.info(f"Row {row_index}: Identified '{identified_primary_type}' from column '{identified_primary['column']}' as primary.")

        elif len(primary_candidates) > 1:
            # Case B: Multiple primary IDs - Treat as relationship row (or pick one and warn)
            # --- Using "Pick First and Warn" strategy from previous example ---
            identified_primary = primary_candidates[0]
            identified_primary_type = identified_primary["type"]
            primary_properties[identified_primary["id_prop"]] = identified_primary["id_value"] # Add its ID
            logger.warning(f"Row {row_index}: Multiple primary candidate IDs found ({len(primary_candidates)}). Processing based on first: {identified_primary_type} from '{identified_primary['column']}'. Others treated as related.")
            # Add the *other* primary candidates to the temporary relations dict
            for pc in primary_candidates[1:]:
                potential_relations_temp[pc['column']] = (pc['type'], pc['id_value'])
            # --- End "Pick First" Strategy ---
            # Alternative strategies (join table detection) could go here.

        # If a primary entity was chosen (Case A or B), categorize the temporary relations
        if identified_primary_type:
            for col_name, (entity_type_sanitized, value) in potential_relations_temp.items():
                if entity_type_sanitized == identified_primary_type:
                    # Mapped to SAME type as primary -> Treat as Property
                    prop_key = self.sanitize_label(col_name).lower()
                    if prop_key not in primary_properties: # Avoid overwriting essential props like ID
                        primary_properties[prop_key] = value
                        logger.debug(f"Row {row_index}: Treating column '{col_name}' as property of primary type '{identified_primary_type}'.")
                    elif prop_key == identified_primary['id_prop']:
                        # If somehow the non-ID column has the same sanitized name as the ID prop, log warning but maybe allow? Check use case.
                        logger.warning(f"Row {row_index}: Column '{col_name}' has same sanitized name ('{prop_key}') as ID property but different original name. Check data/schema.")
                        primary_properties[prop_key] = value # Allow overwrite? Or prioritize ID column value? Prioritizing ID value (already set).
                    else:
                        logger.debug(f"Row {row_index}: Property '{prop_key}' from column '{col_name}' already set, likely from primary ID or unmapped column processing.")

                else:
                    # Mapped to DIFFERENT type -> Treat as Related Entity Link
                    potential_relations.append((col_name, entity_type_sanitized, value))
                    logger.debug(f"Row {row_index}: Treating column '{col_name}' ({entity_type_sanitized}) as related entity link.")
        else:
            # Case C: No primary candidates found. Cannot categorize potential_relations_temp.
            # Add them all to properties dict for now? Or log error? Let's add them.
            logger.debug(f"Row {row_index}: No primary entity identified. Adding remaining mapped columns {list(potential_relations_temp.keys())} as potential properties.")
            for col_name, (_, value) in potential_relations_temp.items():
                prop_key = self.sanitize_label(col_name).lower()
                if prop_key not in primary_properties:
                    primary_properties[prop_key] = value


        # --- 2. Process Primary Entity Node (if identified) ---
        primary_entity_graph_id = None
        # primary_entity_type is already set if identified_primary is not None

        if identified_primary:
            primary_id_prop = identified_primary["id_prop"]
            primary_id_value = identified_primary["id_value"]

            # Ensure 'name' property exists for resolution, using ID as fallback
            if 'name' not in primary_properties:
                # Check if a 'name' column exists in the raw row data
                raw_name_col = next((col for col in row.index if col.lower() == 'name'), None)
                if raw_name_col and pd.notna(row[raw_name_col]):
                    primary_properties['name'] = row[raw_name_col]
                else:
                    # Fallback to using the ID value string as name
                    primary_properties['name'] = str(primary_id_value)
                    logger.debug(f"Row {row_index}: Using ID value '{primary_id_value}' as fallback name for primary entity {identified_primary_type}")


            name_for_resolution = str(primary_properties['name'])
            logger.debug(f"Row {row_index}: Processing primary entity '{name_for_resolution}' ({identified_primary_type}) with ID {primary_id_prop}={primary_id_value}. Properties: {list(primary_properties.keys())}")

            primary_details = self._add_or_update_entity(
                name=name_for_resolution,
                entity_type=identified_primary_type, # Use the determined type
                properties=primary_properties,
                source=source_file
            )

            if primary_details and primary_details.get("id"):
                primary_entity_graph_id = primary_details["id"]
                row_stats["primary_entities_processed"] += 1
                logger.debug(f"Row {row_index}: Successfully processed primary entity {identified_primary_type} with ID {primary_entity_graph_id}")
            else:
                logger.error(f"Row {row_index}: Failed to process primary entity {identified_primary_type} with ID {primary_id_prop}={primary_id_value}. Skipping relationships for this row.")
                # If Case B (multi-ID row treated as relationship) failed here, we lose the relationship.
                return row_stats

        else:
            # No primary candidate identified earlier
            logger.warning(f"Row {row_index}: No primary entity identifier found based on mappings. Cannot process row structurally.")
            return row_stats

        # --- 3. Process Related Entities and Create Relationships ---
        #    (Only runs if a primary entity was successfully processed in Step 2)
        if primary_entity_graph_id:
            processed_related_entities = {} # {graph_id: {type:..., name:...}}
            # Use the finalized 'potential_relations' list from Step 1b
            for col_name, related_entity_type, related_value in potential_relations:
                # Process the related entity (minimal properties)
                related_properties = {
                    "source": source_file, "row_index": row_index,
                    "name": str(related_value) # Use value as name initially
                }
                # Add identifier if known for this related type
                related_id_prop = self.entity_resolution.get_identifier_property(related_entity_type)
                if related_id_prop:
                    related_properties[related_id_prop] = related_value # Use the actual value from cell

                name_for_related_resolution = str(related_properties['name'])
                logger.debug(f"Row {row_index}: Processing related entity '{name_for_related_resolution}' ({related_entity_type}) from column '{col_name}'")

                related_details = self._add_or_update_entity(
                    name=name_for_related_resolution,
                    entity_type=related_entity_type,
                    properties=related_properties,
                    source=source_file
                )

                # Check success and process relationship
                if related_details and related_details.get("id"):
                    related_entity_graph_id = related_details["id"]
                    row_stats["related_entities_processed"] += 1
                    processed_related_entities[related_entity_graph_id] = {
                        "type": related_entity_type,
                        "name": name_for_related_resolution
                    }
                    logger.debug(f"Row {row_index}: Successfully processed related entity {related_entity_type} with ID {related_entity_graph_id}")

                    # --- 4. Create Relationship based on Schema ---
                    # primary_entity_type and primary_entity_graph_id are known from step 2
                    rel_name = None
                    lookup_key_pr = (identified_primary_type, related_entity_type) # Use identified primary type
                    if lookup_key_pr in self.schema_relation_map:
                        rel_name = self.schema_relation_map[lookup_key_pr]
                        logger.debug(f"Row {row_index}: Creating relationship: ({identified_primary_type}:{primary_entity_graph_id})-[{rel_name}]->({related_entity_type}:{related_entity_graph_id})")
                        self._create_relationship(primary_entity_graph_id, related_entity_graph_id, rel_name, f"{source_file}:row_{row_index}")
                        row_stats["relationships_created"] += 1
                    else:
                        lookup_key_rp = (related_entity_type, identified_primary_type) # Use identified primary type
                        if lookup_key_rp in self.schema_relation_map:
                            rel_name = self.schema_relation_map[lookup_key_rp]
                            logger.debug(f"Row {row_index}: Creating relationship: ({related_entity_type}:{related_entity_graph_id})-[{rel_name}]->({identified_primary_type}:{primary_entity_graph_id})")
                            self._create_relationship(related_entity_graph_id, primary_entity_graph_id, rel_name, f"{source_file}:row_{row_index}")
                            row_stats["relationships_created"] += 1

                    # Log warning only if no relationship was found in either direction
                    if not rel_name:
                        logger.warning(f"Row {row_index}: No relationship defined in schema between primary entity '{identified_primary_type}' and related entity '{related_entity_type}' found in column '{col_name}'.")

                else:
                    logger.error(f"Row {row_index}: Failed processing related entity {related_entity_type} from column '{col_name}'.")

        return row_stats

    # ========================================================================
    # Entity & Relationship Handling Helpers
    # ========================================================================

    def _add_or_update_entity(self, name: str, entity_type: str, properties: Dict[str, Any],
                              source: str) -> Optional[Dict[str, Any]]:
        """
        Add a new entity or update/merge an existing one. Uses EntityResolution component.
        Updates global stats based on resolution outcome.
        Returns dict {"id": graph_id, "merged": bool} on success, None on failure.
        """
        sanitized_entity_type = self.sanitize_label(entity_type) # Ensure type is sanitized

        # Discover entity type if not known by schema manager
        if not self.schema_manager.has_entity_type(sanitized_entity_type):
             logger.info(f"Discovering new entity type: '{sanitized_entity_type}' from source {source}")
             # Use original name for description if available, else use sanitized
             original_type_name = entity_type if entity_type != sanitized_entity_type else sanitized_entity_type
             self.schema_manager.discover_entity_type(original_type_name, confidence=0.9, source=source)
             # Consider processing pending changes immediately or batching them later
             # self.schema_manager.process_pending_changes()

        # Find matching entity using resolution strategies
        match_result = self.entity_resolution.find_matching_entity(name, sanitized_entity_type, properties)

        entity_id = None
        merged = False

        if match_result:
            # Merge properties into the existing entity
            entity_id = match_result["id"]
            self.entity_resolution.merge_entity_properties(entity_id, properties, source)
            # Stats for merged entities are now handled *within* merge_entity_properties
            merged = True
            logger.debug(f"Merged entity '{name}' ({sanitized_entity_type}) with existing ID {entity_id}")
        else:
            # Create a new entity
            entity_id = self.entity_resolution.create_new_entity(name, sanitized_entity_type, properties, source)
            # Stats for new entities are handled within create_new_entity
            merged = False
            if entity_id:
                logger.debug(f"Created new entity '{name}' ({sanitized_entity_type}) with ID {entity_id}")
            else:
                logger.error(f"Failed to create new entity '{name}' ({sanitized_entity_type})")
                return None # Indicate failure

        # Update global stats based on EntityResolution's internal counters AFTER the operation
        resolution_stats = self.entity_resolution.get_stats()
        self.stats["entities_created"] = resolution_stats.get("new_entities", 0)
        self.stats["entities_merged"] = resolution_stats.get("merged_entities", 0)

        return {"id": entity_id, "merged": merged}


    def _create_relationship(self, source_id: str, target_id: str, rel_type: str, source: str, properties: Optional[Dict] = None):
        """ Helper function to create a relationship in Neo4j """
        if not source_id or not target_id or not rel_type:
             logger.warning(f"Skipping relationship creation due to missing ID or type: {source_id=}, {target_id=}, {rel_type=}")
             return

        # Sanitize relationship type (uppercase, underscores - allow starting underscore)
        sanitized_rel_type = re.sub(r'\W', '_', rel_type.upper()).replace('__', '_').strip('_')
        if not sanitized_rel_type:
             sanitized_rel_type = "RELATED_TO"
             logger.warning(f"Relationship type '{rel_type}' sanitized to default 'RELATED_TO'")

        # Base query using MERGE to avoid duplicate relationships between the same nodes
        # Note: MERGE matches based on the relationship *type* and *direction*.
        # Adding properties to MERGE makes it specific to those properties too.
        # For simplicity, we MERGE only on type/direction and update properties.
        # query = f"""
        # MATCH (a), (b)
        # WHERE elementId(a) = $source_id AND elementId(b) = $target_id
        # MERGE (a)-[r:`{sanitized_rel_type}`]->(b)
        # ON CREATE SET r.created_at = timestamp(), r.sources = [$source]
        # ON MATCH SET r.sources = CASE WHEN $source IN coalesce(r.sources, []) THEN r.sources ELSE coalesce(r.sources, []) + $source END,
        #            r.last_seen = timestamp()
        # // Add custom properties using SET, applicable on both CREATE and MATCH
        # """
        query = f"""
        MATCH (a) WHERE elementId(a) = $source_id
        MATCH (b) WHERE elementId(b) = $target_id
        MERGE (a)-[r:`{sanitized_rel_type}`]->(b)
        ON CREATE SET r.created_at = timestamp(), r.sources = [$source]
        ON MATCH SET r.sources = CASE WHEN $source IN coalesce(r.sources, []) THEN r.sources ELSE coalesce(r.sources, []) + $source END,
                r.last_seen = timestamp()
        """
        params = {
            "source_id": source_id,
            "target_id": target_id,
            "source": source
        }

        # Add optional properties using SET clauses
        prop_set_clauses = []
        if properties:
             for key, value in properties.items():
                  prop_key_sanitized = self.sanitize_label(key).lower() # Sanitize property keys
                  if prop_key_sanitized and prop_key_sanitized not in ['sources', 'created_at', 'last_seen']: # Avoid overwriting standard props
                       param_name = f"prop_{prop_key_sanitized}"
                       params[param_name] = value
                       prop_set_clauses.append(f"r.`{prop_key_sanitized}` = ${param_name}")

        if prop_set_clauses:
             query += f" SET {', '.join(prop_set_clauses)}"

        try:
            # logger.debug(f"Executing relationship query: {query} with params: {list(params.keys())}")
            self.graph_db.query(query, params)
            # Increment total relationships - handled by the calling function now
            # self.stats["relationships_created"] += 1 # NO - Let caller increment specific counters
        except Exception as e:
            logger.error(f"Error creating relationship '{sanitized_rel_type}' between {source_id} and {target_id} from source '{source}': {e}", exc_info=True)
            logger.error(f"Failed Query: {query}")
            logger.error(f"Failed Params: {params}")


    # ========================================================================
    # PDF Processing
    # ========================================================================

    def process_pdf_file(self, file_path: str) -> Dict[str, Any]:
        """ Process a PDF file, extract entities/relations, add to graph """
        logger.info(f"Processing PDF file: {file_path}")
        file_basename = os.path.basename(file_path)
        pdf_stats = {
            "file_path": file_path,
            "chunks_processed": 0,
            "entities_extracted": 0,
            "pdf_entities_processed_in_graph": 0, # Count entities added/merged from PDF
            "relationships_extracted": 0,
            "relationships_created": 0 # Relationships added to graph from PDF extraction
        }

        try:
            chunks, metadata = self.data_processor.process_pdf(file_path)
            pdf_stats["chunks_processed"] = len(chunks)
            logger.info(f"PDF processed into {len(chunks)} chunks.")

            all_chunk_entities_details = [] # Store graph details of entities processed per chunk

            for chunk_idx, chunk_text in enumerate(chunks):
                logger.debug(f"Processing chunk {chunk_idx + 1}/{len(chunks)}")

                # 1. Extract entities from chunk text using DataProcessor
                extracted_entities = self.data_processor.extract_entities_from_text(chunk_text)
                pdf_stats["entities_extracted"] += len(extracted_entities)
                logger.debug(f"Extracted {len(extracted_entities)} entities from chunk {chunk_idx}.")

                # 2. Process extracted entities (add/update in graph)
                chunk_graph_entities = {} # Map entity name in chunk to graph ID/details
                for entity_data in extracted_entities:
                    sanitized_type = self.sanitize_label(entity_data["type"])
                    properties = entity_data.get("attributes", {})
                    properties["source"] = file_basename # Base source
                    properties["chunk_index"] = chunk_idx
                    if "context" in entity_data and entity_data["context"]:
                        properties["context"] = entity_data["context"][:800] # Limit context length

                    # Add other metadata if available (e.g., page number from splitter)
                    chunk_meta = self.data_processor.document_metadata.get(chunk_idx, {})
                    properties.update({k:v for k,v in chunk_meta.items() if k not in ['source_file', 'start_index']}) # Add relevant metadata

                    # Use entity name from extraction for resolution
                    entity_name_from_extraction = entity_data["name"]

                    entity_details = self._add_or_update_entity(
                        name=entity_name_from_extraction,
                        entity_type=sanitized_type,
                        properties=properties,
                        source=f"{file_basename}_chunk_{chunk_idx}" # More specific source for merge tracking
                    )

                    if entity_details and entity_details.get("id"):
                         graph_id = entity_details["id"]
                         chunk_graph_entities[entity_name_from_extraction] = {
                             "id": graph_id,
                             "name": entity_name_from_extraction, # Store original name for relation mapping
                             "type": sanitized_type
                         }
                         pdf_stats["pdf_entities_processed_in_graph"] += 1
                         # Store details for potential intra-PDF linking later
                         all_chunk_entities_details.append(chunk_graph_entities[entity_name_from_extraction])
                    else:
                         logger.warning(f"Failed to process entity '{entity_name_from_extraction}' ({sanitized_type}) from chunk {chunk_idx} into graph.")


                # 3. Extract relationships from chunk text (if enough entities processed)
                if len(chunk_graph_entities) > 1:
                    # Pass the original extracted entity structures (name, type) to the relation extractor
                    extracted_relations = self.data_processor.extract_relations_from_text(chunk_text, extracted_entities)
                    pdf_stats["relationships_extracted"] += len(extracted_relations)
                    logger.debug(f"Extracted {len(extracted_relations)} relationships from chunk {chunk_idx}.")

                    # 4. Process extracted relationships (add to graph)
                    for relation_data in extracted_relations:
                        source_entity_name = relation_data["source"]
                        target_entity_name = relation_data["target"]
                        rel_type = relation_data["relation"]

                        # Find the corresponding graph entities processed earlier from this chunk
                        source_graph_entity = chunk_graph_entities.get(source_entity_name)
                        target_graph_entity = chunk_graph_entities.get(target_entity_name)

                        if source_graph_entity and target_graph_entity and rel_type:
                             # Check/discover relationship type in schema
                             sanitized_rel_type = self.sanitize_label(rel_type.upper())
                             if not self.schema_manager.has_relation_type(sanitized_rel_type):
                                  logger.info(f"Discovering new relationship type: '{sanitized_rel_type}' from PDF {file_basename}")
                                  # Use original type name for generation if available
                                  original_rel_type = rel_type if rel_type != sanitized_rel_type else sanitized_rel_type
                                  self.schema_manager.discover_relation_type(original_rel_type, 0.7, f"pdf:{file_basename}:chunk{chunk_idx}")
                                  # Optionally process: self.schema_manager.process_pending_changes()

                             rel_props = {"chunk_index": chunk_idx}
                             if "context" in relation_data and relation_data["context"]:
                                  rel_props["context"] = relation_data["context"][:800]

                             self._create_relationship(
                                 source_graph_entity["id"],
                                 target_graph_entity["id"],
                                 sanitized_rel_type,
                                 f"{file_basename}_chunk_{chunk_idx}", # Specific source
                                 rel_props
                             )
                             pdf_stats["relationships_created"] += 1
                        else:
                             logger.warning(f"Skipping relationship creation in chunk {chunk_idx} due to missing entity mapping or type: {relation_data}. Entities available in chunk: {list(chunk_graph_entities.keys())}")

            # 5. Optional: Intra-PDF cross-referencing (e.g., co-occurrence) - Keep if desired
            # logger.info("Performing intra-PDF entity cross-referencing...")
            # intra_pdf_rels = self._cross_reference_pdf_entities(all_chunk_entities_details, file_basename)
            # pdf_stats["relationships_created"] += intra_pdf_rels # Add count of co-occurrence rels
            # logger.info(f"Created {intra_pdf_rels} intra-PDF relationships (e.g., co-occurrence).")


            # Update global stats
            self.stats["pdf_files_processed"] += 1
            self.stats["pdf"]["chunks_processed"] += pdf_stats["chunks_processed"]
            self.stats["pdf"]["entities_extracted"] += pdf_stats["entities_extracted"]
            self.stats["pdf"]["relationships_created"] += pdf_stats["relationships_created"] # Graph additions from PDF

        except Exception as e:
            logger.error(f"Failed to process PDF file {file_path}: {e}", exc_info=True)
            pdf_stats["error"] = str(e)

        return pdf_stats


    def _cross_reference_pdf_entities(self, all_entities_details: List[Dict[str, Any]],
                                      source_file: str) -> int:
        """
        Simple cross-referencing within a PDF: creates OCCURS_WITH relationships
        between unique entities found across chunks (or within the same chunk).
        This version uses the list of processed entities directly.
        Returns the number of relationships created.
        """
        rels_created = 0
        rel_type = "OCCURS_WITH" # Sanitize if needed: self.sanitize_label("OCCURS_WITH")
        if not self.schema_manager.has_relation_type(rel_type):
            self.schema_manager.discover_relation_type(rel_type, 0.9, f"intra_pdf_inference:{source_file}")
            # Optionally process: self.schema_manager.process_pending_changes()

        # Get unique entity IDs processed from this PDF
        unique_entity_ids = list({entity['id'] for entity in all_entities_details})

        if len(unique_entity_ids) < 2:
            logger.info(f"Skipping intra-PDF cross-referencing for {source_file}, less than 2 unique entities found.")
            return 0

        # Create relationship between all unique pairs
        for i in range(len(unique_entity_ids)):
            for j in range(i + 1, len(unique_entity_ids)):
                id1 = unique_entity_ids[i]
                id2 = unique_entity_ids[j]
                props = {"type": "co-occurrence"} # Add more context if possible (e.g., list of chunk indices)
                # Use helper to create/merge relationship
                self._create_relationship(id1, id2, rel_type, source_file, props)
                rels_created += 1 # Count each potential merge/create attempt

        logger.info(f"Attempted to create {rels_created} intra-PDF co-occurrence relationships for {source_file}.")
        return rels_created

    # ========================================================================
    # Cross-Referencing Between Sources (CSV vs PDF)
    # ========================================================================

    def cross_reference_data_sources(self) -> Dict[str, Any]:
        """
        Cross-reference entities between different data sources (CSV vs PDF).
        Uses name matching via bridge entities and vector similarity (if enabled).
        """
        logger.info("Cross-referencing data sources (CSV vs PDF)...")
        cross_ref_stats = {
            "name_based_evaluations": 0,
            "name_based_rels_created": 0,
            "vector_based_evaluations": 0,
            "vector_based_rels_created": 0,
        }

        try:
            # 1. Name-based cross-referencing
            logger.info("Performing name-based cross-referencing via bridge entities...")
            self._cross_reference_by_name(cross_ref_stats)

            # 2. Vector-based cross-referencing
            if self.vector_enabled:
                 logger.info("Performing vector-based cross-referencing...")
                 self._cross_reference_by_vector(cross_ref_stats)
            else:
                 logger.info("Skipping vector-based cross-referencing (disabled or embeddings unavailable).")

        except Exception as e:
             logger.error(f"Error during cross-referencing sources: {e}", exc_info=True)
             cross_ref_stats["error"] = str(e)

        # Update global stats
        self.stats["cross_ref"]["name_based_rels_created"] += cross_ref_stats["name_based_rels_created"]
        self.stats["cross_ref"]["vector_based_rels_created"] += cross_ref_stats["vector_based_rels_created"]
        # Update total relationships
        self.stats["relationships_created"] += cross_ref_stats["name_based_rels_created"] + cross_ref_stats["vector_based_rels_created"]


        return cross_ref_stats

    def _cross_reference_by_name(self, stats: Dict[str, int]) -> None:
        """ Find entities mentioned in both CSV and PDF sources ('bridge') and evaluate links between their neighbors. """
        try:
            # Query finds nodes where the 'sources' array contains entries indicating both CSV and PDF origins.
            # Adjust 'ENDS WITH' if your source naming convention differs.
            query = """
            MATCH (n)
            WHERE size(coalesce(n.sources, [])) > 1
              AND any(s IN n.sources WHERE s CONTAINS '.csv')  // More robust check
              AND any(s IN n.sources WHERE s CONTAINS '.pdf')
            RETURN elementId(n) AS bridge_id, n.name AS bridge_name, labels(n)[0] AS bridge_type, n.sources as sources
            LIMIT 500 // Limit to avoid excessive processing
            """
            bridge_entities = self.graph_db.query(query)
            logger.info(f"Found {len(bridge_entities)} potential bridge entities for name-based cross-referencing.")

            processed_pairs = set() # Avoid evaluating the same pair multiple times

            for bridge in bridge_entities:
                bridge_id = bridge["bridge_id"]
                logger.debug(f"Evaluating bridge entity: {bridge['bridge_name']} ({bridge['bridge_type']}) ID: {bridge_id}")

                # Find neighbors linked via relationships where the *relationship* source indicates CSV origin
                # This is more precise than checking node sources alone.
                csv_neighbors_query = """
                MATCH (bridge)-[r]-(neighbor)
                WHERE elementId(bridge) = $bridge_id
                  AND any(s IN r.sources WHERE s CONTAINS '.csv') // Check relationship source
                  AND elementId(neighbor) <> $bridge_id // Avoid self-loops
                RETURN DISTINCT elementId(neighbor) AS id, neighbor.name AS name, labels(neighbor)[0] AS type
                LIMIT 25 // Limit neighbors per bridge side
                """
                csv_neighbors = self.graph_db.query(csv_neighbors_query, {"bridge_id": bridge_id})

                # Find neighbors linked via relationships where the *relationship* source indicates PDF origin
                pdf_neighbors_query = """
                MATCH (bridge)-[r]-(neighbor)
                WHERE elementId(bridge) = $bridge_id
                 AND any(s IN r.sources WHERE s CONTAINS '.pdf') // Check relationship source
                 AND elementId(neighbor) <> $bridge_id
                RETURN DISTINCT elementId(neighbor) AS id, neighbor.name AS name, labels(neighbor)[0] AS type
                LIMIT 25
                """
                pdf_neighbors = self.graph_db.query(pdf_neighbors_query, {"bridge_id": bridge_id})

                if not csv_neighbors or not pdf_neighbors:
                    logger.debug(f"Skipping bridge {bridge_id}, neighbors missing from CSV or PDF context.")
                    continue # Need neighbors from both sides to cross-reference

                # Evaluate potential connections between CSV neighbors and PDF neighbors
                for csv_entity in csv_neighbors:
                    for pdf_entity in pdf_neighbors:
                        id1 = csv_entity["id"]
                        id2 = pdf_entity["id"]
                        if id1 == id2: continue # Don't link entity to itself

                        # Ensure pair is processed only once (order doesn't matter)
                        pair_key = tuple(sorted((id1, id2)))
                        if pair_key in processed_pairs: continue
                        processed_pairs.add(pair_key)

                        stats["name_based_evaluations"] += 1

                        # Check if already directly connected
                        existing_rel_query = """
                        MATCH (a)-[r]-(b)
                        WHERE elementId(a) = $id1 AND elementId(b) = $id2
                        RETURN count(r) > 0 AS connected LIMIT 1
                        """
                        already_connected = self.graph_db.query(existing_rel_query, {"id1": id1, "id2": id2})

                        if not already_connected or not already_connected[0]["connected"]:
                            # Use LLM to evaluate if a direct connection makes sense
                            relation_type = self._evaluate_cross_source_connection(
                                csv_entity, pdf_entity, bridge
                            )

                            if relation_type: # LLM suggested a relationship type
                                # Create the relationship
                                rel_props = {
                                    "confidence": 0.7, # Indicate inferred nature
                                    "bridge_entity_id": bridge["bridge_id"],
                                    "bridge_entity_name": bridge["bridge_name"],
                                    "bridge_entity_type": bridge["bridge_type"]
                                }
                                self._create_relationship(
                                    id1,
                                    id2,
                                    relation_type, # Use sanitized type from LLM evaluation method
                                    "cross_reference_name_based",
                                    rel_props
                                )
                                stats["name_based_rels_created"] += 1
                                # self.stats["relationships_created"] updated in main cross_reference method

                        # else: logger.debug(f"Relationship already exists between {id1} and {id2}.")

        except Exception as e:
             logger.error(f"Error during name-based cross-referencing query or processing: {e}", exc_info=True)


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
              AND any(s IN coalesce(n.sources, []) WHERE s CONTAINS '.csv')
            WITH n LIMIT 1000 // Sample more nodes if needed, balance performance
            RETURN elementId(n) AS id, n.name AS name, labels(n)[0] AS type, n.embedding AS embedding
            """
            csv_entities = self.graph_db.query(csv_entities_query)
            logger.info(f"Found {len(csv_entities)} CSV entities with embeddings for vector cross-referencing.")

            processed_pairs = set() # Avoid evaluating the same pair multiple times

            for csv_entity in csv_entities:
                 if not csv_entity.get("embedding") or not csv_entity.get("id"):
                     logger.debug("Skipping CSV entity with missing embedding or ID.")
                     continue

                 entity_id = csv_entity["id"]
                 entity_embedding = csv_entity["embedding"]

                 # Find similar entities originating from PDFs using the vector index
                 vector_query = """
                 CALL db.index.vector.queryNodes($index_name, $top_k, $embedding) YIELD node, score
                 WHERE elementId(node) <> $entity_id // Don't match self
                   AND score >= $threshold
                   AND any(s IN coalesce(node.sources, []) WHERE s CONTAINS '.pdf') // Ensure PDF origin
                 RETURN elementId(node) AS id, node.name AS name, labels(node)[0] AS type, score
                 ORDER BY score DESC
                 LIMIT $limit // Limit similar matches evaluated per source entity
                 """

                 try:
                     similar_pdf_entities = self.graph_db.query(
                         vector_query,
                         {
                             "index_name": "global_embedding_index", # Ensure this matches your index name
                             "top_k": 10, # Look at top 10 neighbors
                             "embedding": entity_embedding,
                             "entity_id": entity_id,
                             "threshold": self.vector_similarity_threshold,
                             "limit": 5
                         }
                     )
                 except Exception as index_e:
                      if "NoSuchIndexException" in str(index_e) or "index does not exist" in str(index_e):
                           logger.error(f"Vector index 'global_embedding_index' not found during query. Disabling vector cross-ref. Error: {index_e}")
                           self.vector_enabled = False # Stop trying
                           return # Exit the vector cross-ref process
                      else:
                           logger.error(f"Error querying vector index for entity {entity_id}: {index_e}", exc_info=True)
                           continue # Skip this entity if query fails


                 if similar_pdf_entities:
                     logger.debug(f"Found {len(similar_pdf_entities)} potential vector matches from PDF for CSV entity {csv_entity['name']} ({entity_id})")

                     for pdf_entity in similar_pdf_entities:
                         id1 = entity_id
                         id2 = pdf_entity["id"]

                         # Ensure pair is processed only once
                         pair_key = tuple(sorted((id1, id2)))
                         if pair_key in processed_pairs: continue
                         processed_pairs.add(pair_key)

                         stats["vector_based_evaluations"] += 1

                         # Check if already directly connected
                         existing_rel_query = """
                         MATCH (a)-[r]-(b)
                         WHERE elementId(a) = $id1 AND elementId(b) = $id2
                         RETURN count(r) > 0 AS connected LIMIT 1
                         """
                         already_connected = self.graph_db.query(existing_rel_query, {"id1": id1, "id2": id2})

                         if not already_connected or not already_connected[0]["connected"]:
                              # Use LLM to validate the semantic connection based on similarity score
                              should_connect = self._evaluate_vector_similarity_connection(
                                  csv_entity, pdf_entity, pdf_entity["score"]
                              )

                              if should_connect:
                                  rel_type = "CROSS_SOURCE_SIMILAR_TO" # Specific type for vector links
                                  rel_type_sanitized = self.sanitize_label(rel_type)
                                  # Discover if needed
                                  if not self.schema_manager.has_relation_type(rel_type_sanitized):
                                       self.schema_manager.discover_relation_type(rel_type, 0.8, "vector_cross_reference") # Use original name

                                  rel_props = {
                                      "similarity_score": round(pdf_entity["score"], 4),
                                      "confidence": round(pdf_entity["score"] * 0.8, 4) # Confidence related to score
                                  }
                                  self._create_relationship(
                                      id1,
                                      id2,
                                      rel_type_sanitized,
                                      "cross_reference_vector_based",
                                      rel_props
                                  )
                                  stats["vector_based_rels_created"] += 1
                                  # self.stats["relationships_created"] updated in main cross_reference method
                         # else: logger.debug(f"Relationship already exists between {id1} and {id2}.")

        except Exception as e:
             logger.error(f"Error during vector-based cross-referencing loop: {e}", exc_info=True)


    def _evaluate_cross_source_connection(self, entity1: Dict[str, Any], entity2: Dict[str, Any],
                                          bridge: Dict[str, Any]) -> Optional[str]:
        """ Uses LLM to suggest a relationship type between two entities linked via a bridge entity. Returns sanitized type or None."""
        prompt = f"""
        Analyze the potential relationship between two entities identified from different sources (CSV and PDF),
        which are both linked to a common 'bridge' entity.

        Entity 1 (from CSV context):
        - Name: {entity1.get('name', 'N/A')}
        - Type: {entity1.get('type', 'N/A')}
        - ID: {entity1.get('id', 'N/A')}

        Entity 2 (from PDF context):
        - Name: {entity2.get('name', 'N/A')}
        - Type: {entity2.get('type', 'N/A')}
        - ID: {entity2.get('id', 'N/A')}

        Common Bridge Entity:
        - Name: {bridge.get('bridge_name', 'N/A')}
        - Type: {bridge.get('bridge_type', 'N/A')}
        - ID: {bridge.get('bridge_id', 'N/A')}

        **Important Consideration:** Entity 1 has ID {entity1.get('id')} and Entity 2 has ID {entity2.get('id')}. If these entities were originally defined with distinct primary keys in their source files (e.g., different doctor_ids), be very cautious about suggesting a strong direct relationship unless the context strongly supports it. A generic 'RELATED_TO' might be acceptable if they are linked through the bridge context but otherwise distinct.

        Based on these entities and their types, is there a likely direct relationship between Entity 1 and Entity 2?
        If YES, suggest a concise, descriptive relationship type in UPPER_SNAKE_CASE representing how Entity 1 relates to Entity 2 (e.g., MANUFACTURES, INTERACTS_WITH, IS_VARIANT_OF, CITES, RELATED_TO).
        If NO, or if the relationship is too indirect or uncertain, respond with "NO".

        Consider the types: A {entity1.get('type')} and a {entity2.get('type')} related via a {bridge.get('bridge_type')}.

        Response (Relationship type in UPPER_SNAKE_CASE or "NO"):
        """
        try:
            response = self.llm.invoke(prompt).content.strip().upper()

            if response == "NO" or len(response) < 3 or ' ' in response or not re.match(r'^[A-Z_]+$', response): # Basic validation
                logger.debug(f"LLM evaluation suggests NO direct relationship between {entity1.get('name')} and {entity2.get('name')}.")
                return None
            else:
                 # Sanitize response to be safe
                 relation_type_sanitized = self.sanitize_label(response)
                 logger.info(f"LLM suggested relationship '{relation_type_sanitized}' between '{entity1.get('name')}' and '{entity2.get('name')}'.")
                 # Discover the type if new (use original LLM response for discovery)
                 if not self.schema_manager.has_relation_type(relation_type_sanitized):
                      self.schema_manager.discover_relation_type(response, 0.7, "llm_cross_ref_name_based")
                 return relation_type_sanitized # Return the sanitized version
        except Exception as e:
            logger.warning(f"LLM call failed during cross-source connection evaluation: {e}", exc_info=True)
            return None

    def _evaluate_vector_similarity_connection(self, entity1: Dict[str, Any], entity2: Dict[str, Any],
                                              similarity: float) -> bool:
        """ Uses LLM to validate if a connection based on vector similarity makes sense. """
        prompt = f"""
        Evaluate if a direct relationship should be created between two entities based on their semantic similarity,
        calculated using vector embeddings.

        Entity 1 (e.g., from CSV):
        - Name: {entity1.get('name', 'N/A')}
        - Type: {entity1.get('type', 'N/A')}
        - ID: {entity1.get('id', 'N/A')}

        Entity 2 (e.g., from PDF):
        - Name: {entity2.get('name', 'N/A')}
        - Type: {entity2.get('type', 'N/A')}
        - ID: {entity2.get('id', 'N/A')}

        Vector Similarity Score: {similarity:.4f} (Range: 0 to 1, higher means more similar)
        Threshold considered: {self.vector_similarity_threshold}

        Considering the entity names, types, and the similarity score (which is >= threshold), does it make semantic sense
        to create a 'CROSS_SOURCE_SIMILAR_TO' relationship between them? They might represent the same concept,
        related concepts, or it could be a coincidental similarity based on the embedding text. Use your judgment.

        Respond with YES or NO.

        Response (YES or NO):
        """
        try:
            response = self.llm.invoke(prompt).content.strip().upper()
            decision = "YES" in response
            logger.debug(f"LLM evaluation for vector similarity ({similarity:.4f}) between {entity1.get('name')} ({entity1.get('id')}) and {entity2.get('name')} ({entity2.get('id')}): {'Connect' if decision else 'Do not connect'}")
            return decision
        except Exception as e:
            logger.warning(f"LLM call failed during vector similarity connection evaluation: {e}", exc_info=True)
            # Fallback to simple threshold check if LLM fails? Or default to False?
            # Defaulting to False to be conservative if LLM fails.
            # return similarity >= self.vector_similarity_threshold
            logger.warning("LLM failed, defaulting to NO connection for vector similarity.")
            return False

    # ========================================================================
    # Statistics
    # ========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """ Get processing statistics """
        self.stats["end_time"] = datetime.now().isoformat()

        # Get latest stats from EntityResolution
        resolution_stats = self.entity_resolution.get_stats()
        self.stats["entity_resolution"] = resolution_stats
        # Update final counts based on resolution stats
        self.stats["entities_created"] = resolution_stats.get("new_entities", 0)
        self.stats["entities_merged"] = resolution_stats.get("merged_entities", 0)

        # Get schema stats
        if hasattr(self.schema_manager, 'get_pending_changes_count'):
             self.stats["pending_schema_changes"] = self.schema_manager.get_pending_changes_count()
        elif hasattr(self.schema_manager, 'pending_changes'):
             self.stats["pending_schema_changes"] = {
                 "entity_types": len(self.schema_manager.pending_changes.get("entity_types", {})),
                 "relation_types": len(self.schema_manager.pending_changes.get("relation_types", {}))
             }

        # Calculate total relationships created across all stages
        self.stats["total_relationships_created"] = (
            self.stats["csv"]["relationships_created"] +
            self.stats["pdf"]["relationships_created"] +
            self.stats["cross_ref"]["name_based_rels_created"] +
            self.stats["cross_ref"]["vector_based_rels_created"]
            # Add intra-pdf rels if calculated separately
        )

        return self.stats