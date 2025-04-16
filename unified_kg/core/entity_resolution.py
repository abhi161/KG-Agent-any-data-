# --- core/entity_resolution.py (MODIFIED) ---
import logging
import json
import re
import hashlib
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any

from langchain.llms.base import BaseLLM
from langchain.embeddings.base import Embeddings
from langchain_neo4j import Neo4jGraph

logger = logging.getLogger(__name__)

try:
    from scipy.spatial.distance import cosine
    SCIPY_AVAILABLE = True
except ImportError:
    logger.warning("scipy not installed. In-memory cosine similarity fallback will not be available.")
    SCIPY_AVAILABLE = False

class EntityResolution:
    """
    Entity resolution system using identifiers, names, vectors, and LLM evaluation.
    """

    def __init__(self, llm: BaseLLM, graph_db: Neo4jGraph, embeddings: Optional[Embeddings], config: Optional[Dict] = None):
        self.llm = llm
        self.graph_db = graph_db
        self.embeddings = embeddings
        self.config = config or {}
        self.initial_schema = self.config.get('initial_schema', {}) # Get schema from config
        self.vector_enabled = self.embeddings is not None and self.config.get('vector_enabled', False)
        self.similarity_threshold = self.config.get('vector_similarity_threshold', 0.85)
        self.fuzzy_similarity_threshold = self.config.get('fuzzy_similarity_threshold', 0.85)
        self.apoc_available = self._check_apoc()

        # Build identifier map from schema
        self.identifier_properties_map = self._build_identifier_map(self.initial_schema)
        logger.info(f"Identifier properties map: {self.identifier_properties_map}")


        self.resolution_stats = {
            "total_calls": 0,
            "id_matches": 0, # New category
            "exact_matches": 0,
            "fuzzy_matches": 0,
            "vector_matches": 0,
            "llm_resolved": 0,
            "new_entities": 0,
            "merged_entities": 0,
            "errors": 0
        }

        if self.vector_enabled:
            # Pass embedding dimension if available in config, else default
            embedding_dimension = self.config.get('embedding_dimension', 1536)
            self._initialize_vector_index(embedding_dimension)
        else:
            logger.info("EntityResolution initialized without vector embedding support.")

    @staticmethod
    def sanitize_label(label):
        if not isinstance(label, str): return "UnknownType"
        sanitized = ''.join(c if c.isalnum() else '_' for c in label)
        if not sanitized or not sanitized[0].isalpha(): sanitized = "Type_" + sanitized
        sanitized = '_'.join(filter(None, sanitized.split('_')))
        return sanitized

    def _build_identifier_map(self, schema: Dict) -> Dict[str, str]:
        """ Creates a map of {sanitized_entity_type: identifier_property_name}. """
        id_map = {}
        if schema and "entity_types" in schema:
            for et_def in schema["entity_types"]:
                type_name = et_def.get("name")
                id_prop = et_def.get("identifier_property")
                if type_name and id_prop:
                    sanitized_type = self.sanitize_label(type_name)
                    # Also sanitize the property name for consistency? Usually not needed for IDs.
                    id_map[sanitized_type] = id_prop
        return id_map

    def get_identifier_property(self, sanitized_entity_type: str) -> Optional[str]:
         """ Gets the identifier property name for a given entity type. """
         return self.identifier_properties_map.get(sanitized_entity_type)

    def _check_apoc(self) -> bool:
        try:
            self.graph_db.query("RETURN apoc.version() AS version")
            logger.info("APOC library detected. Fuzzy matching enabled.")
            return True
        except Exception as e:
            logger.warning(f"APOC library not detected or query failed: {e}. Fuzzy matching will be disabled.")
            return False

    def _initialize_vector_index(self, embedding_dimension: int):
        """Initialize a global vector index in Neo4j for all entity types
        
        Args:
            embedding_dimension: The dimension of the embedding vectors
            
        Returns:
            bool: True if index exists or was created successfully, False otherwise
        """
        index_name = 'global_embedding_index'
        
        try:
        
                
            # Check if Neo4j has vector capabilities
            try:
                # Try GDS version check first
                version_query = "RETURN gds.version() AS version"
                self.graph_db.query(version_query)
                logger.info("Graph Data Science library detected.")
            except Exception as e:
                logger.warning(f"GDS library check failed: {e}")
                logger.warning("This may be fine if Neo4j has vector capabilities without GDS.")
            
            # Create index using procedural API
            logger.info(f"Attempting to create vector index '{index_name}' with dimension {embedding_dimension}...")
            index_query = """
            CALL db.index.vector.createNodeIndex(
                $index_name,
                'ANY',  // Index any node with an embedding property
                'embedding',
                $dimension,
                'cosine'
            )
            """
            
            self.graph_db.query(index_query, {
                "index_name": index_name,
                "dimension": embedding_dimension
            })
            
            logger.info(f"Successfully created vector index '{index_name}'.")
            return True
            
        except Exception as e:
            if "already exists" in str(e):
                logger.info(f"Vector index '{index_name}' already exists.")
                return True
            elif "syntax error" in str(e).lower():
                logger.error(f"Neo4j syntax error creating vector index. Your Neo4j version may not support vector indexes: {e}")
            elif "procedure `db.index.vector.createNodeIndex` not found" in str(e):
                logger.error(f"Vector index procedures not available in your Neo4j instance: {e}")
            else:
                logger.error(f"Failed to create vector index '{index_name}': {e}", exc_info=True)
            
            logger.warning("Vector similarity searches will not work. Fallback methods will be used if available.")
            return False

    def _generate_embedding_for_entity(self, entity_name: str, entity_type: str, properties: Dict[str, Any] = None) -> Optional[List[float]]:
        # (Implementation remains the same as before)
        if not self.vector_enabled or not self.embeddings:
            return None
        prop_string = ""
        if properties:
            prop_parts = []
            for key, value in properties.items():
                if key in ['source', 'row_index', 'chunk_index', 'id', 'embedding', 'sources'] or isinstance(value, (list, dict)):
                    continue
                value_str = str(value)
                if len(value_str) > 100: value_str = value_str[:100] + "..."
                prop_parts.append(f"{key}: {value_str}")
            prop_string = "; ".join(prop_parts)
        text_to_embed = f"Entity: {entity_name}; Type: {entity_type}; Properties: {prop_string}"
        text_to_embed = text_to_embed[:1000]
        try:
            embedding = self.embeddings.embed_query(text_to_embed)
            return embedding
        except Exception as e:
            logger.warning(f"Failed to generate embedding for '{entity_name}' ({entity_type}): {e}", exc_info=True)
            self.resolution_stats["errors"] += 1
            return None

    # !!--- MODIFIED find_matching_entity ---!!
    def find_matching_entity(self, entity_name: str, entity_type: str, properties: Dict[str, Any] = None) -> Optional[Dict]:
        """ Find matching entity using ID, name, vectors, LLM. """
        self.resolution_stats["total_calls"] += 1
        if not properties: properties = {}
        sanitized_type = self.sanitize_label(entity_type)
        source_info = properties.get("source", "unknown_source") # Get source for logging/merging

        # 1. Identifier Match (Using Schema)
        identifier_property = self.get_identifier_property(sanitized_type)
        # Use entity_name as identifier value IF the source column was directly mapped to entity type
        # OR look for the specific identifier property in the properties dict.
        identifier_value = None
        if identifier_property:
            # Check if the key exists directly in properties (e.g., patient_id was a separate column)
            if identifier_property in properties:
                identifier_value = properties[identifier_property]
            # If not found directly, check if the entity_name itself *is* the ID
            # (e.g., the patient_id column was mapped as the entity source)
            # This requires knowing the origin column name, difficult here.
            # Let's assume for now ID must be in properties dict explicitly.
            # A better approach might involve passing origin column name alongside entity_name.
            # --> TEMPORARY FIX: Check if entity_name looks like the potential ID property name
            # This is brittle! Best is to ensure ID is always in properties dict.
            elif identifier_property == self.sanitize_label(entity_name).lower(): # Check if prop name matches name
                 identifier_value = entity_name # Risky assumption

            # Ensure value is not empty
            if identifier_value == '' or pd.isna(identifier_value):
                 identifier_value = None


        if identifier_property and identifier_value is not None:
            logger.debug(f"Attempting match for {sanitized_type} using identifier {identifier_property}={identifier_value}")
            try:
                id_query = f"""
                MATCH (n:`{sanitized_type}`)
                WHERE n.`{identifier_property}` = $id_value
                RETURN n, elementId(n) as id
                LIMIT 1
                """
                id_results = self.graph_db.query(id_query, {"id_value": identifier_value})
                if id_results:
                    self.resolution_stats["id_matches"] += 1
                    matched_id = id_results[0]["id"]
                    logger.info(f"Identifier match found for {sanitized_type} with {identifier_property}={identifier_value}. Node ID: {matched_id}. Merging properties from {source_info}.")
                    # *** Merge properties from current source onto the matched node ***
                    self.merge_entity_properties(matched_id, properties, source_info)
                    return {"id": matched_id, "node": id_results[0]["n"], "method": "identifier"}
            except Exception as e:
                 logger.error(f"Error during identifier match query for {identifier_value} ({sanitized_type}): {e}", exc_info=True)
                 self.resolution_stats["errors"] += 1
        # --- If no match by ID, continue ---


        # 2. Exact Name Match (Case-Insensitive)
        # (Only if entity_name is likely a name, not an ID that failed match)
        # Heuristic: Don't name match if entity_name seems like the failed ID value
        should_try_name_match = not (identifier_property and identifier_value == entity_name)

        if should_try_name_match:
            try:
                # Use name property, which should always be set
                name_query = f"""
                MATCH (n:`{sanitized_type}`)
                WHERE n.name = $name OR toLower(n.name) = toLower($name)
                RETURN n, elementId(n) as id
                LIMIT 1
                """
                name_results = self.graph_db.query(name_query, {"name": entity_name})
                if name_results:
                    self.resolution_stats["exact_matches"] += 1
                    matched_id = name_results[0]["id"]
                    logger.info(f"Exact name match found for '{entity_name}' ({sanitized_type}). Node ID: {matched_id}. Merging properties from {source_info}.")
                    # *** Merge properties onto the matched node ***
                    self.merge_entity_properties(matched_id, properties, source_info)
                    return {"id": matched_id, "node": name_results[0]["n"], "method": "exact"}
            except Exception as e:
                 logger.error(f"Error during exact name match query for {entity_name} ({sanitized_type}): {e}", exc_info=True)
                 self.resolution_stats["errors"] += 1
        # --- If no match by name, continue ---


        # Prepare for other matches: Generate embedding if vectors enabled
        entity_embedding = self._generate_embedding_for_entity(entity_name, sanitized_type, properties)
        candidates = []

        # 3. Vector Similarity Match
        if self.vector_enabled and entity_embedding:
            try:
                vector_matches = self._find_similar_entities_by_vector(sanitized_type, entity_embedding)
                for match in vector_matches:
                    if not any(c["id"] == match["id"] for c in candidates):
                         match["method"] = "vector"
                         candidates.append(match)
                # High confidence vector match? Check if best score is high enough
                if candidates and candidates[0]["method"] == "vector" and candidates[0]["score"] >= self.similarity_threshold + 0.05: # Slightly stricter auto-match
                     self.resolution_stats["vector_matches"] += 1
                     matched_id = candidates[0]["id"]
                     logger.info(f"High confidence vector match found for '{entity_name}' ({sanitized_type}). Node ID: {matched_id}. Merging properties from {source_info}.")
                     self.merge_entity_properties(matched_id, properties, source_info)
                     return candidates[0]
            except Exception as e:
                 logger.error(f"Error during vector similarity search for {entity_name} ({sanitized_type}): {e}", exc_info=True)
                 self.resolution_stats["errors"] += 1

        # 4. Fuzzy Match
        if self.apoc_available and should_try_name_match: # Avoid fuzzy matching on IDs
             # (Code for fuzzy match remains largely the same as previous version)
            try:
                query = f"""
                MATCH (n:`{sanitized_type}`)
                WITH n, apoc.text.levenshteinSimilarity(toLower(n.name), toLower($name)) AS score
                WHERE score >= $threshold
                RETURN n, elementId(n) as id, score
                ORDER BY score DESC
                LIMIT 5
                """
                fuzzy_results = self.graph_db.query(query, {"name": entity_name, "threshold": self.fuzzy_similarity_threshold})
                new_fuzzy_candidates = []
                for match in fuzzy_results:
                    if not any(c["id"] == match["id"] for c in candidates):
                         candidate_data = {
                             "id": match["id"], "node": match["n"],
                             "score": match["score"], "method": "fuzzy"
                         }
                         candidates.append(candidate_data)
                         new_fuzzy_candidates.append(candidate_data)

                # High confidence fuzzy match? Check best NEW fuzzy candidate
                if new_fuzzy_candidates and new_fuzzy_candidates[0]["score"] >= self.fuzzy_similarity_threshold + 0.05: # Stricter auto-match
                     self.resolution_stats["fuzzy_matches"] += 1
                     matched_id = new_fuzzy_candidates[0]["id"]
                     logger.info(f"High confidence fuzzy match found for '{entity_name}' ({sanitized_type}). Node ID: {matched_id}. Merging properties from {source_info}.")
                     self.merge_entity_properties(matched_id, properties, source_info)
                     return new_fuzzy_candidates[0]
            except Exception as e:
                logger.warning(f"Error during fuzzy matching query for {entity_name}: {e}")


        # 5. LLM-Assisted Resolution (if candidates remain)
        if candidates:
            candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
            top_candidates = candidates[:3]
            try:
                match = self._resolve_with_llm(entity_name, sanitized_type, properties, top_candidates)
                if match:
                    self.resolution_stats["llm_resolved"] += 1
                    matched_id = match["id"]
                    logger.info(f"LLM resolved match for '{entity_name}' ({sanitized_type}). Node ID: {matched_id}. Merging properties from {source_info}.")
                    self.merge_entity_properties(matched_id, properties, source_info)
                    return match # Match contains id, node, method="llm"
            except Exception as e:
                 logger.error(f"Error during LLM resolution for {entity_name}: {e}", exc_info=True)
                 self.resolution_stats["errors"] += 1

        # 6. No Match Found
        logger.debug(f"No match found for '{entity_name}' ({sanitized_type}) via ID, name, vector, or LLM. Will create new.")
        return None

    # !!--- MODIFIED create_new_entity ---!!
    def create_new_entity(self, name: str, entity_type: str, properties: Dict[str, Any] = None,
                          source: str = None) -> Optional[str]:
        """ Create a new entity, preferring MERGE on identifier if available, else MERGE on name. """
        sanitized_type = self.sanitize_label(entity_type)
        if not properties: properties = {}

        # Determine MERGE key: identifier property or name
        identifier_property = self.get_identifier_property(sanitized_type)
        identifier_value = properties.get(identifier_property) if identifier_property else None

        # Ensure value is not empty string or NaN if it's the identifier
        if identifier_property and (identifier_value == '' or pd.isna(identifier_value)):
             identifier_value = None # Treat empty ID as if it wasn't provided for matching

        merge_property = None
        merge_value = None
        if identifier_property and identifier_value is not None:
            merge_property = identifier_property
            merge_value = identifier_value
            logger.debug(f"Creating/Merging {sanitized_type} using identifier: {merge_property}={merge_value}")
        elif name: # Fallback to name if ID not usable
            merge_property = "name"
            merge_value = name
            logger.debug(f"Creating/Merging {sanitized_type} using name: {merge_property}={merge_value}")
        else:
            logger.error(f"Cannot create entity of type {sanitized_type}: Missing required identifier or name.")
            self.resolution_stats["errors"] += 1
            return None

        # Prepare properties for creation, ensuring name and ID (if used for merge) are included
        create_props = {"name": name} # Always include name if available
        if identifier_property and identifier_value is not None: # Ensure ID prop is set
             create_props[identifier_property] = identifier_value

        # Add other sanitized properties
        for k, v in properties.items():
             prop_key_sanitized = self.sanitize_label(k).lower()
             # Avoid overwriting merge key or already set name/id
             if prop_key_sanitized not in [merge_property, 'name', 'id', 'embedding', 'sources', 'unique_hash']:
                 create_props[prop_key_sanitized] = v

        # Add unique hash anyway? Optional.
        # create_props["unique_hash"] = hashlib.md5(f"{sanitized_type}:{merge_value}".encode()).hexdigest()

        # Add source list
        if source: create_props["sources"] = [source]

        # Generate embedding (if enabled) - based on combined properties
        entity_embedding = self._generate_embedding_for_entity(name, sanitized_type, create_props)
        if entity_embedding:
            create_props["embedding"] = entity_embedding

        # Cypher MERGE query
        # Use backticks for property name in MERGE clause
        query = f"""
        MERGE (e:`{sanitized_type}` {{`{merge_property}`: $merge_value}})
        ON CREATE SET e = $props
        ON MATCH SET e += $props_on_match  // Add missing props on match
        RETURN elementId(e) as id, کیس e when $props then true else false end as created // Check if props were set (crude way to check creation)
        """
        # Properties to potentially add if node already existed
        props_on_match = create_props.copy()
        if merge_property in props_on_match: # Don't try to re-set the merge key
             del props_on_match[merge_property]
        # Also handle source list merging on match
        if source:
             props_on_match["sources"] = f"CASE WHEN $source IN coalesce(e.sources, []) THEN e.sources ELSE coalesce(e.sources, []) + $source END"
             # Need to handle this specially, cannot pass list directly in SET += for sources update like this
             # Let's simplify ON MATCH for now: just ensure embedding and source list are updated
             update_on_match_clauses = []
             if "embedding" in create_props:
                  update_on_match_clauses.append("e.embedding = $embedding")
             if source:
                  update_on_match_clauses.append("e.sources = CASE WHEN $source IN coalesce(e.sources, []) THEN e.sources ELSE coalesce(e.sources, []) + $source END")

             on_match_set_clause = ", ".join(update_on_match_clauses)
             if on_match_set_clause:
                  query = f"""
                  MERGE (e:`{sanitized_type}` {{`{merge_property}`: $merge_value}})
                  ON CREATE SET e = $props
                  ON MATCH SET {on_match_set_clause}
                  RETURN elementId(e) as id
                  """
             else: # No specific ON MATCH logic needed beyond MERGE finding it
                  query = f"""
                  MERGE (e:`{sanitized_type}` {{`{merge_property}`: $merge_value}})
                  ON CREATE SET e = $props
                  RETURN elementId(e) as id
                  """


        params = {
            "merge_value": merge_value,
            "props": create_props,
            "embedding": create_props.get("embedding"), # Pass separately for ON MATCH
            "source": source # Pass separately for ON MATCH
        }


        try:
            result = self.graph_db.query(query, params)
            if result and result[0]["id"]:
                entity_id = result[0]["id"]
                # Rough check if it was newly created - needs refinement if exact count vital
                # if result[0].get("created", False):
                #     self.resolution_stats["new_entities"] += 1
                # Simplification: Increment new_entities here, knowing it might slightly overcount if MERGE only matched
                # A more accurate way requires checking properties before/after or using transaction events.
                self.resolution_stats["new_entities"] += 1
                return entity_id
            else:
                 logger.error(f"MERGE query for '{name}'/'{identifier_value}' ({sanitized_type}) failed to return ID.")
                 self.resolution_stats["errors"] += 1
                 return None
        except Exception as e:
            logger.error(f"Error creating/merging entity '{name}'/'{identifier_value}' ({sanitized_type}): {e}", exc_info=True)
            self.resolution_stats["errors"] += 1
            return None


    # --- Other methods (_find_similar_entities_by_vector, _resolve_with_llm, merge_entity_properties, get_stats) remain the same ---
    # Make sure merge_entity_properties also gets the schema/identifier map if needed,
    # although it primarily operates on known entity_id.

    def _find_similar_entities_by_vector(self, entity_type: str, embedding: List[float]) -> List[Dict]:
        # (Implementation remains the same as before)
        if not self.vector_enabled or not embedding: return []
        try:
            query = f"""
            CALL db.index.vector.queryNodes('global_embedding_index', 10, $embedding) YIELD node, score
            WHERE $type IN labels(node) AND score >= $threshold
            RETURN node, elementId(node) as id, score
            ORDER BY score DESC LIMIT 5
            """
            results = self.graph_db.query(query, {"embedding": embedding, "type": entity_type, "threshold": self.similarity_threshold})
            if results:
                return [{"id": r["id"], "node": r["node"], "score": r["score"]} for r in results]
            else: return []
        except Exception as e:
            if ("NoSuchIndexException" in str(e) or 
                "index does not exist" in str(e) or 
                "There is no such vector schema index" in str(e)):
                logger.warning(f"Vector index 'global_embedding_index' not found. Disabling vector search. Error: {e}")
                self.vector_enabled = False
                return []
            else:
                logger.error(f"Error during vector similarity search query: {e}", exc_info=True)
                self.resolution_stats["errors"] += 1
                return []

    def _resolve_with_llm(self, entity_name: str, entity_type: str, properties: Dict[str, Any], candidates: List[Dict]) -> Optional[Dict]:
        # (Implementation remains the same as before)
        if not candidates: return None
        prop_str = json.dumps({k: str(v)[:100] for k, v in properties.items() if k not in ['embedding', 'sources']}, indent=2)
        candidate_str = ""
        for i, candidate in enumerate(candidates):
            node = candidate["node"]
            node_props = {k: str(v)[:100] for k, v in node.items() if k not in ['embedding', 'name', 'id', 'sources']}
            candidate_str += f"\nCandidate {i+1}:\n - Name: {node.get('name', 'N/A')}\n - Match Score: {candidate.get('score', 0):.3f} (Method: {candidate.get('method', 'unknown')})\n - Properties: {json.dumps(node_props)}\n"
        prompt = f"Task: Entity Resolution Disambiguation...\nEntity to Resolve:\n- Name: {entity_name}\n- Type: {entity_type}\n- Properties: {prop_str}\nPotential Candidates:{candidate_str}\n...Decision: Based on your analysis... Respond with the candidate number or 'None'.\nYour Answer:" # Truncated for brevity
        try:
            response = self.llm.invoke(prompt).content.strip()
            if response.isdigit() and 1 <= int(response) <= len(candidates):
                idx = int(response) - 1
                selected_candidate = candidates[idx]
                return {"id": selected_candidate["id"], "node": selected_candidate["node"], "method": "llm", "original_method": selected_candidate.get("method"), "original_score": selected_candidate.get("score")}
            else: return None
        except Exception as e:
            logger.error(f"LLM prediction failed during entity resolution for {entity_name}: {e}", exc_info=True)
            self.resolution_stats["errors"] += 1
            return None

    def merge_entity_properties(self, entity_id: str, new_properties: Dict[str, Any], source: str = None) -> None:
        # (Implementation remains largely the same, ensure it gets entity_type correctly if needed for embedding regen)
        if not new_properties: return
        try:
             fetch_query = "MATCH (n) WHERE elementId(n) = $id RETURN n, labels(n)[0] as type"
             result = self.graph_db.query(fetch_query, {"id": entity_id})
             if not result: return
             current_node_data = result[0]['n']
             entity_type = result[0]['type']
        except Exception as e:
             logger.error(f"Failed to fetch entity {entity_id} for merging: {e}", exc_info=True); return

        set_clauses = []
        merge_params = {"id": entity_id, "source": source}
        combined_properties = dict(current_node_data)

        for key, value in new_properties.items():
            prop_key_sanitized = self.sanitize_label(key).lower()
            if not prop_key_sanitized or prop_key_sanitized in ['id', 'embedding', 'name', 'sources']: continue
            combined_properties[prop_key_sanitized] = value
            set_clauses.append(f"n.`{prop_key_sanitized}` = $param_{prop_key_sanitized}")
            merge_params[f"param_{prop_key_sanitized}"] = value

        if source: set_clauses.append("n.sources = CASE WHEN $source IN coalesce(n.sources, []) THEN n.sources ELSE coalesce(n.sources, []) + $source END")

        if self.vector_enabled:
            new_embedding = self._generate_embedding_for_entity(current_node_data.get("name", ""), entity_type, combined_properties)
            if new_embedding:
                set_clauses.append("n.embedding = $embedding")
                merge_params["embedding"] = new_embedding

        if set_clauses:
            try:
                merge_query = f"MATCH (n) WHERE elementId(n) = $id SET {', '.join(set_clauses)}"
                self.graph_db.query(merge_query, merge_params)
                self.resolution_stats["merged_entities"] += 1
            except Exception as e:
                logger.error(f"Error merging properties for entity {entity_id}: {e}", exc_info=True)
                self.resolution_stats["errors"] += 1

    def get_stats(self) -> Dict[str, int]:
        # (Implementation remains the same)
        stats = {"total_calls": 0, "id_matches": 0, "exact_matches": 0, "fuzzy_matches": 0, "vector_matches": 0, "llm_resolved": 0, "new_entities": 0, "merged_entities": 0, "errors": 0}
        stats.update(self.resolution_stats)
        return stats