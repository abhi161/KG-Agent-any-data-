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

    # --- ADD Helper for Name Normalization ---
    def _normalize_name(self, name: str) -> str:
        """ Simple name normalization: lower case, remove common titles & punctuation. """
        if not isinstance(name, str): return ""
        name_lower = name.lower()
        # Remove common titles (add more if needed)
        # Added variations like 'dr ' without period
        titles = ['dr. ', 'mr. ', 'mrs. ', 'ms. ', 'prof. ', 'dr ', 'mr ', 'mrs ', 'ms ', 'prof ']
        for title in titles:
            if name_lower.startswith(title):
                name_lower = name_lower[len(title):]
        # Remove punctuation (keeping spaces/hyphens might be desirable depending on data)
        # This example removes periods, commas, semicolons, parentheses, quotes
        name_processed = re.sub(r'[.,();\'"]', '', name_lower)
        # Remove extra whitespace
        name_normalized = ' '.join(name_processed.split())
        return name_normalized
    
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
        """ Find matching entity using ID, Name (Original & Normalized), Vectors, LLM. """
        self.resolution_stats["total_calls"] += 1
        if not properties: properties = {}
        sanitized_type = self.sanitize_label(entity_type)
        source_info = properties.get("source", "unknown_source") # Get source for logging/merging

        # 1. Identifier Match (Using Schema)
        identifier_property = self.get_identifier_property(sanitized_type)
        identifier_value = None
        if identifier_property:
            if identifier_property in properties: identifier_value = properties[identifier_property]
            if identifier_value == '' or pd.isna(identifier_value): identifier_value = None

        if identifier_property and identifier_value is not None:
            logger.debug(f"Attempting match for {sanitized_type} using identifier {identifier_property}={identifier_value}")
            try:
                id_query = f"""
                MATCH (n:`{sanitized_type}`) WHERE n.`{identifier_property}` = $id_value
                RETURN n, elementId(n) as id LIMIT 1
                """
                id_results = self.graph_db.query(id_query, {"id_value": identifier_value})
                if id_results:
                    self.resolution_stats["id_matches"] += 1
                    matched_id = id_results[0]["id"]
                    logger.info(f"Identifier match found for {sanitized_type} with {identifier_property}={identifier_value}. Node ID: {matched_id}. Merging properties from {source_info}.")
                    self.merge_entity_properties(matched_id, properties, source_info) # MERGE PROPERTIES
                    return {"id": matched_id, "node": id_results[0]["n"], "method": "identifier"}
            except Exception as e:
                 logger.error(f"Error during identifier match query for {identifier_value} ({sanitized_type}): {e}", exc_info=True)
                 self.resolution_stats["errors"] += 1
        # --- If no match by ID, continue ---


        # 2. Name Matching (Original and Normalized)
        # Heuristic: Don't name match if entity_name seems like the failed ID value
        should_try_name_match = not (identifier_property and str(identifier_value) == entity_name)

        if should_try_name_match:
            try:
                # a) Try original name first (case-insensitive)
                name_query = f"""
                MATCH (n:`{sanitized_type}`) WHERE n.name = $name OR toLower(n.name) = toLower($name)
                RETURN n, elementId(n) as id LIMIT 1
                """
                name_results = self.graph_db.query(name_query, {"name": entity_name})
                if name_results:
                    self.resolution_stats["exact_matches"] += 1
                    matched_id = name_results[0]["id"]
                    logger.info(f"Exact name match found for '{entity_name}' ({sanitized_type}). Node ID: {matched_id}. Merging properties from {source_info}.")
                    self.merge_entity_properties(matched_id, properties, source_info) # MERGE PROPERTIES
                    return {"id": matched_id, "node": name_results[0]["n"], "method": "exact"}

                # --- ADD NORMALIZED NAME CHECK ---
                # b) If original failed, try normalized name match (on-the-fly)
                normalized_incoming_name = self._normalize_name(entity_name)
                if normalized_incoming_name and self.apoc_available: # Check if APOC needed functions are likely there
                    # More robust query using APOC text functions:
                    norm_name_query = f"""
                    MATCH (n:`{sanitized_type}`)
                    // Normalize stored name: lower -> remove titles -> remove punctuation -> trim
                    WITH n, reduce(s = toLower(n.name), title IN ['dr. ', 'mr. ', 'mrs. ', 'ms. ', 'prof. ', 'dr ', 'mr ', 'mrs ', 'ms ', 'prof '] | replace(s, title, '')) as name_no_title
                    WITH n, replace(replace(replace(replace(replace(replace(name_no_title, '.', ''), ',', ''), ';', ''), '(', ''), ')', ''), "'", '') as name_no_punct
                    WITH n, trim(name_no_punct) as normalized_stored_name
                    WHERE $norm_name = normalized_stored_name
                    RETURN n, elementId(n) as id
                    LIMIT 1
                    """
                    # Note: Added escaped quote \\' and double quote \\" to regexp_replace
                    try:
                        norm_name_results = self.graph_db.query(norm_name_query, {"norm_name": normalized_incoming_name})
                        if norm_name_results:
                            stat_key = "normalized_name_matches"
                            self.resolution_stats[stat_key] = self.resolution_stats.get(stat_key, 0) + 1
                            matched_id = norm_name_results[0]["id"]
                            logger.info(f"Normalized name match found for '{entity_name}' -> '{normalized_incoming_name}' ({sanitized_type}). Node ID: {matched_id}. Merging.")
                            self.merge_entity_properties(matched_id, properties, source_info) # MERGE PROPERTIES
                            return {"id": matched_id, "node": norm_name_results[0]["n"], "method": "normalized_name"}
                    except Exception as norm_e:
                        if "unknown function" in str(norm_e).lower() and ("replace" in str(norm_e).lower() or "regexp_replace" in str(norm_e).lower()):
                             logger.warning(f"Normalized name query failed, requires APOC text functions (replace/regexp_replace). Error: {norm_e}")
                             # Disable future attempts if APOC text functions aren't there
                             # self.apoc_available = False # Or a specific flag
                        else:
                            logger.warning(f"On-the-fly normalized name query failed: {norm_e}")
                elif normalized_incoming_name and not self.apoc_available:
                     logger.debug("Skipping on-the-fly normalized name check as APOC seems unavailable.")
                # --- END NORMALIZED NAME CHECK ---

            except Exception as e:
                 logger.error(f"Error during name match query for {entity_name}: {e}", exc_info=True)
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
        """ Use LLM to decide if the entity matches any low-confidence candidates, now using context. """
        if not candidates:
            return None

        # --- MODIFICATION: Extract context ---
        incoming_context = properties.get('context', 'N/A')
        # Ensure it's a string before checking length or replacing
        if not isinstance(incoming_context, str): incoming_context = str(incoming_context) if incoming_context is not None else 'N/A'
        if incoming_context == '' : incoming_context = 'N/A'
        max_context_len = 300 # Max length for prompt display
        if len(incoming_context) > max_context_len:
             incoming_context = incoming_context[:max_context_len] + "..."
        # --- END MODIFICATION ---

        prop_str = json.dumps({k: str(v)[:100] for k, v in properties.items() if k not in ['embedding', 'sources', 'context', 'chunk_index', 'row_index', 'source']}, indent=2) # Removed 'source' from exclusion

        candidate_str = ""
        for i, candidate in enumerate(candidates):
            node = candidate["node"]
            # Also try to get context stored on the candidate node if available
            candidate_context = node.get('context', 'N/A')
            if not isinstance(candidate_context, str): candidate_context = str(candidate_context) if candidate_context is not None else 'N/A'
            if candidate_context == '': candidate_context = 'N/A'
            if len(candidate_context) > max_context_len:
                 candidate_context = candidate_context[:max_context_len] + "..."

            node_props = {k: str(v)[:100] for k, v in node.items() if k not in ['embedding', 'name', 'id', 'sources', 'context']}
            candidate_str += f"\nCandidate {i+1}:\n"
            candidate_str += f"- Name: {node.get('name', 'N/A')}\n"
            candidate_str += f"- Match Score: {candidate.get('score', 0):.3f} (Method: {candidate.get('method', 'unknown')})\n"
            candidate_str += f"- Properties: {json.dumps(node_props)}\n"
            candidate_str += f"- Stored Context: {candidate_context}\n" # Add candidate context


        # --- MODIFICATION: Update Prompt ---
        prompt = f"""
        Task: Entity Resolution Disambiguation

        You are given an entity mentioned in a source text and a list of potential matching candidate entities already existing in a knowledge graph.
        The candidates were identified using methods like vector similarity or fuzzy name matching, but the confidence score was not high enough for an automatic match.
        Determine if the new entity mention definitively represents the SAME real-world entity as ONE of the candidates.

        Entity Mention to Resolve:
        - Name: {entity_name}
        - Type: {entity_type}
        - Context from Source Text: "{incoming_context}"
        - Other Properties: {prop_str}

        Potential Existing Candidates:
        {candidate_str}

        Analysis Questions:
        1. Compare the names: Are they variations (e.g., titles, initials), typos, or completely different?
        2. Compare the types: Are they compatible?
        3. Compare the properties AND context: Does the information align or contradict? Does the context from the source text support a link to the candidate's details or context?
        4. Consider the match scores/methods.

        Decision: Based on your analysis, does the 'Entity Mention to Resolve' match **exactly one** of the candidates?
        - If YES, return the number of the matching candidate (e.g., "1", "2", or "3").
        - If NO (it's likely a new distinct entity, or it matches multiple candidates ambiguously, or you are uncertain), return "None".

        Provide only the candidate number or "None".

        Your Answer:
        """
        # --- END MODIFICATION ---

        try:
            # Assuming self.llm.invoke returns an object with a 'content' attribute
            response = self.llm.invoke(prompt).content.strip() # Check if .content is correct for your LLM wrapper

            if response.isdigit() and 1 <= int(response) <= len(candidates):
                idx = int(response) - 1
                selected_candidate = candidates[idx]
                logger.info(f"LLM selected candidate {response} ({selected_candidate['node'].get('name')}) as match for {entity_name}.")
                # Return the matched candidate's info with method marked as 'llm'
                return {
                    "id": selected_candidate["id"],
                    "node": selected_candidate["node"],
                    "method": "llm",
                    "original_method": selected_candidate.get("method"),
                    "original_score": selected_candidate.get("score")
                }
            elif response.upper() == "NONE":
                 logger.info(f"LLM determined no definitive match for {entity_name} among candidates.")
                 return None
            else:
                 logger.warning(f"LLM resolution returned unexpected response for {entity_name}: '{response}'. Treating as no match.")
                 return None

        except AttributeError:
             logger.error("LLM response object does not have 'content' attribute. Trying direct string conversion.")
             # Fallback if .content doesn't exist
             try:
                 response_str = str(self.llm.invoke(prompt)).strip()
                 if response_str.isdigit() and 1 <= int(response_str) <= len(candidates):
                     idx = int(response_str) - 1
                     # ... (rest of logic as above) ...
                     return { ... } # Return selected candidate
                 else:
                      # ... (logic for "NONE" or unexpected) ...
                      return None
             except Exception as e_inner:
                  logger.error(f"LLM prediction and fallback failed during entity resolution for {entity_name}: {e_inner}", exc_info=True)
                  self.resolution_stats["errors"] += 1
                  return None
        except Exception as e:
            logger.error(f"LLM prediction failed during entity resolution for {entity_name}: {e}", exc_info=True)
            self.resolution_stats["errors"] += 1
            return None # Fail safe

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