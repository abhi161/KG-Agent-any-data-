# --- core/entity_resolution.py ---
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
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.prompts import PromptTemplate

logger = logging.getLogger(__name__)

try:
    from scipy.spatial.distance import cosine
    SCIPY_AVAILABLE = True
except ImportError:
    logger.warning("scipy not installed. In-memory cosine similarity fallback will not be available.")
    SCIPY_AVAILABLE = False


# Add these at the top of your entity_resolution.py or in a shared models file
from pydantic import BaseModel, Field, field_validator
from typing import Union, Literal, Optional # Ensure Optional is imported

class LLMResolutionDecision(BaseModel):
    decision: Union[int, Literal["None"]] = Field(
        description="The candidate number (1-indexed) if a match is found, or the string 'None' if no definitive match."
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="A brief explanation for the decision, especially if 'None' or if the choice was complex."
    )

    @field_validator('decision')
    def decision_int_must_be_positive(cls, v):
        if isinstance(v, int) and v < 1:
            raise ValueError("Candidate number, if an integer, must be 1 or greater.")
        return v
    

class EntityResolution:
    """
    Entity resolution system using identifiers, names, vectors, and LLM evaluation.
    """

    def __init__(self, llm: BaseLLM, graph_db: Neo4jGraph, embeddings: Optional[Embeddings], config: Optional[Dict] = None):
        self.llm = llm
        self.graph_db = graph_db
        self.embeddings = embeddings
        self.config = config or {}
        self.initial_schema = self.config.get('initial_schema', {}) 
        self.vector_enabled = self.embeddings is not None and self.config.get('vector_enabled', False)
        self.similarity_threshold = self.config.get('vector_similarity_threshold', 0.85)
        self.fuzzy_similarity_threshold = self.config.get('fuzzy_similarity_threshold', 0.65)
        self.apoc_available = self._check_apoc()

        # Build identifier map from schema
        self.identifier_properties_map = self._build_identifier_map(self.initial_schema)
        logger.info(f"Identifier properties map: {self.identifier_properties_map}")

        self.llm_resolution_parser = PydanticOutputParser(pydantic_object=LLMResolutionDecision)
        self.llm_resolution_fixing_parser = OutputFixingParser.from_llm(parser=self.llm_resolution_parser, llm=self.llm)

        self.resolution_stats = {
            "total_calls": 0,
            "id_matches": 0, 
            "exact_matches": 0,
            "normalized_name_matches": 0,
            "fuzzy_matches": 0,
            "vector_matches": 0,
            "llm_resolved": 0,
            "new_entities": 0,
            "merged_entities": 0,
            "errors": 0
        }

        if self.vector_enabled:
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
                    id_map[sanitized_type] = id_prop
        return id_map

    def get_identifier_property(self, sanitized_entity_type: str) -> Optional[str]:
         """ Gets the identifier property name for a given entity type. """
         return self.identifier_properties_map.get(sanitized_entity_type)

    # --- ADD Helper for Name Normalization ---
    def _normalize_name(self, name: str) -> str:
        """
        Generic name normalization that works for any entity type.
        Handles prefixes, suffixes, abbreviations, and special characters.
        """
        if not isinstance(name, str): return ""
        
        # Convert to lowercase
        name_lower = name.lower()
        
        # Remove common prefixes (titles, designations, etc.)
        prefixes = ['dr. ', 'mr. ', 'mrs. ', 'ms. ', 'prof. ', 'dr ', 'mr ', 'mrs ', 'ms ', 'prof ', 
                    'doctor ', 'professor ', 'the ', 'inc. ', 'inc ', 'corp. ', 'corp ', 'ltd. ', 'ltd ']
        for prefix in prefixes:
            if name_lower.startswith(prefix):
                name_lower = name_lower[len(prefix):]
        
        # Handle initial format (e.g., "E. Wong" -> "e wong")
        name_lower = re.sub(r'([a-z])\.\s+', r'\1 ', name_lower)
        
        # Remove common suffixes
        suffixes = [' inc', ' corp', ' ltd', ' llc', ' co', ' company', ' corporation', ' limited']
        for suffix in suffixes:
            if name_lower.endswith(suffix):
                name_lower = name_lower[:-len(suffix)]
        
        # Remove punctuation but preserve meaningful separators
        name_processed = re.sub(r'[.,();\'":!?]', '', name_lower)
        
        # Replace multiple spaces with single space
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
            # we can remove the check since we already have check in start but for safety we can consider it
            try:
                version_query = "RETURN gds.version() AS version"
                self.graph_db.query(version_query)
                logger.info("Graph Data Science library detected.")
            except Exception as e:
                logger.warning(f"GDS library check failed: {e}")
                logger.warning("This may be fine if Neo4j has vector capabilities without GDS.")
            

            # Create index using API
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

    # In EntityResolution class

    def find_matching_entity(self, entity_name: str, entity_type: str, properties: Dict[str, Any] = None) -> Optional[Dict]:
        """
        Find matching entity using a multi-stage approach:
        1. Identifier Match (highest priority)
        2. Name Match (Exact & Normalized) -> If found, use as a strong signal for further checks (Vector/LLM)
        3. Broader Vector Search (if no strong name match)
        4. Broader Fuzzy Search (if no strong name match)
        5. LLM-Assisted Resolution with collected candidates.
        """
        self.resolution_stats["total_calls"] += 1
        if not properties: properties = {}
        sanitized_type = self.sanitize_label(entity_type)
        source_info = properties.get("source", "unknown_source")

        # 1. Identifier Match (Using Schema) - Highest Priority
        identifier_property = self.get_identifier_property(sanitized_type)
        identifier_value = properties.get(identifier_property) if identifier_property else None
        if identifier_property and (identifier_value == '' or pd.isna(identifier_value)):
            identifier_value = None

        if identifier_property and identifier_value is not None:
            logger.debug(f"Attempting match for {sanitized_type} using identifier {identifier_property}={identifier_value}")
            try:
                id_query = f"MATCH (n:`{sanitized_type}`) WHERE n.`{identifier_property}` = $id_value RETURN n, elementId(n) as id LIMIT 1"
                id_results = self.graph_db.query(id_query, {"id_value": identifier_value})
                if id_results:
                    self.resolution_stats["id_matches"] += 1
                    matched_id = id_results[0]["id"]
                    logger.info(f"Identifier match found for {sanitized_type} ID {identifier_property}={identifier_value}. Node ID: {matched_id}. Merging properties from {source_info}.")
                    self.merge_entity_properties(matched_id, properties, source_info)
                    return {"id": matched_id, "node": id_results[0]["n"], "method": "identifier"}
            except Exception as e:
                logger.error(f"Error during identifier match query for {identifier_value} ({sanitized_type}): {e}", exc_info=True)
                self.resolution_stats["errors"] += 1

        # --- If no ID match, proceed to other methods ---
        candidates = []
        # Generate embedding for the incoming entity - needed for vector searches
        incoming_entity_embedding = self._generate_embedding_for_entity(entity_name, sanitized_type, properties)

        # 2. Name Matching (Exact & Normalized) - These become strong initial candidates if found
        name_match_found_node = None # Store the node if a name match occurs
        name_match_method = None

        should_try_name_match = not (identifier_property and identifier_value is not None and str(identifier_value) == entity_name)
        if should_try_name_match:
            # a) Exact Name Match
            try:
                name_query = f"MATCH (n:`{sanitized_type}`) WHERE n.name = $name OR toLower(n.name) = toLower($name) RETURN n, elementId(n) as id LIMIT 1"
                name_results = self.graph_db.query(name_query, {"name": entity_name})
                if name_results:
                    name_match_found_node = name_results[0]["n"]
                    name_match_found_node["id"] = name_results[0]["id"] # Add ID to the node dict
                    name_match_method = "exact_name"
                    logger.info(f"Exact name match found for '{entity_name}' ({sanitized_type}). Node: {name_match_found_node.get('name')}. Will gather more evidence.")
            except Exception as e:
                logger.error(f"Error during exact name matching for {entity_name}: {e}", exc_info=True)

            # b) Normalized Name Match (if no exact match yet)
            if not name_match_found_node:
                normalized_incoming_name = self._normalize_name(entity_name)
                if normalized_incoming_name:
                    try:
                        norm_query = f"MATCH (n:`{sanitized_type}`) WHERE n.normalized_name = $norm_name RETURN n, elementId(n) as id LIMIT 1"
                        norm_results = self.graph_db.query(norm_query, {"norm_name": normalized_incoming_name})
                        if norm_results:
                            name_match_found_node = norm_results[0]["n"]
                            name_match_found_node["id"] = norm_results[0]["id"] # Add ID
                            name_match_method = "normalized_name"
                            logger.info(f"Normalized name match found for '{entity_name}' ({sanitized_type}). Node: {name_match_found_node.get('name')}. Will gather more evidence.")
                    except Exception as e:
                        logger.error(f"Error during normalized name matching for {entity_name}: {e}", exc_info=True)

        # --- Logic based on whether a Name Match was found ---
        if name_match_found_node:
            # A strong name match exists. Add it as the primary candidate.
            # The score here is high because it's a direct name match.
            candidates.append({
                "id": name_match_found_node["id"],
                "node": name_match_found_node,
                "score": 0.98 if name_match_method == "exact_name" else 0.92, # Assign high scores
                "method": name_match_method
            })

            # OPTIONAL: Perform a *targeted* vector search around the name_match_found_node
            # to see if the incoming entity is *also* semantically very close to this specific node.
            # This is to confirm the name match with semantic similarity if embeddings are available.
            if self.vector_enabled and incoming_entity_embedding and name_match_found_node.get('embedding'):
                # Calculate cosine similarity between incoming entity and the name-matched node's embedding
                # Note: _find_similar_entities_by_vector does a broader search. Here we want a direct comparison.
                try:
                    if SCIPY_AVAILABLE:
                        similarity_score = 1 - cosine(incoming_entity_embedding, name_match_found_node['embedding'])
                        logger.debug(f"Direct vector similarity with name-matched node '{name_match_found_node.get('name')}': {similarity_score:.4f}")
                        # You could add this specific similarity as another piece of evidence or adjust score
                        # For simplicity, we'll let the broader LLM use contexts.
                        # Or, if similarity_score is very low, it might cast doubt on the name match.
                        if similarity_score < self.similarity_threshold - 0.1: # If semantically quite different despite name match
                            logger.warning(f"Name match for '{entity_name}' with '{name_match_found_node.get('name')}', but vector similarity is low ({similarity_score:.3f}). Proceeding with caution to LLM.")
                            # Potentially lower the score of this candidate if vector sim is low
                            candidates[0]['score'] *= 0.8 # Example: Penalize score
                    else:
                        logger.debug("Scipy not available for direct cosine similarity calculation with name-matched node.")
                except Exception as e:
                    logger.warning(f"Error calculating direct vector similarity with name-matched node: {e}")

        else:
            # No direct name match was found. Perform broader searches.
            # 3. Broader Vector Similarity Match
            if self.vector_enabled and incoming_entity_embedding:
                try:
                    logger.debug(f"No direct name match. Performing broader vector search for '{entity_name}' ({sanitized_type}).")
                    vector_matches = self._find_similar_entities_by_vector(sanitized_type, incoming_entity_embedding)
                    for match in vector_matches:
                        if not any(c["id"] == match["id"] for c in candidates):
                            match["method"] = "vector"
                            candidates.append(match)
                except Exception as e:
                    logger.error(f"Error during broader vector similarity search for {entity_name} ({sanitized_type}): {e}", exc_info=True)
                    self.resolution_stats["errors"] += 1

            # 4. Broader Fuzzy Match
            if self.apoc_available and should_try_name_match:
                try:
                    logger.debug(f"No direct name match. Performing broader fuzzy search for '{entity_name}' ({sanitized_type}).")
                    query = (
                        f"MATCH (n:`{sanitized_type}`) "
                        "WITH n, apoc.text.levenshteinSimilarity(toLower(n.name), toLower($name)) AS score "
                        "WHERE score >= $threshold RETURN n, elementId(n) as id, score ORDER BY score DESC LIMIT 5"
                    )
                    fuzzy_results = self.graph_db.query(query, {"name": entity_name, "threshold": self.fuzzy_similarity_threshold})
                    for match in fuzzy_results:
                        if not any(c["id"] == match["id"] for c in candidates):
                            candidate_data = {"id": match["id"], "node": match["n"], "score": match["score"], "method": "fuzzy"}
                            candidates.append(candidate_data)
                except Exception as e:
                    logger.warning(f"Error during broader fuzzy matching query for {entity_name}: {e}")

        # 5. LLM-Assisted Resolution (if any candidates were found)
        if candidates:
            candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
            logger.info(f"Found {len(candidates)} potential candidates for '{entity_name}' ({sanitized_type}). Top candidate: '{candidates[0]['node'].get('name', 'N/A')}' by {candidates[0]['method']} (Score: {candidates[0]['score']:.3f}). Sending to LLM.")
            top_llm_candidates = candidates[:3] # Send top 3 to LLM for disambiguation

            try:
                match_from_llm = self._resolve_with_llm(entity_name, sanitized_type, properties, top_llm_candidates)
                if match_from_llm:
                    # LLM made a decision
                    self.resolution_stats["llm_resolved"] += 1
                    # Update stats based on the original method of the LLM-chosen candidate if desired
                    chosen_candidate_original_method = match_from_llm.get("original_method", "llm_confirmed")
                    if chosen_candidate_original_method == "exact_name": self.resolution_stats["exact_matches"] +=1
                    elif chosen_candidate_original_method == "normalized_name": self.resolution_stats["normalized_name_matches"] +=1
                    elif chosen_candidate_original_method == "vector": self.resolution_stats["vector_matches"] +=1
                    elif chosen_candidate_original_method == "fuzzy": self.resolution_stats["fuzzy_matches"] +=1

                    logger.info(f"LLM resolved match for '{entity_name}' to '{match_from_llm['node'].get('name')}'. Method: {chosen_candidate_original_method} confirmed by LLM. Merging properties.")
                    self.merge_entity_properties(match_from_llm["id"], properties, source_info)
                    return match_from_llm
            except Exception as e:
                logger.error(f"Error during LLM resolution for {entity_name}: {e}", exc_info=True)
                self.resolution_stats["errors"] += 1
        else:
            logger.debug(f"No candidates found for '{entity_name}' ({sanitized_type}) after name, vector, and fuzzy searches.")


        # 6. No Match Found after all stages
        logger.debug(f"No definitive match found for '{entity_name}' ({sanitized_type}). Will create new.")
        return None

    # !!--- MODIFIED create_new_entity ---!!
    def create_new_entity(self, name: str, entity_type: str, properties: Dict[str, Any] = None,
                          source: str = None) -> Optional[str]:
        """ Create a new entity, preferring MERGE on identifier if available, else MERGE on name.
            Prioritizes 'name' from properties dict over name argument for node's name property. Stores normalized_name."""
        
        sanitized_type = self.sanitize_label(entity_type)
        if not properties: properties = {}

        # Determine MERGE key: identifier property or name
        identifier_property = self.get_identifier_property(sanitized_type)
        # Get potential ID value directly from properties first
        identifier_value = properties.get(identifier_property) if identifier_property else None
        # Also consider the incoming 'name' argument IF it matches the identifier property name (less common)
        if not identifier_value and identifier_property and name and self.sanitize_label(identifier_property).lower() == 'name':
             identifier_value = name # If 'name' arg IS the id value

        if identifier_property and (identifier_value == '' or pd.isna(identifier_value)):
             identifier_value = None

        merge_property = None
        merge_value = None
        if identifier_property and identifier_value is not None:
            merge_property = identifier_property
            merge_value = identifier_value
            logger.debug(f"Creating/Merging {sanitized_type} using identifier: {merge_property}={merge_value}")
        # Fallback to name ONLY if identifier is not usable AND a name is provided
        elif name:
            # Check if a better name exists in properties before deciding to merge on name
            name_from_props = properties.get('name') # Check the 'name' key in the properties dict
            if name_from_props and not pd.isna(name_from_props) and name_from_props != '':
                 merge_on_this_name = str(name_from_props)
            else:
                 merge_on_this_name = name # Fallback to the 'name' argument if properties['name'] is missing/empty

            merge_property = "name"
            merge_value = merge_on_this_name
            logger.debug(f"Creating/Merging {sanitized_type} using name: {merge_property}={merge_value}")
        else:
            logger.error(f"Cannot create entity of type {sanitized_type}: Missing required identifier or usable name.")
            self.resolution_stats["errors"] += 1
            return None

        # --- Determine the authoritative name for the node's 'name' property ---
        authoritative_name = name # Default to the incoming name argument (e.g., 'D002')
        name_from_props_val = properties.get('name') # Check the actual 'name' property passed in
        if name_from_props_val and not pd.isna(name_from_props_val) and name_from_props_val != '':
            # If a non-empty 'name' exists in the properties dict, STRONGLY prefer it
            authoritative_name = str(name_from_props_val)
            logger.debug(f"Using name '{authoritative_name}' from properties for node instead of initial argument '{name}'.")
        elif not authoritative_name: # Handle case where name arg itself was empty/None
             authoritative_name = f"{sanitized_type}_{identifier_value or 'Unknown'}" # Create a fallback name
             logger.warning(f"No suitable name found for {sanitized_type} with ID '{identifier_value}'. Using generated name: '{authoritative_name}'")
        # --- End Authoritative Name Determination ---


        # Prepare properties for creation using the authoritative name
        create_props = {"name": authoritative_name}
        if identifier_property and identifier_value is not None: # Ensure ID prop is set if used for merge
             create_props[identifier_property] = identifier_value

        # Calculate normalized name based on the authoritative name
        normalized_name = self._normalize_name(authoritative_name)
        if normalized_name:
            create_props["normalized_name"] = normalized_name

        # Add other sanitized properties (ensure authoritative 'name' and 'normalized_name' are not overwritten)
        for k, v in properties.items():
             prop_key_sanitized = self.sanitize_label(k).lower()
             # Exclude keys already handled or special keys
             if prop_key_sanitized not in [merge_property, 'id', 'embedding', 'sources', 'unique_hash', 'normalized_name', identifier_property]:
                 if isinstance(v, (int, float, bool, str)): create_props[prop_key_sanitized] = v
                 elif pd.isna(v): create_props[prop_key_sanitized] = None
                 else: create_props[prop_key_sanitized] = str(v) # Convert others

        if source: create_props["sources"] = [source]

        entity_embedding = self._generate_embedding_for_entity(authoritative_name, sanitized_type, create_props) # Use authoritative name for embedding
        if entity_embedding:
            create_props["embedding"] = entity_embedding

        # Cypher MERGE query
        query = f"""
        MERGE (e:`{sanitized_type}` {{`{merge_property}`: $merge_value}})
        ON CREATE SET e = $props
        RETURN elementId(e) as id
        """
        params = {
            "merge_value": merge_value,
            "props": create_props
        }

        try:
            result = self.graph_db.query(query, params)
            if result and result[0]["id"]:
                entity_id = result[0]["id"]
                self.resolution_stats["new_entities"] += 1 # Increment assuming creation intent
                logger.debug(f"Created/Merged entity '{authoritative_name}' ({sanitized_type}) with ID {entity_id}, merge key {merge_property}='{merge_value}'.")
                return entity_id
            else:
                 logger.error(f"MERGE query for entity using merge key '{merge_property}'='{merge_value}' ({sanitized_type}) failed to return ID.")
                 self.resolution_stats["errors"] += 1
                 return None
        except Exception as e:
            if "already exists" in str(e) and merge_property == "name":
                 logger.warning(f"Constraint violation merging {sanitized_type} on name='{merge_value}'. An entity might already exist with a different identifier. Error: {e}")
            else:
                 logger.error(f"Error creating/merging entity using merge key '{merge_property}'='{merge_value}' ({sanitized_type}): {e}", exc_info=True)
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
        """ Use LLM to decide if the entity matches any candidates, using Pydantic for structured output. """
        if not candidates:
            return None

        # --- Context and Property String Preparation (same as before) ---
        incoming_context = properties.get('context', 'N/A')
        if not isinstance(incoming_context, str): incoming_context = str(incoming_context) if incoming_context is not None else 'N/A'
        if not incoming_context.strip(): incoming_context = 'N/A'
        max_context_len = 300
        if len(incoming_context) > max_context_len:
            incoming_context = incoming_context[:max_context_len] + "..."

        prop_str = json.dumps({k: str(v)[:100] for k, v in properties.items() if k not in ['embedding', 'sources', 'context', 'chunk_index', 'row_index', 'source']}, indent=2)

        candidate_details_str = ""
        for i, candidate_data in enumerate(candidates):
            node = candidate_data["node"]
            candidate_context = node.get('context', 'N/A')
            if not isinstance(candidate_context, str): candidate_context = str(candidate_context) if candidate_context is not None else 'N/A'
            if not candidate_context.strip(): candidate_context = 'N/A'
            if len(candidate_context) > max_context_len:
                candidate_context = candidate_context[:max_context_len] + "..."

            node_props = {k: str(v)[:100] for k, v in node.items() if k not in ['embedding', 'name', 'id', 'sources', 'context']}
            candidate_details_str += (
                f"\nCandidate {i+1}:\n"
                f"- Name: {node.get('name', 'N/A')}\n"
                f"- Match Score: {candidate_data.get('score', 0):.3f} (Method: {candidate_data.get('method', 'unknown')})\n"
                f"- Existing Properties: {json.dumps(node_props)}\n"
                f"- Existing Stored Context: {candidate_context}\n"
            )
        # --- End Context and Property String Preparation ---


        # In EntityResolution._resolve_with_llm prompt_template_str:

        prompt_template_str = """
        Task: Entity Resolution Disambiguation

        You are given an "Entity Mention to Resolve" from a source text and a list of "Potential Existing Candidates" from a knowledge graph.
        Your goal is to determine if the "Entity Mention to Resolve" definitively represents the SAME real-world entity as EXACTLY ONE of the candidates.

        **Entity Mention to Resolve:**
        - Name: {entity_name}
        - Type: {entity_type}
        - Context from Source Text: "{incoming_context}"
        - Other Properties from Source: {prop_str}

        **Potential Existing Candidates (from Knowledge Graph):**
        {candidate_details_str}

        **Analysis Guidelines (General Principles):**

        1.  **Name Compatibility:**
            *   Are the names identical, or plausible variations (e.g., use of initials, full names vs. abbreviations, common titles, minor spelling differences)?
            *   Consider if one name is a more specific or a more general version of the other.

        2.  **Identifier Consistency:**
            *   If unique identifiers are present for any candidate (e.g., `employee_id`, `product_sku`, `document_doi`), do they conflict with any identifier information in the "Entity Mention"?
            *   Explicitly different unique identifiers generally indicate different entities, even if names are similar. An absent identifier in one entity doesn't automatically mean it's different if other evidence is strong.

        3.  **Type Coherence:**
            *   Are the entity types the same or semantically compatible (e.g., "Professor" and "Researcher" might be compatible for a person)?

        4.  **Property and Contextual Alignment:**
            *   **Shared Specific Attributes/Affiliations:** Pay close attention if both the "Entity Mention" (through its properties or source context) and a "Candidate" share very specific, non-trivial attributes or affiliations (e.g., working for the **exact same named organization**, being located at the **exact same specific address**, co-authoring the **same specific publication**). Such overlaps are strong positive indicators if names are also compatible.
            *   **Role/Function/Activity Consistency:** Do their described roles, functions, or activities (derived from properties or context) align, or are they contradictory? Could the role described for the "Entity Mention" plausibly be performed by the "Candidate" given its known attributes?
            *   **Overall Contextual Cohesion:** Does the broader information (all properties, source context, stored context) paint a consistent picture, or are there significant contradictions?

        5.  **Weighing Evidence:**
            *   No single factor is always decisive (except perhaps conflicting unique identifiers).
            *   A strong combination of evidence (e.g., compatible names + shared specific affiliation + consistent roles) is needed for a definitive match.
            *   The provided "Match Score & Method" is a hint from a previous step; use your comprehensive analysis to make the final judgment.

        **Decision Output:**
        Based on your analysis, provide your decision in the specified JSON format.
        - If the "Entity Mention to Resolve" matches **exactly one** candidate based on a strong combination of evidence, set "decision" to the candidate's number (e.g., 1, 2).
        - If it's likely a new distinct entity, matches multiple candidates ambiguously, or you are uncertain due to insufficient or conflicting evidence, set "decision" to "None".
        - Provide a brief "reasoning" for your decision, highlighting the key factors.

        {format_instructions}
        """

        prompt = PromptTemplate(
            template=prompt_template_str,
            input_variables=[
                "entity_name", "entity_type", "incoming_context",
                "prop_str", "candidate_details_str"
            ],
            partial_variables={"format_instructions": self.llm_resolution_parser.get_format_instructions()}
        )

        try:
            chain = prompt | self.llm
            raw_llm_response_content = chain.invoke({
                "entity_name": entity_name,
                "entity_type": entity_type,
                "incoming_context": incoming_context,
                "prop_str": prop_str,
                "candidate_details_str": candidate_details_str
            }).content

            logger.debug(f"LLM raw response for '{entity_name}' disambiguation: '{raw_llm_response_content}'")

            try:
                parsed_output: LLMResolutionDecision = self.llm_resolution_parser.parse(raw_llm_response_content)
            except Exception as parse_error:
                logger.warning(f"Failed to parse LLM response for '{entity_name}' with Pydantic. Attempting to fix. Error: {parse_error}")
                logger.debug(f"Original failing response text: {raw_llm_response_content}")
                parsed_output: LLMResolutionDecision = self.llm_resolution_fixing_parser.parse(raw_llm_response_content)

            logger.info(f"LLM resolution for '{entity_name}': Decision='{parsed_output.decision}', Reasoning='{parsed_output.reasoning}'")

            if isinstance(parsed_output.decision, int):
                decision_num = parsed_output.decision # Already validated by Pydantic validator to be >= 1 if int
                if 1 <= decision_num <= len(candidates): # Now check upper bound
                    idx = decision_num - 1
                    selected_candidate = candidates[idx]
                    logger.info(f"LLM selected candidate {decision_num} ('{selected_candidate['node'].get('name')}') as match for '{entity_name}'.")
                    return {
                        "id": selected_candidate["id"],
                        "node": selected_candidate["node"],
                        "method": "llm_pydantic",
                        "original_method": selected_candidate.get("method"),
                        "original_score": selected_candidate.get("score"),
                        "llm_reasoning": parsed_output.reasoning
                    }
                else:
                    logger.warning(f"LLM returned out-of-range candidate number {decision_num} for '{entity_name}'. Candidates: {len(candidates)}. Treating as no match.")
                    return None
            elif parsed_output.decision == "None":
                logger.info(f"LLM determined no definitive match (decision: 'None') for '{entity_name}'.")
                return None
            else:
                logger.error(f"LLM Pydantic output for '{entity_name}' was unexpected type: {type(parsed_output.decision)} with value {parsed_output.decision}. Treating as no match.")
                return None

        except Exception as e:
            logger.error(f"LLM prediction or Pydantic parsing failed during entity resolution for {entity_name}: {e}", exc_info=True)
            self.resolution_stats["errors"] += 1
            return None
        
        
    def merge_entity_properties(self, entity_id: str, new_properties: Dict[str, Any], source: str = None) -> None:
        if not new_properties:
            logger.debug(f"No new properties provided for merge into entity {entity_id}.")
            return
        try:
             fetch_query = "MATCH (n) WHERE elementId(n) = $id RETURN n, labels(n)[0] as type"
             result = self.graph_db.query(fetch_query, {"id": entity_id})
             if not result:
                 logger.warning(f"Could not fetch entity {entity_id} for merging properties.")
                 return
             current_node_data = result[0]['n']
             entity_type = result[0]['type'] # Get entity type for embedding regen
             current_name = current_node_data.get("name", "") # Get current name
        except Exception as e:
             logger.error(f"Failed to fetch entity {entity_id} for merging: {e}", exc_info=True)
             return

        set_clauses = []
        merge_params = {"id": entity_id, "source": source}
        combined_properties = dict(current_node_data)
        properties_updated = False
        name_changed = False
        new_name_candidate = None # Store the name from new_properties if present

        for key, value in new_properties.items():
            prop_key_sanitized = self.sanitize_label(key).lower()

            # --- Skip internal/special keys ---
            if not prop_key_sanitized or prop_key_sanitized in ['id', 'embedding', 'sources', 'normalized_name']:
                continue

            # --- Store the incoming name candidate ---
            if prop_key_sanitized == 'name':
                if pd.notna(value) and str(value).strip(): # Check if incoming name is valid
                    new_name_candidate = str(value)
                    continue # Handle name update logic separately below

            # --- Handle other properties ---
            current_value = current_node_data.get(prop_key_sanitized)
            new_value_processed = None
            # ... (value processing logic remains the same - handle NaN, int, float, bool, str conversion) ...
            if isinstance(value, (int, float, bool, str)):
                new_value_processed = value
            elif pd.isna(value):
                new_value_processed = None # Use Neo4j null
            else:
                try:
                    new_value_processed = str(value)
                except Exception:
                    logger.warning(f"Could not convert value for property '{prop_key_sanitized}' to string for entity {entity_id}. Skipping property.")
                    continue

            # Only add SET clause if value is new or different, or explicitly setting to null
            if new_value_processed != current_value:
                logger.debug(f"Updating property '{prop_key_sanitized}' for node {entity_id} from '{current_value}' to '{new_value_processed}'")
                properties_updated = True
                set_clauses.append(f"n.`{prop_key_sanitized}` = $param_{prop_key_sanitized}")
                merge_params[f"param_{prop_key_sanitized}"] = new_value_processed
                combined_properties[prop_key_sanitized] = new_value_processed # Update for embedding

        # --- Smart Name Update Logic ---
        if new_name_candidate and new_name_candidate != current_name:
            # Condition: Only update name if the new name doesn't look like the primary ID
            # (This prevents overwriting "John Smith" with "P001")
            # Get the primary ID property and value for comparison
            identifier_property = self.get_identifier_property(entity_type)
            identifier_value_str = str(current_node_data.get(identifier_property)) if identifier_property else None

            # Update name if:
            # 1. Current name is missing/empty OR
            # 2. New name looks substantially different from the ID (heuristic) OR
            # 3. New name is longer than current name (might be more descriptive)
            should_update_name = False
            if not current_name:
                should_update_name = True
                logger.debug(f"Setting name for node {entity_id} to '{new_name_candidate}' because current name is missing.")
            elif identifier_value_str and new_name_candidate.lower() != identifier_value_str.lower():
                # Only update if new name isn't just the ID string itself
                should_update_name = True
                logger.debug(f"Updating name for node {entity_id} from '{current_name}' to '{new_name_candidate}'.")
            # Optional: Add length check? -> elif len(new_name_candidate) > len(current_name): should_update_name = True

            if should_update_name:
                properties_updated = True
                name_changed = True
                current_name = new_name_candidate # Update current_name variable
                set_clauses.append("n.name = $name_param")
                merge_params["name_param"] = current_name
                combined_properties["name"] = current_name # Update for embedding

        # --- Handle source list update (always attempt if source provided) ---
        # ... (source list update logic remains the same) ...
        if source:
            current_sources = current_node_data.get('sources', [])
            if source not in current_sources:
                set_clauses.append("n.sources = CASE WHEN $source IN coalesce(n.sources, []) THEN n.sources ELSE coalesce(n.sources, []) + $source END")
                properties_updated = True
                logger.debug(f"Adding source '{source}' to node {entity_id}")
            else:
                logger.debug(f"Source '{source}' already present for node {entity_id}. Skipping source update.")

        # --- Update Normalized Name if Name Changed ---
        if name_changed:
            new_normalized_name = self._normalize_name(current_name)
            if new_normalized_name and new_normalized_name != current_node_data.get("normalized_name"):
                logger.debug(f"Updating normalized_name for node {entity_id} to '{new_normalized_name}'")
                set_clauses.append("n.normalized_name = $normalized_name")
                merge_params["normalized_name"] = new_normalized_name
                properties_updated = True
                combined_properties["normalized_name"] = new_normalized_name

        # Regenerate embedding only if necessary
        if self.vector_enabled and properties_updated:
            # ... (embedding regeneration logic remains the same, using updated current_name and combined_properties) ...
            logger.debug(f"Regenerating embedding for updated node {entity_id} ('{current_name}')")
            new_embedding = self._generate_embedding_for_entity(current_name, entity_type, combined_properties)
            if new_embedding:
                set_clauses.append("n.embedding = $embedding")
                merge_params["embedding"] = new_embedding

        # Execute update only if changes were detected
        if set_clauses:
            try:
                merge_query = f"MATCH (n) WHERE elementId(n) = $id SET {', '.join(set_clauses)}"
                self.graph_db.query(merge_query, merge_params)
                self.resolution_stats["merged_entities"] += 1 # Count actual merges
                logger.info(f"Successfully merged properties into entity {entity_id} ('{current_name}') from source {source}.")
            except Exception as e:
                logger.error(f"Error merging properties for entity {entity_id}: {e}", exc_info=True)
                self.resolution_stats["errors"] += 1
        else:
            logger.debug(f"No property changes detected for entity {entity_id} from source {source}. Merge skipped.")

    def get_stats(self) -> Dict[str, int]:
        # (Implementation remains the same)
        stats = {"total_calls": 0, "id_matches": 0, "exact_matches": 0, "fuzzy_matches": 0, "vector_matches": 0, "llm_resolved": 0, "new_entities": 0, "merged_entities": 0, "errors": 0}
        stats.update(self.resolution_stats)
        return stats