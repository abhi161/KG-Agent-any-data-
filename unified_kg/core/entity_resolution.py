from typing import Dict, List, Optional, Tuple, Any
import json
import logging
import numpy as np
from scipy.spatial.distance import cosine
from langchain.llms.base import BaseLLM
from langchain.embeddings.base import Embeddings

logger = logging.getLogger(__name__)

class EntityResolution:
    """
    Entity resolution system to match entities across different data sources using
    multiple strategies including vector embeddings
    """
    
    def __init__(self, llm: BaseLLM, graph_db, embeddings: Embeddings):
        self.llm = llm
        self.graph_db = graph_db
        self.embeddings = embeddings
        self.resolution_stats = {
            "exact_matches": 0,
            "fuzzy_matches": 0,
            "vector_matches": 0,
            "llm_resolved": 0,
            "new_entities": 0
        }
        
        # Initialize Neo4j global vector index
        self._initialize_vector_index()
    
    @staticmethod
    def sanitize_label(label):
        """Convert entity type names to valid Neo4j labels (no spaces)"""
        return label.replace(' ', '_')
    
    def _initialize_vector_index(self):
        """Initialize a global vector index in Neo4j for all entity types"""
        try:
            # Check if Neo4j has the vector plugin installed
            version_query = "RETURN gds.version() AS version"
            result = self.graph_db.query(version_query)
            
            # Create a single global vector index for all entities
            index_query = """
            CALL db.index.vector.createNodeIndex(
                'global_embedding_index',
                'ANY',  // Index any node with an embedding property
                'embedding',
                1536,
                'cosine'
            )
            """
            
            try:
                self.graph_db.query(index_query)
                logger.info("Created global vector index for all entities")
            except Exception as e:
                # Index might already exist, which is fine
                if "already exists" not in str(e):
                    logger.warning(f"Error creating global vector index: {e}")
                else:
                    logger.info("Global vector index already exists")
        
        except Exception as e:
            logger.warning(f"Error initializing vector index: {e}")
            logger.warning("Vector similarity will fall back to in-memory calculations")
    
    def _generate_embedding_for_entity(self, entity_name: str, entity_type: str, properties: Dict[str, Any] = None) -> List[float]:
        """
        Generate an embedding for an entity based on its name, type, and properties
        
        Args:
            entity_name: Name of the entity
            entity_type: Type of the entity
            properties: Additional properties of the entity
            
        Returns:
            List of float values representing the embedding
        """
        # Create a rich context string for the entity
        context = f"Entity Name: {entity_name}\nEntity Type: {entity_type}\n"
        
        # Add important properties to the context
        if properties:
            # Filter out unimportant or non-textual properties
            important_props = {}
            for key, value in properties.items():
                # Skip metadata properties and non-textual values
                if key in ['source', 'row_index', 'chunk_index', 'id'] or not isinstance(value, str):
                    continue
                important_props[key] = value
            
            if important_props:
                context += "Properties:\n"
                for key, value in important_props.items():
                    # Truncate very long values
                    value_str = str(value)
                    if len(value_str) > 500:
                        value_str = value_str[:500] + "..."
                    context += f"{key}: {value_str}\n"
        
        # Generate embedding using the LangChain embeddings model
        try:
            embedding = self.embeddings.embed_query(context)
            return embedding
        except Exception as e:
            logger.warning(f"Error generating embedding: {e}")
            # Return a zero vector as fallback
            return [0.0] * 1536  # Assuming OpenAI's embedding size
    
    def find_matching_entity(self, entity_name: str, entity_type: str, properties: Dict[str, Any] = None) -> Optional[Dict]:
        """
        Find a matching entity in the knowledge graph using multiple strategies:
        1. Exact match
        2. Fuzzy match
        3. Vector similarity
        4. LLM-assisted resolution
        
        Args:
            entity_name: Name of the entity to match
            entity_type: Type of the entity
            properties: Additional properties of the entity
            
        Returns:
            Matched entity or None if no match is found
        """
        if not properties:
            properties = {}
        entity_type = self.sanitize_label(entity_type) 
        
        # Try exact match first (case insensitive)
        query = """
        MATCH (n:{entity_type})
        WHERE toLower(n.name) = toLower($name)
        RETURN n, elementId(n) as id
        """.format(entity_type=entity_type)
        
        results = self.graph_db.query(query, {"name": entity_name})
        
        if results:
            self.resolution_stats["exact_matches"] += 1
            return {"id": results[0]["id"], "node": results[0]["n"]}
        
        # Generate embedding for the entity
        entity_embedding = self._generate_embedding_for_entity(entity_name, entity_type, properties)
        
        # Try vector similarity match
        vector_matches = self._find_similar_entities_by_vector(entity_name, entity_type, entity_embedding)
        
        if vector_matches and vector_matches[0]["score"] > 0.9:  # High confidence vector match
            self.resolution_stats["vector_matches"] += 1
            return {"id": vector_matches[0]["id"], "node": vector_matches[0]["node"]}
        
        # Try fuzzy match using Levenshtein distance
        try:
            # This requires APOC to be installed in Neo4j
            query = """
            MATCH (n:{entity_type})
            WHERE apoc.text.levenshteinSimilarity(toLower(n.name), toLower($name)) > 0.8
            RETURN n, elementId(n) as id, apoc.text.levenshteinSimilarity(toLower(n.name), toLower($name)) as score
            ORDER BY score DESC
            LIMIT 3
            """.format(entity_type=entity_type)
            
            fuzzy_results = self.graph_db.query(query, {"name": entity_name})
            
            if fuzzy_results and fuzzy_results[0]["score"] > 0.9:  # High confidence fuzzy match
                self.resolution_stats["fuzzy_matches"] += 1
                return {"id": fuzzy_results[0]["id"], "node": fuzzy_results[0]["n"]}
            
            # Combine all candidates for LLM resolution
            all_candidates = []
            
            # Add vector candidates
            for match in vector_matches:
                all_candidates.append({
                    "id": match["id"],
                    "node": match["node"],
                    "score": match["score"],
                    "method": "vector"
                })
            
            # Add fuzzy candidates
            for match in fuzzy_results:
                # Check if this candidate is already in the list (from vector search)
                existing = next((c for c in all_candidates if c["id"] == match["id"]), None)
                if existing:
                    # Update with the better score
                    existing["score"] = max(existing["score"], match["score"])
                    existing["method"] = f"{existing['method']},fuzzy"
                else:
                    all_candidates.append({
                        "id": match["id"],
                        "node": match["n"],
                        "score": match["score"],
                        "method": "fuzzy"
                    })
            
            # If we have candidates, use LLM to decide
            if all_candidates:
                # Sort by score
                all_candidates.sort(key=lambda x: x["score"], reverse=True)
                
                # Take top 3
                top_candidates = all_candidates[:3]
                
                match = self._resolve_with_llm(entity_name, entity_type, properties, top_candidates)
                if match:
                    self.resolution_stats["llm_resolved"] += 1
                    return match
        except Exception as e:
            logger.warning(f"Error during fuzzy matching: {e}")
            
        # No match found
        self.resolution_stats["new_entities"] += 1
        return None
    
    def _find_similar_entities_by_vector(self, entity_name: str, entity_type: str, embedding: List[float]) -> List[Dict]:
        """
        Find similar entities using vector similarity with global index
        
        Args:
            entity_name: Name of the entity
            entity_type: Type of the entity
            embedding: Embedding vector of the entity
            
        Returns:
            List of matching entities with similarity scores
        """
        try:
            # Try using Neo4j's global vector index
            query = f"""
            CALL db.index.vector.queryNodes(
                'global_embedding_index',  // Global index for all entities
                10,  // Return top 10 results
                $embedding
            ) YIELD node, score
            WHERE node:{entity_type}  // Filter by entity type
            AND score > 0.7
            RETURN node, elementId(node) as id, score
            ORDER BY score DESC
            LIMIT 5
            """
            
            results = self.graph_db.query(query, {"embedding": embedding})
            
            if results:
                return [{"id": r["id"], "node": r["node"], "score": r["score"]} for r in results]
            
            # Fallback to in-memory similarity calculation if no results from vector index
            # Get all entities of the given type
            query = f"""
            MATCH (n:{entity_type})
            WHERE n.embedding IS NOT NULL
            RETURN n, elementId(n) as id
            LIMIT 100  // Reasonable limit for in-memory processing
            """
            
            entities = self.graph_db.query(query)
            
            if not entities:
                return []
            
            matches = []
            for entity in entities:
                node = entity["n"]
                
                # Calculate cosine similarity
                similarity = 1 - cosine(embedding, node["embedding"])
                
                if similarity > 0.7:  # Same threshold as Neo4j query
                    matches.append({
                        "id": entity["id"],
                        "node": node,
                        "score": similarity
                    })
            
            # Sort by similarity score
            matches.sort(key=lambda x: x["score"], reverse=True)
            
            return matches[:5]  # Return top 5 matches
            
        except Exception as e:
            logger.warning(f"Error in vector similarity search: {e}")
            return []
    
    def _resolve_with_llm(self, entity_name: str, entity_type: str, properties: Dict[str, Any], candidates: List[Dict]) -> Optional[Dict]:
        """
        Use LLM to decide if the entity matches any of the candidates
        
        Args:
            entity_name: Name of the entity
            entity_type: Type of the entity
            properties: Properties of the entity
            candidates: List of candidate matches with scores and methods
            
        Returns:
            Matched entity or None
        """
        entity_type = self.sanitize_label(entity_type) 
        
        # Prepare prompt for LLM
        prompt = f"""
        Task: Determine if the entity matches any of the candidate entities.
        
        Entity:
        - Name: {entity_name}
        - Type: {entity_type}
        - Properties: {json.dumps(properties)}
        
        Candidates:
        """
        
        for i, candidate in enumerate(candidates):
            node = candidate["node"]
            prompt += f"""
            Candidate {i+1}:
            - Name: {node.get('name')}
            - Match Score: {candidate.get('score', 0):.2f} (method: {candidate.get('method', 'unknown')})
            - Properties: {json.dumps({k: v for k, v in node.items() if k not in ['embedding', 'name']})}
            """
        
        prompt += """
        
        Based on the information provided, does the entity match any of the candidates?
        If yes, return the candidate number. If no, return "None".
        
        Your answer (just the candidate number or "None"):
        """
        
        # Get response from LLM
        response = self.llm.predict(prompt).strip()
        
        # Parse the response
        if response.isdigit() and 1 <= int(response) <= len(candidates):
            idx = int(response) - 1
            return {"id": candidates[idx]["id"], "node": candidates[idx]["node"]}
        
        return None
    
    def merge_entity_properties(self, entity_id: int, new_properties: Dict[str, Any], 
                                source: str = None, new_embedding: List[float] = None) -> None:
        """
        Merge new properties into an existing entity and update its embedding
        
        Args:
            entity_id: ID of the entity to update
            new_properties: New properties to merge
            source: Source of the new properties
            new_embedding: New embedding to merge (if None, embedding will be regenerated)
        """
        if not new_properties and not new_embedding:
            return
            
        # First, get the current entity to combine properties properly
        query = """
        MATCH (n) WHERE elementId(n) = $id
        RETURN n, labels(n)[0] as type
        """
        
        results = self.graph_db.query(query, {"id": entity_id})
        
        if not results:
            logger.warning(f"Entity with ID {entity_id} not found for property merging")
            return
        
        current_entity = results[0]["n"]
        entity_type = results[0]["type"]
        
        # Prepare Cypher SET clauses for properties
        set_clauses = []
        params = {"id": entity_id}
        
        for key, value in new_properties.items():
            if key == "name" or key == "id" or key == "embedding":
                continue
                
            param_key = f"prop_{key}"
            set_clauses.append(f"n.{key} = CASE WHEN n.{key} IS NULL THEN ${param_key} ELSE n.{key} END")
            params[param_key] = value
        
        # Add source information
        if source:
            set_clauses.append("""
            n.sources = CASE 
                WHEN n.sources IS NULL THEN [$source] 
                WHEN $source IN n.sources THEN n.sources
                ELSE n.sources + $source 
            END
            """)
            params["source"] = source
        
        # Update embedding if provided or regenerate it
        if new_embedding:
            set_clauses.append("n.embedding = $embedding")
            params["embedding"] = new_embedding
        else:
            # Combine existing and new properties for embedding generation
            combined_properties = {k: v for k, v in current_entity.items() if k not in ["name", "id", "embedding"]}
            combined_properties.update(new_properties)
            
            # Regenerate embedding based on enriched entity
            embedding = self._generate_embedding_for_entity(
                current_entity.get("name"), 
                entity_type,
                combined_properties
            )
            
            set_clauses.append("n.embedding = $embedding")
            params["embedding"] = embedding
        
        if set_clauses:
            # Update entity
            query = f"""
            MATCH (n) WHERE elementId(n) = $id
            SET {', '.join(set_clauses)}
            """
            
            self.graph_db.query(query, params)
    
    def create_new_entity(self, name: str, entity_type: str, properties: Dict[str, Any] = None,
                          source: str = None) -> str:
        """
        Create a new entity with embedding
        
        Args:
            name: Entity name
            entity_type: Entity type
            properties: Entity properties
            source: Data source
            
        Returns:
            Entity ID
        """
        entity_type = self.sanitize_label(entity_type)
        if not properties:
            properties = {}
            
        # Generate a unique ID
        import hashlib
        unique_id = hashlib.md5(f"{entity_type}:{name}".encode()).hexdigest()
        
        # Generate embedding
        embedding = self._generate_embedding_for_entity(name, entity_type, properties)
        
        # Create entity
        query = f"""
        CREATE (e:{entity_type} {{id: $id, name: $name, embedding: $embedding}})
        RETURN elementId(e) as id
        """
        
        params = {
            "id": unique_id,
            "name": name,
            "embedding": embedding
        }
        
        result = self.graph_db.query(query, params)
        entity_id = result[0]["id"]
        
        # Add properties
        if properties or source:
            property_params = {"id": entity_id}
            set_clauses = []
            
            for k, v in properties.items():
                if k not in ["id", "name", "embedding"]:
                    property_params[k] = v
                    set_clauses.append(f"e.{k} = ${k}")
            
            if source:
                property_params["source"] = source
                set_clauses.append("e.sources = [$source]")
            
            if set_clauses:
                property_query = f"""
                MATCH (e) WHERE elementId(e) = $id
                SET {', '.join(set_clauses)}
                """
                
                self.graph_db.query(property_query, property_params)
        
        return entity_id
    
    def get_stats(self) -> Dict[str, int]:
        """Get entity resolution statistics"""
        return self.resolution_stats