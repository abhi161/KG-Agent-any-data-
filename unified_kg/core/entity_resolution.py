# unified_kg/core/entity_resolution.py
from typing import Dict, List, Optional, Tuple, Any
import json
import logging
from langchain.llms.base import BaseLLM
from langchain.embeddings.base import Embeddings

logger = logging.getLogger(__name__)

class EntityResolution:
    """
    Entity resolution system to match entities across different data sources
    """
    
    def __init__(self, llm: BaseLLM, graph_db, embeddings: Optional[Embeddings] = None):
        self.llm = llm
        self.graph_db = graph_db
        self.embeddings = embeddings
        self.resolution_stats = {
            "exact_matches": 0,
            "fuzzy_matches": 0,
            "llm_resolved": 0,
            "new_entities": 0
        }
    
    @staticmethod
    def sanitize_label(label):
        """Convert entity type names to valid Neo4j labels (no spaces)"""
        return label.replace(' ', '_')

    def find_matching_entity(self, entity_name: str, entity_type: str, properties: Dict[str, Any] = None) -> Optional[Dict]:
        """
        Find a matching entity in the knowledge graph
        
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
            
            results = self.graph_db.query(query, {"name": entity_name})
            
            if results:
                # If score is very high, return the match
                if results[0]["score"] > 0.9:
                    self.resolution_stats["fuzzy_matches"] += 1
                    return {"id": results[0]["id"], "node": results[0]["n"]}
                
                # Otherwise use LLM to decide
                match = self._resolve_with_llm(entity_name, entity_type, properties, results)
                if match:
                    self.resolution_stats["llm_resolved"] += 1
                    return match
        except Exception as e:
            logger.warning(f"Error during fuzzy matching: {e}")
            
        # No match found
        self.resolution_stats["new_entities"] += 1
        return None
    
    def _resolve_with_llm(self, entity_name: str, entity_type: str, properties: Dict[str, Any], candidates: List[Dict]) -> Optional[Dict]:
        """
        Use LLM to decide if the entity matches any of the candidates
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
            node = candidate["n"]
            prompt += f"""
            Candidate {i+1}:
            - Name: {node.get('name')}
            - Score: {candidate.get('score')}
            - Properties: {json.dumps({k: v for k, v in node.items() if k != 'name'})}
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
            return {"id": candidates[idx]["id"], "node": candidates[idx]["n"]}
        
        return None
    
    def merge_entity_properties(self, entity_id: int, new_properties: Dict[str, Any], source: str = None) -> None:
        """
        Merge new properties into an existing entity
        
        Args:
            entity_id: ID of the entity to update
            new_properties: New properties to merge
            source: Source of the new properties
        """
        if not new_properties:
            return
            
        # Prepare Cypher SET clauses for properties
        set_clauses = []
        params = {"id": entity_id}
        
        for key, value in new_properties.items():
            if key == "name" or key == "id":
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
        
        if set_clauses:
            # Update entity
            query = f"""
            MATCH (n) WHERE elementId(n) = $id
            SET {', '.join(set_clauses)}
            """
            
            self.graph_db.query(query, params)
    
    def get_stats(self) -> Dict[str, int]:
        """Get entity resolution statistics"""
        return self.resolution_stats