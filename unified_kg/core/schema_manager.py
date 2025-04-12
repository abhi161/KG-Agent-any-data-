# unified_kg/core/schema_manager.py
from typing import Dict, List, Any, Optional
import json
import logging

logger = logging.getLogger(__name__)

class SchemaManager:
    """
    Manages the knowledge graph schema, including evolution and validation
    """
    
    def __init__(self, llm, graph_db, initial_schema: Optional[Dict[str, Any]] = None):
        self.llm = llm
        self.graph_db = graph_db
        self.schema = initial_schema or {
            "entity_types": [],
            "relation_types": []
        }
        self.pending_changes = {
            "entity_types": [],
            "relation_types": [],
            "properties": []
        }
    
    def initialize_schema(self) -> None:
        """
        Initialize the schema in the database
        """
        if not self.schema:
            return
            
        # Create constraints for entity types
        for entity_type in self.schema.get("entity_types", []):
            self._create_entity_type_constraint(entity_type["name"])
            
        # Create metadata nodes for entity and relation types
        for entity_type in self.schema.get("entity_types", []):
            self._create_entity_type_metadata(entity_type)
            
        for relation_type in self.schema.get("relation_types", []):
            self._create_relation_type_metadata(relation_type)
    
    @staticmethod
    def sanitize_label(label):
        """Convert entity type names to valid Neo4j labels (no spaces)"""
        return label.replace(' ', '_')
    
    def _create_entity_type_constraint(self, entity_type: str) -> None:
        """
        Create a unique constraint for an entity type
        """
        entity_type = self.sanitize_label(entity_type) 
        query = f"""
        CREATE CONSTRAINT IF NOT EXISTS FOR (n:{entity_type}) REQUIRE n.id IS UNIQUE
        """
        
        try:
            self.graph_db.query(query)
            logger.info(f"Created constraint for entity type: {entity_type}")
        except Exception as e:
            logger.warning(f"Failed to create constraint for {entity_type}: {e}")
    
    def _create_entity_type_metadata(self, entity_type: Dict[str, Any]) -> None:
        """
        Create metadata node for an entity type
        """
        query = """
        MERGE (et:_EntityType {name: $name})
        SET et.description = $description,
            et.properties = $properties,
            et.created_at = datetime()
        """
        
        params = {
            "name": entity_type["name"],
            "description": entity_type.get("description", ""),
            "properties": json.dumps(entity_type.get("properties", []))
        }
        
        try:
            self.graph_db.query(query, params)
            logger.info(f"Created metadata for entity type: {entity_type['name']}")
        except Exception as e:
            logger.warning(f"Failed to create metadata for {entity_type['name']}: {e}")
    
    def _create_relation_type_metadata(self, relation_type: Dict[str, Any]) -> None:
        """
        Create metadata node for a relationship type
        """
        query = """
        MERGE (rt:_RelationType {name: $name})
        SET rt.description = $description,
            rt.source_types = $source_types,
            rt.target_types = $target_types,
            rt.created_at = datetime()
        """
        
        params = {
            "name": relation_type["name"],
            "description": relation_type.get("description", ""),
            "source_types": json.dumps(relation_type.get("source_types", [])),
            "target_types": json.dumps(relation_type.get("target_types", []))
        }
        
        try:
            self.graph_db.query(query, params)
            logger.info(f"Created metadata for relation type: {relation_type['name']}")
        except Exception as e:
            logger.warning(f"Failed to create metadata for {relation_type['name']}: {e}")
    
    def discover_entity_type(self, entity_type: str, confidence: float = 0.7, source: str = None) -> None:
        """
        Register discovery of a new entity type
        
        Args:
            entity_type: Name of the discovered entity type
            confidence: Confidence level (0-1)
            source: Source of the discovery
        """
        entity_type = self.sanitize_label(entity_type) 

        # Check if entity type already exists
        for et in self.schema.get("entity_types", []):
            if et["name"].lower() == entity_type.lower():
                return
                
        # Check if already in pending changes
        for et in self.pending_changes["entity_types"]:
            if et["name"].lower() == entity_type.lower():
                return
                
        # Add to pending changes
        self.pending_changes["entity_types"].append({
            "name": entity_type,
            "confidence": confidence,
            "source": source,
            "auto_approve": confidence >= 0.9
        })
        
        logger.info(f"Discovered new entity type: {entity_type} (confidence: {confidence})")
    
    def discover_relation_type(self, relation_type: str, confidence: float = 0.7, source: str = None) -> None:
        """
        Register discovery of a new relation type
        
        Args:
            relation_type: Name of the discovered relation type
            confidence: Confidence level (0-1)
            source: Source of the discovery
        """
        # Check if relation type already exists
        for rt in self.schema.get("relation_types", []):
            if rt["name"].lower() == relation_type.lower():
                return
                
        # Check if already in pending changes
        for rt in self.pending_changes["relation_types"]:
            if rt["name"].lower() == relation_type.lower():
                return
                
        # Add to pending changes
        self.pending_changes["relation_types"].append({
            "name": relation_type,
            "confidence": confidence,
            "source": source,
            "auto_approve": confidence >= 0.9
        })
        
        logger.info(f"Discovered new relation type: {relation_type} (confidence: {confidence})")
    
    def process_pending_changes(self, auto_approve_threshold: float = 0.9) -> None:
        """
        Process pending schema changes
        
        Args:
            auto_approve_threshold: Threshold for automatic approval
        """
        # Auto-approve high-confidence changes
        for entity_type in self.pending_changes["entity_types"]:
            if entity_type.get("confidence", 0) >= auto_approve_threshold:
                self.approve_entity_type(entity_type["name"])
                
        for relation_type in self.pending_changes["relation_types"]:
            if relation_type.get("confidence", 0) >= auto_approve_threshold:
                self.approve_relation_type(relation_type["name"])
    
    def approve_entity_type(self, entity_type_name: str) -> bool:
        """
        Approve a pending entity type
        
        Args:
            entity_type_name: Name of the entity type to approve
            
        Returns:
            True if approved, False otherwise
        """
        
        # Find the entity type in pending changes
        for i, et in enumerate(self.pending_changes["entity_types"]):
            if et["name"].lower() == entity_type_name.lower():
                # Generate description using LLM
                description = self._generate_entity_description(et["name"])
                
                # Create entity type
                entity_type = {
                    "name": et["name"],
                    "description": description,
                    "properties": ["name", "id"]
                }
                
                # Add to schema
                self.schema["entity_types"].append(entity_type)
                
                # Create constraint and metadata
                self._create_entity_type_constraint(entity_type["name"])
                self._create_entity_type_metadata(entity_type)
                
                # Remove from pending changes
                self.pending_changes["entity_types"].pop(i)
                
                logger.info(f"Approved entity type: {entity_type_name}")
                return True
                
        logger.warning(f"Entity type not found in pending changes: {entity_type_name}")
        return False
    
    def approve_relation_type(self, relation_type_name: str) -> bool:
        """
        Approve a pending relation type
        
        Args:
            relation_type_name: Name of the relation type to approve
            
        Returns:
            True if approved, False otherwise
        """
        # Find the relation type in pending changes
        for i, rt in enumerate(self.pending_changes["relation_types"]):
            if rt["name"].lower() == relation_type_name.lower():
                # Generate description using LLM
                description = self._generate_relation_description(rt["name"])
                
                # Create relation type
                relation_type = {
                    "name": rt["name"],
                    "description": description,
                    "properties": []
                }
                
                # Add to schema
                self.schema["relation_types"].append(relation_type)
                
                # Create metadata
                self._create_relation_type_metadata(relation_type)
                
                # Remove from pending changes
                self.pending_changes["relation_types"].pop(i)
                
                logger.info(f"Approved relation type: {relation_type_name}")
                return True
                
        logger.warning(f"Relation type not found in pending changes: {relation_type_name}")
        return False
    
    def _generate_entity_description(self, entity_type: str) -> str:
        """
        Generate a description for an entity type using LLM
        """
        prompt = f"""
        Generate a brief, clear description for an entity type named "{entity_type}" 
        in a knowledge graph. The description should explain what this entity represents 
        and what kind of real-world objects would be modeled as this entity type.
        
        Keep the description under 100 characters.
        """
        
        try:
            return self.llm.predict(prompt).strip()
        except Exception as e:
            logger.warning(f"Error generating description for {entity_type}: {e}")
            return f"Represents a {entity_type.lower()} entity"
    
    def _generate_relation_description(self, relation_type: str) -> str:
        """
        Generate a description for a relation type using LLM
        """
        prompt = f"""
        Generate a brief, clear description for a relationship type named "{relation_type}" 
        in a knowledge graph. The description should explain what this relationship represents 
        and how it connects different entities.
        
        Keep the description under 100 characters.
        """
        
        try:
            return self.llm.predict(prompt).strip()
        except Exception as e:
            logger.warning(f"Error generating description for {relation_type}: {e}")
            return f"Represents a {relation_type.lower()} relationship"
    
    def suggest_schema_improvements(self) -> Dict[str, Any]:
        """
        Use LLM to suggest improvements to the schema
        
        Returns:
            Dictionary with schema improvement suggestions
        """
        # Get current schema
        current_schema = self.schema
        
        # Get sample entities and relationships
        entities = self.graph_db.query(
            "MATCH (n) RETURN DISTINCT labels(n)[0] as type, count(*) as count LIMIT 20"
        )
        
        relationships = self.graph_db.query(
            "MATCH ()-[r]->() RETURN DISTINCT type(r) as type, count(*) as count LIMIT 20"
        )
        
        prompt = f"""
        Based on the current schema and data in the knowledge graph, suggest improvements.
        
        Current schema:
        {json.dumps(current_schema, indent=2)}
        
        Entity types in data:
        {json.dumps(entities, indent=2)}
        
        Relationship types in data:
        {json.dumps(relationships, indent=2)}
        
        Suggest:
        1. New entity types that would improve the schema
        2. New relationship types that would improve the schema
        3. New properties for existing entity types
        
        Return the suggestions in this JSON format:
        {{
            "new_entity_types": [
                {{"name": "EntityType", "description": "Description", "properties": ["prop1", "prop2"]}}
            ],
            "new_relation_types": [
                {{"name": "RELATION_TYPE", "description": "Description", "source_types": ["SourceType"], "target_types": ["TargetType"]}}
            ],
            "new_properties": [
                {{"entity_type": "EntityType", "name": "propertyName", "description": "Description"}}
            ]
        }}
        """
        
        try:
            response = self.llm.predict(prompt)
            
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response.replace('\n', ''), re.DOTALL)
            if json_match:
                suggestions = json.loads(json_match.group(0))
                return suggestions
            return {}
        except Exception as e:
            logger.warning(f"Error generating schema suggestions: {e}")
            return {}