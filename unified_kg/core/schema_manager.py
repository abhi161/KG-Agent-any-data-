import logging
import json
import os
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class SchemaManager:
    """
    Manages the knowledge graph schema, including discovery and persistence.
    """


    def __init__(self, llm, graph_db, initial_schema: Optional[Dict] = None, schema_file_path: Optional[str] = None):

        self.llm = llm 
        self.graph_db = graph_db
        self.schema = initial_schema or {"entity_types": [], "relation_types": []}
        self.pending_changes = {"entity_types": {}, "relation_types": {}} # Store discovered types temporarily
        self.schema_file_path = schema_file_path
        # self.schema = self._load_schema(initial_schema)

        # Store known types for quick lookup (use sanitized names)

        self._known_entity_types = {self._sanitize(et['name']) for et in self.schema.get('entity_types', []) if 'name' in et}
        self._known_relation_types = {self._sanitize(rt['name']) for rt in self.schema.get('relation_types', []) if 'name' in rt}

        logger.info(f"SchemaManager initialized with {len(self._known_entity_types)} entity types and {len(self._known_relation_types)} relation types.")

    # def _load_schema(self, initial_schema_obj: Optional[Dict]) -> Dict:
    #     """Loads schema from file if path provided, else uses initial object."""
    #     if self.schema_file_path and os.path.exists(self.schema_file_path):
    #         try:
    #             with open(self.schema_file_path, 'r') as f:
    #                 logger.info(f"Loading schema from file: {self.schema_file_path}")
    #                 return json.load(f)
    #         except (IOError, json.JSONDecodeError) as e:
    #             logger.error(f"Error loading schema file {self.schema_file_path}, using initial object/default. Error: {e}")
    #             return initial_schema_obj or {"entity_types": [], "relation_types": []}
    #     else:
    #         logger.info("No schema file path provided or file not found, using initial schema object.")
    #         return initial_schema_obj or {"entity_types": [], "relation_types": []}

    def _save_schema(self):
        """Saves the current schema back to the file if path is defined."""
        if not self.schema_file_path:
            logger.warning("Schema file path not set. Cannot persist schema updates.")
            return

        try:
            logger.info(f"Saving updated schema to: {self.schema_file_path}")
            # Ensure 'entity_types' and 'relation_types' keys exist
            if "entity_types" not in self.schema: self.schema["entity_types"] = []
            if "relation_types" not in self.schema: self.schema["relation_types"] = []

            with open(self.schema_file_path, 'w') as f:
                json.dump(self.schema, f, indent=2)
        except IOError as e:
            logger.error(f"Error saving schema file {self.schema_file_path}: {e}")

    def _generate_description(self, item_name: str, item_category: str = "entity type") -> str:

        """Uses LLM to generate a brief description (optional)."""

        if not self.llm: return f"A type of {item_category}."
        try:
             prompt = f"Provide a concise, one-sentence description for the following knowledge graph {item_category}: '{item_name}'. Focus on its typical role or definition in a general or medical context.\nDescription:"
             description = self.llm.invoke(prompt).content.strip()
             
             if description.endswith('.'): description = description[:-1]
             return description[:150] 
        except Exception as e:
             logger.warning(f"Failed to generate description for {item_name}: {e}")
             return f"Represents a {item_name}." # Fallback

    def get_current_schema_definition(self, format_for_prompt=True) -> Any:
         """Returns the current schema, potentially formatted for an LLM prompt."""
         current_schema = self.schema

         if format_for_prompt:
              # Create a simplified text representation for the LLM
              schema_str = "Available Entity Types:\n"
              for et in current_schema.get("entity_types", []):
                   schema_str += f"- {et.get('name')}: {et.get('description', 'No description')}\n"
              # if required add relation types if helpful for the specific prompt
              # schema_str += "\nAvailable Relation Types:\n"
              # for rt in current_schema.get("relation_types", []):
              #     schema_str += f"- {rt.get('name')}\n"
              return schema_str
         else:
              return current_schema


    def _sanitize(self, name: str) -> str:

        if not isinstance(name, str): return "Unknown"
        sanitized = ''.join(c if c.isalnum() else '_' for c in name)

        if not sanitized or not sanitized[0].isalpha(): sanitized = "Type_" + sanitized
        sanitized = '_'.join(filter(None, sanitized.split('_')))

        return sanitized.upper() if ' ' not in name else sanitized 



    def initialize_schema(self):
        """ Apply initial schema constraints or indexes in Neo4j. """
        logger.info("Applying schema constraints/indexes (basic implementation)...")

        logger.info("Schema initialization step complete.")
        pass
      

    def has_entity_type(self, entity_type: str) -> bool:
        """ Check if an entity type is known (case-insensitive check on sanitized name). """
        return self._sanitize(entity_type) in self._known_entity_types

    def has_relation_type(self, relation_type: str) -> bool:
        """ Check if a relation type is known (case-insensitive check on sanitized name). """
        return self._sanitize(relation_type) in self._known_relation_types


    def discover_entity_type(self, entity_type: str, confidence: float, source: str):

        """ Register a newly discovered entity type. """
        sanitized_type = self._sanitize(entity_type)
        if sanitized_type not in self._known_entity_types and sanitized_type not in self.pending_changes["entity_types"]:
            logger.info(f"Schema Discovery: New entity type '{entity_type}' (Sanitized: {sanitized_type}) found from {source} with confidence {confidence:.2f}")
            
            description = self._generate_description(entity_type, "entity type")
            self.pending_changes["entity_types"][sanitized_type] = {
                "original_name": entity_type,
                "description": description, 
                "confidence": confidence, 
                "sources": [source]
            }
            
        elif sanitized_type in self.pending_changes["entity_types"]:
             
             # Update confidence/sources if found again before approval
             self.pending_changes["entity_types"][sanitized_type]["confidence"] = max(confidence, self.pending_changes["entity_types"][sanitized_type]["confidence"])
             if source not in self.pending_changes["entity_types"][sanitized_type]["sources"]:
                  self.pending_changes["entity_types"][sanitized_type]["sources"].append(source)


    def discover_relation_type(self, relation_type: str, confidence: float, source: str):
        """ Register a newly discovered relationship type. """
        sanitized_type = self._sanitize(relation_type) # Usually uppercase for relations
        if sanitized_type not in self._known_relation_types and sanitized_type not in self.pending_changes["relation_types"]:
            logger.info(f"Schema Discovery: New relation type '{relation_type}' (Sanitized: {sanitized_type}) found from {source} with confidence {confidence:.2f}")
             # Generate description
            description = self._generate_description(relation_type, "relationship type")
            self.pending_changes["relation_types"][sanitized_type] = {
                "original_name": relation_type,
                "description": description, # Store description
                "confidence": confidence,
                "sources": [source]
            }
        elif sanitized_type in self.pending_changes["relation_types"]:
             self.pending_changes["relation_types"][sanitized_type]["confidence"] = max(confidence, self.pending_changes["relation_types"][sanitized_type]["confidence"])
             if source not in self.pending_changes["relation_types"][sanitized_type]["sources"]:
                  self.pending_changes["relation_types"][sanitized_type]["sources"].append(source)


    def process_pending_changes(self, approval_threshold=0.75):
        """ Process discovered types, add approved ones to schema, and save. """
        logger.info(f"Processing {len(self.pending_changes['entity_types'])} pending entity types and {len(self.pending_changes['relation_types'])} pending relation types...")
        schema_updated = False
         
        approved_entities = 0
        approved_relations = 0

        # Process entities
        for sanitized_type, details in list(self.pending_changes["entity_types"].items()):
            if details["confidence"] >= approval_threshold:
                logger.info(f"Approving and adding new entity type '{details['original_name']}' (Sanitized: {sanitized_type}) to schema.")
                self._known_entity_types.add(sanitized_type)
                # Add to self.schema structure
                if "entity_types" not in self.schema: self.schema["entity_types"] = []
                # Avoid adding duplicates if somehow already present
                if not any(et['name'] == details['original_name'] for et in self.schema["entity_types"]):
                    self.schema["entity_types"].append({
                        "name": details['original_name'],
                        "description": details['description'],
                        "properties": [] # Start with empty properties list for new types
                        # Keep identifier_property empty unless inferred later
                    })
                    schema_updated = True
                # Apply constraints/indexes if needed here (using sanitized_type)
                del self.pending_changes["entity_types"][sanitized_type] # Remove from pending
                approved_entities += 1
            else:
                 logger.debug(f"Entity type '{details['original_name']}' confidence {details['confidence']:.2f} below threshold {approval_threshold}. Remains pending.")


        # Process relations
        for sanitized_type, details in list(self.pending_changes["relation_types"].items()):
            if details["confidence"] >= approval_threshold:
                logger.info(f"Approving and adding new relation type '{details['original_name']}' (Sanitized: {sanitized_type}) to schema.")
                self._known_relation_types.add(sanitized_type)
                if "relation_types" not in self.schema: self.schema["relation_types"] = []
                if not any(rt['name'] == details['original_name'] for rt in self.schema["relation_types"]):
                    self.schema["relation_types"].append({
                        "name": details['original_name'],
                        "description": details['description'],
                        "source_types": [], # Start empty, maybe infer later?
                        "target_types": []
                    })
                    schema_updated = True
                del self.pending_changes["relation_types"][sanitized_type]
                approved_relations += 1
            else:
                 logger.debug(f"Relation type '{details['original_name']}' confidence {details['confidence']:.2f} below threshold {approval_threshold}. Remains pending.")

        # Save the schema if it was updated
        if schema_updated:
            self._save_schema()

        logger.info(f"Schema processing complete. Approved {approved_entities} entity types, {approved_relations} relation types.")

    def get_pending_changes_count(self) -> Dict[str, int]:
         """ Returns the count of pending schema changes. """
         return {
             "pending_entity_types": len(self.pending_changes["entity_types"]),
             "pending_relation_types": len(self.pending_changes["relation_types"]),
         }