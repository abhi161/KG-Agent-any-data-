import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class SchemaManager:
    """
    Manages the knowledge graph schema, including discovery and persistence.
    (This is a basic placeholder - needs full implementation)
    """
    def __init__(self, llm, graph_db, initial_schema: Optional[Dict] = None):
        self.llm = llm # May be needed for schema refinement/validation
        self.graph_db = graph_db
        self.schema = initial_schema or {"entity_types": [], "relation_types": []}
        self.pending_changes = {"entity_types": {}, "relation_types": {}} # Store discovered types temporarily

        # Store known types for quick lookup (use sanitized names)
        self._known_entity_types = {self._sanitize(et['name']) for et in self.schema.get('entity_types', []) if 'name' in et}
        self._known_relation_types = {self._sanitize(rt['name']) for rt in self.schema.get('relation_types', []) if 'name' in rt}
        logger.info(f"SchemaManager initialized with {len(self._known_entity_types)} entity types and {len(self._known_relation_types)} relation types.")


    def _sanitize(self, name: str) -> str:
        """ Consistent sanitization for internal use. """
        if not isinstance(name, str): return "Unknown"
        sanitized = ''.join(c if c.isalnum() else '_' for c in name)
        if not sanitized or not sanitized[0].isalpha(): sanitized = "Type_" + sanitized
        sanitized = '_'.join(filter(None, sanitized.split('_')))
        return sanitized.upper() if ' ' not in name else sanitized # Uppercase for relations likely

    def initialize_schema(self):
        """ Apply initial schema constraints or indexes in Neo4j. """
        logger.info("Applying schema constraints/indexes (basic implementation)...")
        # Example: Create constraints for unique IDs if defined in schema
        # for entity_type in self.schema.get("entity_types", []):
        #     type_name = self._sanitize(entity_type.get("name"))
        #     # Assuming an 'id' property for uniqueness - adjust as needed
        #     id_prop = "id" # Or "unique_hash" or configured primary key
        #     if type_name and "id" in entity_type.get("properties",[]):
        #         try:
        #             # Use specific ID property name if available
        #             query = f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:`{type_name}`) REQUIRE n.{id_prop} IS UNIQUE"
        #             self.graph_db.query(query)
        #             logger.debug(f"Applied unique constraint on {type_name}({id_prop})")
        #         except Exception as e:
        #              # Constraint might already exist, or other issues
        #              logger.warning(f"Could not apply constraint on {type_name}({id_prop}): {e}")
        logger.info("Schema initialization step complete.")
        # Actual implementation would involve creating indexes on properties etc.

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
            self.pending_changes["entity_types"][sanitized_type] = {
                "original_name": entity_type,
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
            self.pending_changes["relation_types"][sanitized_type] = {
                "original_name": relation_type,
                "confidence": confidence,
                "sources": [source]
            }
        elif sanitized_type in self.pending_changes["relation_types"]:
             self.pending_changes["relation_types"][sanitized_type]["confidence"] = max(confidence, self.pending_changes["relation_types"][sanitized_type]["confidence"])
             if source not in self.pending_changes["relation_types"][sanitized_type]["sources"]:
                  self.pending_changes["relation_types"][sanitized_type]["sources"].append(source)


    def process_pending_changes(self, approval_threshold=0.85):
        """ Process discovered types, potentially adding them to the schema based on confidence. """
        logger.info(f"Processing {len(self.pending_changes['entity_types'])} pending entity types and {len(self.pending_changes['relation_types'])} pending relation types...")
        approved_entities = 0
        approved_relations = 0

        # Process entities
        for sanitized_type, details in list(self.pending_changes["entity_types"].items()): # Iterate over copy
            if details["confidence"] >= approval_threshold:
                logger.info(f"Approving new entity type '{details['original_name']}' (Sanitized: {sanitized_type})")
                self._known_entity_types.add(sanitized_type)
                # TODO: Add to self.schema structure and potentially update graph constraints/indexes
                # self.schema["entity_types"].append({"name": details['original_name'], "properties": []}) # Basic addition
                # Apply constraints/indexes if needed
                del self.pending_changes["entity_types"][sanitized_type] # Remove from pending
                approved_entities += 1
            else:
                 logger.debug(f"Entity type '{details['original_name']}' confidence {details['confidence']:.2f} below threshold {approval_threshold}. Remains pending.")


        # Process relations
        for sanitized_type, details in list(self.pending_changes["relation_types"].items()):
            if details["confidence"] >= approval_threshold:
                logger.info(f"Approving new relation type '{details['original_name']}' (Sanitized: {sanitized_type})")
                self._known_relation_types.add(sanitized_type)
                # TODO: Add to self.schema structure
                # self.schema["relation_types"].append({"name": details['original_name'], "source_types": [], "target_types": []})
                del self.pending_changes["relation_types"][sanitized_type]
                approved_relations += 1
            else:
                 logger.debug(f"Relation type '{details['original_name']}' confidence {details['confidence']:.2f} below threshold {approval_threshold}. Remains pending.")

        logger.info(f"Schema processing complete. Approved {approved_entities} entity types, {approved_relations} relation types.")
        # Potentially save updated schema back to a file here

    def get_pending_changes_count(self) -> Dict[str, int]:
         """ Returns the count of pending schema changes. """
         return {
             "pending_entity_types": len(self.pending_changes["entity_types"]),
             "pending_relation_types": len(self.pending_changes["relation_types"]),
         }