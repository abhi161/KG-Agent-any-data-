# unified_kg/core/kg_builder.py
from typing import Dict, List, Any, Optional, Tuple
import logging
import pandas as pd
import json
from datetime import datetime

from .entity_resolution import EntityResolution
from .data_processor import DataProcessor
from .schema_manager import SchemaManager

logger = logging.getLogger(__name__)

class LLMEnhancedKnowledgeGraph:
    """
    Main class for building a unified knowledge graph from structured and unstructured data
    """
    @staticmethod
    def sanitize_label(label):
        """Convert entity type names to valid Neo4j labels (no spaces)"""
        return label.replace(' ', '_')
        
    def __init__(self, llm, graph_db, initial_schema: Optional[Dict[str, Any]] = None, 
                 config: Optional[Dict[str, Any]] = None):
        self.llm = llm
        self.graph_db = graph_db
        self.config = config or {}
        
        # Initialize components
        self.entity_resolution = EntityResolution(llm, graph_db)
        self.data_processor = DataProcessor(
            llm, 
            chunk_size=self.config.get("chunk_size", 3000),
            chunk_overlap=self.config.get("chunk_overlap", 200)
        )
        self.schema_manager = SchemaManager(llm, graph_db, initial_schema)
        
        # Initialize schema
        self.schema_manager.initialize_schema()
        
        # Statistics
        self.stats = {
            "csv_files_processed": 0,
            "pdf_files_processed": 0,
            "entities_created": 0,
            "relationships_created": 0,
            "start_time": datetime.now().isoformat()
        }
    
    def process_csv_file(self, file_path: str, column_mappings: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Process a CSV file and add it to the knowledge graph
        
        Args:
            file_path: Path to the CSV file
            column_mappings: Optional mapping of column names to entity types
            
        Returns:
            Dict with processing statistics
        """
        logger.info(f"Processing CSV file: {file_path}")
        
        # Process the CSV file
        df, metadata = self.data_processor.process_csv(file_path)
        
        # Use provided column mappings or inferred ones
        if column_mappings:
            metadata["column_mappings"] = column_mappings
        
        # Process entities from columns
        entities_by_row = self._process_csv_entities(df, metadata["column_mappings"], file_path)
        
        # Process relationships between entities in same row
        self._process_csv_relationships(entities_by_row, file_path)
        
        # Update statistics
        self.stats["csv_files_processed"] += 1
        
        return {
            "file_path": file_path,
            "entities_created": len(entities_by_row),
            "column_mappings": metadata["column_mappings"]
        }
    
    def _process_csv_entities(self, df: pd.DataFrame, column_mappings: Dict[str, str], 
                              source: str) -> List[Dict[str, Any]]:
        """
        Process entities from a CSV DataFrame
        
        Args:
            df: DataFrame to process
            column_mappings: Mapping of column names to entity types
            source: Source identifier (file path)
            
        Returns:
            List of dictionaries with entities by row
        """
        entities_by_row = []
        batch_size = self.config.get("batch_size", 100)
        
        # Process in batches
        for start_idx in range(0, len(df), batch_size):
            end_idx = min(start_idx + batch_size, len(df))
            batch_df = df.iloc[start_idx:end_idx]
            
            batch_entities = []
            
            # Process each row
            for _, row in batch_df.iterrows():
                row_entities = {}
                
                # Process each mapped column
                for column, entity_type in column_mappings.items():
                    if column in df.columns and pd.notna(row[column]) and row[column]:
                        # Extract the entity from the cell
                        entity_name = str(row[column])
                        
                        # Create properties from the row
                        properties = {
                            "source": source,
                            "row_index": _
                        }
                        
                        # Add additional properties from the row
                        for prop_col in df.columns:
                            if prop_col != column and pd.notna(row[prop_col]) and row[prop_col]:
                                # Clean property name
                                prop_name = prop_col.replace(" ", "_").lower()
                                properties[prop_name] = row[prop_col]
                        
                        # Process entity
                        entity_id = self._add_or_update_entity(entity_name, entity_type, properties, source)
                        
                        row_entities[column] = {
                            "id": entity_id,
                            "name": entity_name,
                            "type": entity_type
                        }
                
                batch_entities.append(row_entities)
            
            entities_by_row.extend(batch_entities)
            
        return entities_by_row
    
    def _add_or_update_entity(self, name: str, entity_type: str, properties: Dict[str, Any], 
                              source: str) -> str:
        """
        Add a new entity or update an existing one
        
        Args:
            name: Entity name
            entity_type: Entity type
            properties: Entity properties
            source: Data source
            
        Returns:
            Entity ID
        """
        entity_type = self.sanitize_label(entity_type) 
        # Make sure the entity type is in the schema
        for et in self.schema_manager.schema.get("entity_types", []):
            if et["name"].lower() == entity_type.lower():
                break
        else:
            # Discover new entity type
            self.schema_manager.discover_entity_type(entity_type, 0.8, source)
            
            # Auto-approve high confidence entity types
            self.schema_manager.process_pending_changes()
        
        # Try to find a matching entity
        match = self.entity_resolution.find_matching_entity(name, entity_type, properties)
        
        if match:
            # Update existing entity
            self.entity_resolution.merge_entity_properties(match["id"], properties, source)
            return match["id"]
        else:
            # Create a new entity
            # Generate a unique ID
            import hashlib
            unique_id = hashlib.md5(f"{entity_type}:{name}".encode()).hexdigest()
            
            # Create entity
            query = f"""
            CREATE (e:{entity_type} {{id: $id, name: $name}})
            RETURN elementId(e) as id
            """
            
            params = {
                "id": unique_id,
                "name": name
            }
            
            result = self.graph_db.query(query, params)
            entity_id = result[0]["id"]
            
            # Add properties
            if properties:
                property_strings = [f"e.{k} = ${k}" for k in properties.keys()]
                set_clause = ", ".join(property_strings)
                
                query = f"""
                MATCH (e) WHERE elementId(e) = $id
                SET {set_clause}
                """
                
                params = {"id": entity_id, **properties}
                self.graph_db.query(query, params)
            
            # Update statistics
            self.stats["entities_created"] += 1
            
            return entity_id
    
    def _process_csv_relationships(self, entities_by_row: List[Dict[str, Any]], source: str) -> None:
        """
        Process relationships between entities in CSV rows
        
        Args:
            entities_by_row: List of dictionaries with entities by row
            source: Source identifier (file path)
        """
        # Process each row
        for row_entities in entities_by_row:
            entities = list(row_entities.values())
            
            # For each pair of entities in the row
            for i in range(len(entities)):
                for j in range(i+1, len(entities)):
                    # Determine relationship type
                    relation_type = self._infer_relationship_type(
                        entities[i]["type"], 
                        entities[j]["type"]
                    )
                    
                    # Create relationship
                    query = f"""
                    MATCH (a), (b) 
                    WHERE elementId(a) = $id1 AND elementId(b) = $id2
                    MERGE (a)-[r:{relation_type}]->(b)
                    SET r.source = $source
                    """
                    
                    params = {
                        "id1": entities[i]["id"],
                        "id2": entities[j]["id"],
                        "source": source
                    }
                    
                    self.graph_db.query(query, params)
                    
                    # Update statistics
                    self.stats["relationships_created"] += 1
    
    def _infer_relationship_type(self, type1: str, type2: str) -> str:
        """
        Infer relationship type between two entity types
        
        Args:
            type1: First entity type
            type2: Second entity type
            
        Returns:
            Inferred relationship type
        """
        # Check if we have a known relationship
        known_relations = self.graph_db.query(
            """
            MATCH (a)-[r]->(b)
            WHERE $type1 in labels(a) AND $type2 in labels(b)
            RETURN type(r) as relation, count(*) as count
            ORDER BY count DESC
            LIMIT 1
            """,
            {"type1": type1, "type2": type2}
        )
        
        if known_relations:
            return known_relations[0]["relation"]
        
        # Generate relationship type
        prompt = f"""
        Generate an appropriate relationship type between two entity types in a knowledge graph.
        
        Source entity type: {type1}
        Target entity type: {type2}
        
        The relationship type should:
        1. Use UPPER_SNAKE_CASE format
        2. Be a verb or verb phrase
        3. Clearly describe how the source entity relates to the target entity
        
        Return just the relationship type without any explanation.
        """
        
        try:
            relation_type = self.llm.predict(prompt).strip()
            
            # Clean up the relation type
            relation_type = relation_type.upper().replace(" ", "_")
            
            # Register the relation type
            self.schema_manager.discover_relation_type(relation_type, 0.7, f"inferred:{type1}->{type2}")
            
            return relation_type
        except Exception as e:
            logger.warning(f"Error inferring relationship type: {e}")
            return "RELATED_TO"
    
    def process_pdf_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process a PDF file and add it to the knowledge graph
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dict with processing statistics
        """
        logger.info(f"Processing PDF file: {file_path}")
        
        # Process the PDF file
        chunks, metadata = self.data_processor.process_pdf(file_path)
        
        # Process each chunk
        entities_by_chunk = []
        relations_by_chunk = []
        
        for chunk_idx, chunk in enumerate(chunks):
            # Extract entities
            entities = self.data_processor.extract_entities_from_text(chunk)
            
            # Process entities
            chunk_entities = self._process_pdf_entities(entities, file_path, chunk_idx)
            entities_by_chunk.append(chunk_entities)
            
            # Extract relationships
            if len(entities) > 1:
                relations = self.data_processor.extract_relations_from_text(chunk, entities)
                
                # Process relationships
                chunk_relations = self._process_pdf_relations(relations, chunk_entities, file_path, chunk_idx)
                relations_by_chunk.append(chunk_relations)
        
        # Cross-reference entities across chunks
        self._cross_reference_pdf_entities(entities_by_chunk, file_path)
        
        # Update statistics
        self.stats["pdf_files_processed"] += 1
        
        return {
            "file_path": file_path,
            "chunks_processed": len(chunks),
            "entities_extracted": sum(len(chunk) for chunk in entities_by_chunk),
            "relationships_extracted": sum(len(chunk) for chunk in relations_by_chunk)
        }
    
    def _process_pdf_entities(self, entities: List[Dict[str, Any]], source: str, 
                         chunk_idx: int) -> Dict[str, Dict[str, Any]]:
        """Process entities from a PDF chunk"""
        chunk_entities = {}
        
        for entity in entities:
            # Create properties
            properties = {
                "source": source,
                "chunk_index": chunk_idx
            }
            
            # Add attributes - handle complex types properly
            if "attributes" in entity:
                for key, value in entity["attributes"].items():
                    # Clean property name
                    prop_name = key.replace(" ", "_").lower()
                    
                    # Handle complex types (like dictionaries)
                    if isinstance(value, dict):
                        # Serialize as JSON string
                        properties[f"{prop_name}_json"] = json.dumps(value)
                        
                        # Flatten into separate properties for better querying
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, (str, int, float, bool)):
                                sub_prop_name = f"{prop_name}_{sub_key.replace(' ', '_').lower()}"
                                properties[sub_prop_name] = sub_value
                    elif isinstance(value, list):
                        # Lists might contain dictionaries - handle carefully
                        if value and isinstance(value[0], dict):
                            # List of dictionaries - serialize as JSON
                            properties[f"{prop_name}_json"] = json.dumps(value)
                        else:
                            # Simple list - store directly
                            properties[prop_name] = value
                    else:
                        # For primitive types, use directly
                        properties[prop_name] = value
            
            # Process entity
            entity_id = self._add_or_update_entity(
                entity["name"],
                entity["type"],
                properties,
                source
            )
            
            chunk_entities[entity["name"]] = {
                "id": entity_id,
                "name": entity["name"],
                "type": entity["type"]
            }
        
        return chunk_entities

    def _process_pdf_relations(self, relations: List[Dict[str, Any]], 
                               chunk_entities: Dict[str, Dict[str, Any]], 
                               source: str, chunk_idx: int) -> List[Dict[str, Any]]:
        """
        Process relationships from a PDF chunk
        
        Args:
            relations: List of extracted relationships
            chunk_entities: Dictionary of entities in the chunk
            source: Source identifier (file path)
            chunk_idx: Chunk index
            
        Returns:
            List of processed relationships
        """
        processed_relations = []
        
        for relation in relations:
            # Find source and target entities
            source_entity = chunk_entities.get(relation["source"])
            target_entity = chunk_entities.get(relation["target"])
            
            if not source_entity or not target_entity:
                continue
            
            # Clean up relation type
            relation_type = relation["relation"].upper().replace(" ", "_")
            
            # Register relation type
            self.schema_manager.discover_relation_type(
                relation_type, 
                0.7, 
                f"{source}:chunk{chunk_idx}"
            )
            
            # Create relationship
            query = f"""
            MATCH (a), (b) 
            WHERE elementId(a) = $id1 AND elementId(b) = $id2
            MERGE (a)-[r:{relation_type}]->(b)
            SET r.source = $source,
                r.chunk_index = $chunk_idx
            """
            
            params = {
                "id1": source_entity["id"],
                "id2": target_entity["id"],
                "source": source,
                "chunk_idx": chunk_idx
            }
            
            self.graph_db.query(query, params)
            
            # Update statistics
            self.stats["relationships_created"] += 1
            
            # Add to processed relations
            processed_relations.append({
                "source": relation["source"],
                "target": relation["target"],
                "relation": relation_type
            })
        
        return processed_relations
    
    def _cross_reference_pdf_entities(self, entities_by_chunk: List[Dict[str, Dict[str, Any]]], 
                                      source: str) -> None:
        """
        Cross-reference entities across chunks in the same PDF
        
        Args:
            entities_by_chunk: List of dictionaries with entities by chunk
            source: Source identifier (file path)
        """
        # Flatten entities
        all_entities = {}
        for chunk_entities in entities_by_chunk:
            for name, entity in chunk_entities.items():
                if name not in all_entities:
                    all_entities[name] = entity
        
        # Find co-occurring entities
        for chunk_idx, chunk_entities in enumerate(entities_by_chunk):
            # Skip chunks with less than 2 entities
            if len(chunk_entities) < 2:
                continue
                
            # For each pair of entities in the chunk
            entity_list = list(chunk_entities.values())
            for i in range(len(entity_list)):
                for j in range(i+1, len(entity_list)):
                    # Create co-occurrence relationship
                    query = """
                    MATCH (a), (b) 
                    WHERE elementId(a) = $id1 AND elementId(b) = $id2
                    MERGE (a)-[r:OCCURS_WITH]->(b)
                    ON CREATE SET r.count = 1, r.source = $source
                    ON MATCH SET r.count = r.count + 1
                    """
                    
                    params = {
                        "id1": entity_list[i]["id"],
                        "id2": entity_list[j]["id"],
                        "source": source
                    }
                    
                    self.graph_db.query(query, params)
                    
                    # Update statistics
                    self.stats["relationships_created"] += 1
    
    def cross_reference_data_sources(self) -> Dict[str, Any]:
        """
        Cross-reference entities between different data sources
        
        Returns:
            Dictionary with cross-referencing statistics
        """
        logger.info("Cross-referencing data sources")
        
        cross_ref_stats = {
            "entity_matches": 0,
            "new_relationships": 0
        }
        
        # Find entities that appear in both structured and unstructured data
        bridge_entities = self.graph_db.query(
            """
            MATCH (n)
            WHERE any(source in [n.source] WHERE source CONTAINS '.csv')
            AND any(source in [n.source] WHERE source CONTAINS '.pdf')
            RETURN n.name as name, labels(n)[0] as type, elementId(n) as id
            """
        )
        
        for bridge in bridge_entities:
            # Get CSV connections
            csv_connections = self.graph_db.query(
                """
                MATCH (bridge)-[r1]-(csv_entity)
                WHERE elementId(bridge) = $bridge_id
                AND any(source in [r1.source] WHERE source CONTAINS '.csv')
                RETURN csv_entity.name as name, labels(csv_entity)[0] as type, elementId(csv_entity) as id
                """,
                {"bridge_id": bridge["id"]}
            )
            
            # Get PDF connections
            pdf_connections = self.graph_db.query(
                """
                MATCH (bridge)-[r2]-(pdf_entity)
                WHERE elementId(bridge) = $bridge_id
                AND any(source in [r2.source] WHERE source CONTAINS '.pdf')
                RETURN pdf_entity.name as name, labels(pdf_entity)[0] as type, elementId(pdf_entity) as id
                """,
                {"bridge_id": bridge["id"]}
            )
            
            # Create potential new connections
            for csv_entity in csv_connections:
                for pdf_entity in pdf_connections:
                    # Skip if they're the same entity
                    if csv_entity["id"] == pdf_entity["id"]:
                        continue
                    
                    # Skip if already connected
                    existing = self.graph_db.query(
                        """
                        MATCH (a)-[r]-(b)
                        WHERE elementId(a) = $id1 AND elementId(b) = $id2
                        RETURN count(r) as count
                        """,
                        {"id1": csv_entity["id"], "id2": pdf_entity["id"]}
                    )
                    
                    if existing[0]["count"] > 0:
                        continue
                    
                    # Use LLM to evaluate connection
                    relation_type = self._evaluate_cross_source_connection(
                        csv_entity, pdf_entity, bridge
                    )
                    
                    if relation_type:
                        # Create relationship
                        query = f"""
                        MATCH (a), (b) 
                        WHERE elementId(a) = $id1 AND elementId(b) = $id2
                        MERGE (a)-[r:{relation_type}]->(b)
                        SET r.source = 'cross_reference',
                            r.confidence = 0.7,
                            r.bridge_entity = $bridge_name
                        """
                        
                        params = {
                            "id1": csv_entity["id"],
                            "id2": pdf_entity["id"],
                            "bridge_name": bridge["name"]
                        }
                        
                        self.graph_db.query(query, params)
                        
                        # Update statistics
                        cross_ref_stats["new_relationships"] += 1
        
        return cross_ref_stats
    
    def _evaluate_cross_source_connection(self, entity1: Dict[str, Any], entity2: Dict[str, Any], 
                                          bridge: Dict[str, Any]) -> Optional[str]:
        """
        Evaluate if two entities from different sources should be connected
        
        Args:
            entity1: First entity
            entity2: Second entity
            bridge: Bridge entity connecting both
            
        Returns:
            Relationship type if they should be connected, None otherwise
        """
        prompt = f"""
        Evaluate if these two entities should be connected in our knowledge graph:
        
        Entity 1: {entity1["name"]} (Type: {entity1["type"]})
        Entity 2: {entity2["name"]} (Type: {entity2["type"]})
        
        Both entities are connected to: {bridge["name"]} (Type: {bridge["type"]})
        
        Should these entities have a direct relationship? If yes, what type of relationship?
        Answer with one of these formats:
        - "NO" if they should not be connected
        - "<RELATIONSHIP_TYPE>" if they should be connected (e.g., "WORKS_FOR", "PART_OF")
        
        Be specific and use UPPER_SNAKE_CASE for the relationship type.
        """
        
        try:
            response = self.llm.predict(prompt).strip().upper()
            
            if response == "NO":
                return None
                
            # Clean up relation type
            relation_type = response.replace(" ", "_")
            
            # Register relation type
            self.schema_manager.discover_relation_type(
                relation_type, 
                0.7, 
                f"cross_reference:{entity1['type']}->{entity2['type']}"
            )
            
            return relation_type
        except Exception as e:
            logger.warning(f"Error evaluating cross-source connection: {e}")
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        self.stats["end_time"] = datetime.now().isoformat()
        self.stats["entity_resolution"] = self.entity_resolution.get_stats()
        self.stats["pending_schema_changes"] = {
            "entity_types": len(self.schema_manager.pending_changes["entity_types"]),
            "relation_types": len(self.schema_manager.pending_changes["relation_types"])
        }
        
        return self.stats