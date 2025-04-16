#!/usr/bin/env python3
"""
Knowledge Graph Inspector Script

This script connects to a Neo4j database and runs various queries to inspect
the structure and contents of a knowledge graph created with the unified_kg system.
It provides information about entity types, relationships, vector indexes, and more.
"""

import os
import json
import argparse
from datetime import datetime
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from neo4j import GraphDatabase

# ANSI color codes for prettier output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class KnowledgeGraphInspector:
    """
    Inspector for analyzing the Neo4j knowledge graph structure and contents
    """
    
    def __init__(self, uri: str, username: str, password: str):
        """Initialize with Neo4j connection parameters"""
        self.uri = uri
        self.username = username
        self.password = password
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
    
    def close(self):
        """Close the Neo4j driver connection"""
        self.driver.close()
    
    def _execute_query(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute a Cypher query and return the results"""
        with self.driver.session() as session:
            result = session.run(query, params or {})
            return [record.data() for record in result]
    
    def check_database_info(self) -> Dict[str, Any]:
        """Get basic database information"""
        info = {}
        
        # Check Neo4j version
        try:
            query = "CALL dbms.components() YIELD name, versions, edition RETURN name, versions, edition"
            results = self._execute_query(query)
            if results:
                info["neo4j_version"] = results[0]["versions"][0]
                info["neo4j_edition"] = results[0]["edition"]
        except Exception as e:
            info["error"] = f"Could not get Neo4j version: {e}"
        
        # Check for GDS plugin
        try:
            query = "RETURN gds.version() AS version"
            results = self._execute_query(query)
            if results:
                info["gds_version"] = results[0]["version"]
        except Exception:
            info["gds_version"] = "Not installed"
        
        # Check for APOC plugin
        try:
            query = "RETURN apoc.version() AS version"
            results = self._execute_query(query)
            if results:
                info["apoc_version"] = results[0]["version"]
        except Exception:
            info["apoc_version"] = "Not installed"
        
        return info
    
    def check_vector_indexes(self) -> List[Dict[str, Any]]:
        """Check for vector indexes in the database"""
        try:
            query = """
            SHOW INDEXES
            YIELD name, type, labelsOrTypes, properties
            WHERE type = 'VECTOR'
            RETURN name, labelsOrTypes, properties
            """
            return self._execute_query(query)
        except Exception:
            # Fallback for older Neo4j versions
            try:
                query = """
                CALL db.indexes()
                YIELD name, type, labelsOrTypes, properties
                WHERE type = 'VECTOR'
                RETURN name, labelsOrTypes, properties
                """
                return self._execute_query(query)
            except Exception:
                return [{"name": "Error", "message": "Could not retrieve vector indexes"}]
    
    def get_entity_types(self) -> List[Dict[str, Any]]:
        """Get all entity types (labels) with counts"""
        query = """
        MATCH (n)
        WITH labels(n) AS labels, count(n) AS count
        UNWIND labels AS label
        RETURN label, sum(count) AS count
        ORDER BY count DESC
        """
        return self._execute_query(query)
    
    def get_relationship_types(self) -> List[Dict[str, Any]]:
        """Get all relationship types with counts"""
        query = """
        MATCH ()-[r]->()
        RETURN type(r) AS type, count(r) AS count
        ORDER BY count DESC
        """
        return self._execute_query(query)
    
    def get_entity_properties(self) -> List[Dict[str, Any]]:
        """Get property statistics for entities"""
        query = """
        MATCH (n)
        UNWIND keys(n) AS property
        RETURN property, count(n) AS count, labels(n)[0] AS common_label
        ORDER BY count DESC
        LIMIT 20
        """
        return self._execute_query(query)
    
    def check_embeddings(self) -> Dict[str, Any]:
        """Check embedding statistics"""
        results = {}
        
        # Count entities with embeddings
        query = """
        MATCH (n)
        WHERE n.embedding IS NOT NULL
        RETURN labels(n)[0] AS type, count(n) AS count
        """
        results["entities_with_embeddings"] = self._execute_query(query)
        
        # Count entities without embeddings
        query = """
        MATCH (n)
        WHERE n.embedding IS NULL
        RETURN labels(n)[0] AS type, count(n) AS count
        """
        results["entities_without_embeddings"] = self._execute_query(query)
        
        # Check embedding dimensions
        if results["entities_with_embeddings"]:
            query = """
            MATCH (n)
            WHERE n.embedding IS NOT NULL
            WITH n LIMIT 1
            RETURN size(n.embedding) AS dimension
            """
            dimension_results = self._execute_query(query)
            if dimension_results:
                results["embedding_dimension"] = dimension_results[0]["dimension"]
        
        return results
    
    def check_cross_source_connections(self) -> Dict[str, Any]:
        """Check connections between CSV and PDF sources"""
        results = {}
        
        # Count entities by source
        query = """
        MATCH (n)
        WHERE n.source IS NOT NULL
        WITH n,
             CASE WHEN n.source CONTAINS '.csv' THEN 'csv' 
                  WHEN n.source CONTAINS '.pdf' THEN 'pdf' 
                  ELSE 'other' END AS source_type
        RETURN source_type, count(n) AS count
        """
        results["entities_by_source"] = self._execute_query(query)
        
        # Count bridge entities (appearing in both sources)
        query = """
        MATCH (n)
        WHERE n.source IS NOT NULL
        AND (n.source CONTAINS '.csv' AND n.source CONTAINS '.pdf'
             OR n.sources IS NOT NULL AND any(source in n.sources WHERE source CONTAINS '.csv') 
             AND any(source in n.sources WHERE source CONTAINS '.pdf'))
        RETURN labels(n)[0] AS type, count(n) AS count
        """
        results["bridge_entities"] = self._execute_query(query)
        
        # Count cross-source relationships
        query = """
        MATCH (a)-[r]->(b)
        WHERE (a.source CONTAINS '.csv' AND b.source CONTAINS '.pdf')
           OR (a.source CONTAINS '.pdf' AND b.source CONTAINS '.csv')
        RETURN type(r) AS relationship_type, count(r) AS count
        ORDER BY count DESC
        """
        results["cross_source_relationships"] = self._execute_query(query)
        
        return results
    
    def get_sample_entities(self, limit: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """Get sample entities for each entity type"""
        # Get all entity types
        entity_types = [record["label"] for record in self.get_entity_types()]
        
        samples = {}
        for entity_type in entity_types:
            query = f"""
            MATCH (n:{entity_type})
            RETURN n.name AS name, n.id AS id, elementId(n) AS neo4j_id, 
                   [key IN keys(n) WHERE key <> 'embedding'] AS properties
            LIMIT {limit}
            """
            samples[entity_type] = self._execute_query(query)
        
        return samples
    
    def get_sample_relationships(self, limit: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """Get sample relationships for each relationship type"""
        # Get all relationship types
        relationship_types = [record["type"] for record in self.get_relationship_types()]
        
        samples = {}
        for rel_type in relationship_types:
            query = f"""
            MATCH (a)-[r:{rel_type}]->(b)
            RETURN a.name AS source, b.name AS target, type(r) AS relationship,
                   [key IN keys(r)] AS properties
            LIMIT {limit}
            """
            samples[rel_type] = self._execute_query(query)
        
        return samples
    
    def get_entity_resolution_stats(self) -> Dict[str, Any]:
        """Check for entity resolution statistics based on sources and properties"""
        results = {}
        
        # Check entities with multiple sources
        query = """
        MATCH (n)
        WHERE n.sources IS NOT NULL AND size(n.sources) > 1
        RETURN labels(n)[0] AS type, count(n) AS count
        """
        results["entities_with_multiple_sources"] = self._execute_query(query)
        
        # Check semantically related entities
        query = """
        MATCH (a)-[r:SEMANTICALLY_SIMILAR_TO]->(b)
        RETURN labels(a)[0] AS source_type, labels(b)[0] AS target_type, count(r) AS count
        """
        results["semantic_relationships"] = self._execute_query(query)
        
        # Check cross-source similar entities
        query = """
        MATCH (a)-[r:CROSS_SOURCE_SIMILAR_TO]->(b)
        RETURN labels(a)[0] AS source_type, labels(b)[0] AS target_type, count(r) AS count
        """
        results["cross_source_relationships"] = self._execute_query(query)
        
        return results
    
    def execute_advanced_queries(self) -> Dict[str, Any]:
        """Run some advanced queries to demonstrate KG capabilities"""
        results = {}
        
        # Find entities connected across different sources
        query = """
        MATCH path = (csv_entity)-[r]-(pdf_entity)
        WHERE ANY(source IN [csv_entity.source] WHERE source CONTAINS '.csv')
        AND ANY(source IN [pdf_entity.source] WHERE source CONTAINS '.pdf')
        RETURN 
            labels(csv_entity)[0] AS csv_entity_type,
            csv_entity.name AS csv_entity_name,
            type(r) AS relationship,
            labels(pdf_entity)[0] AS pdf_entity_type,
            pdf_entity.name AS pdf_entity_name
        LIMIT 10
        """
        results["cross_source_paths"] = self._execute_query(query)
        
        # Example query with vector similarity
        try:
            query = """
            MATCH (n)
            WHERE n.embedding IS NOT NULL
            WITH n LIMIT 1
            CALL db.index.vector.queryNodes(
                'global_embedding_index',
                5,
                n.embedding
            ) YIELD node, score
            RETURN 
                n.name AS source_entity,
                labels(n)[0] AS source_type,
                node.name AS similar_entity,
                labels(node)[0] AS similar_type,
                score
            ORDER BY score DESC
            """
            results["vector_similarity_example"] = self._execute_query(query)
        except Exception as e:
            results["vector_similarity_example"] = [{"error": str(e)}]
        
        return results

def print_section(title: str):
    """Print a formatted section title"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD} {title} {Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.END}\n")

def print_subsection(title: str):
    """Print a formatted subsection title"""
    print(f"\n{Colors.CYAN}{Colors.BOLD}{title}{Colors.END}")
    print(f"{Colors.CYAN}{'-' * len(title)}{Colors.END}")

def print_table(data: List[Dict[str, Any]], headers: List[str] = None):
    """Print data as a simple table"""
    if not data:
        print(f"{Colors.YELLOW}[No data]{Colors.END}")
        return
    
    # If headers not provided, use keys from first record
    if not headers:
        headers = list(data[0].keys())
    
    # Calculate column widths
    col_widths = {header: max(len(str(header)), max([len(str(record.get(header, ""))) for record in data])) 
                 for header in headers}
    
    # Print headers
    header_row = " | ".join([f"{header:{col_widths[header]}}" for header in headers])
    print(f"{Colors.BOLD}{header_row}{Colors.END}")
    print("-" * (sum(col_widths.values()) + 3 * (len(headers) - 1)))
    
    # Print data rows
    for record in data:
        row = " | ".join([f"{str(record.get(header, '')):{col_widths[header]}}" for header in headers])
        print(row)

def print_json(data: Any):
    """Print data as formatted JSON"""
    print(json.dumps(data, indent=2))

def main():
    """Main function to run the inspection"""
    parser = argparse.ArgumentParser(description="Inspect a Neo4j Knowledge Graph")
    parser.add_argument("--uri", help="Neo4j URI", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument("--user", help="Neo4j username", default=os.getenv("NEO4J_USER", "neo4j"))
    parser.add_argument("--password", help="Neo4j password", default=os.getenv("NEO4J_PASSWORD", "password"))
    parser.add_argument("--output", help="Output results to JSON file")
    parser.add_argument("--csv-only", action="store_true", help="Only check CSV-derived entities")
    parser.add_argument("--pdf-only", action="store_true", help="Only check PDF-derived entities")
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Get results file path
    output_file = args.output or f"kg_inspection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Initialize inspector
    inspector = KnowledgeGraphInspector(args.uri, args.user, args.password)
    
    try:
        # Store all results in a dictionary
        all_results = {}
        
        # Basic database info
        print_section("Neo4j Database Information")
        db_info = inspector.check_database_info()
        all_results["database_info"] = db_info
        
        print(f"Neo4j Version: {Colors.GREEN}{db_info.get('neo4j_version', 'Unknown')}{Colors.END}")
        print(f"Edition: {Colors.GREEN}{db_info.get('neo4j_edition', 'Unknown')}{Colors.END}")
        print(f"GDS Plugin: {Colors.GREEN}{db_info.get('gds_version', 'Not installed')}{Colors.END}")
        print(f"APOC Plugin: {Colors.GREEN}{db_info.get('apoc_version', 'Not installed')}{Colors.END}")
        
        # Vector indexes
        print_section("Vector Indexes")
        vector_indexes = inspector.check_vector_indexes()
        all_results["vector_indexes"] = vector_indexes
        
        if vector_indexes and "message" not in vector_indexes[0]:
            print_table(vector_indexes)
        else:
            print(f"{Colors.YELLOW}No vector indexes found or could not retrieve them.{Colors.END}")
        
        # Entity types
        print_section("Entity Types")
        entity_types = inspector.get_entity_types()
        all_results["entity_types"] = entity_types
        
        print_table(entity_types, ["label", "count"])
        
        # Relationship types
        print_section("Relationship Types")
        relationship_types = inspector.get_relationship_types()
        all_results["relationship_types"] = relationship_types
        
        print_table(relationship_types, ["type", "count"])
        
        # Common entity properties
        print_section("Common Entity Properties")
        entity_properties = inspector.get_entity_properties()
        all_results["entity_properties"] = entity_properties
        
        print_table(entity_properties, ["property", "count", "common_label"])
        
        # Embedding stats
        print_section("Embedding Statistics")
        embedding_stats = inspector.check_embeddings()
        all_results["embedding_stats"] = embedding_stats
        
        print_subsection("Entities with Embeddings")
        print_table(embedding_stats["entities_with_embeddings"], ["type", "count"])
        
        print_subsection("Entities without Embeddings")
        print_table(embedding_stats["entities_without_embeddings"], ["type", "count"])
        
        if "embedding_dimension" in embedding_stats:
            print(f"\nEmbedding Dimension: {Colors.GREEN}{embedding_stats['embedding_dimension']}{Colors.END}")
        
        # Cross-source connections
        print_section("Cross-Source Connections")
        cross_source = inspector.check_cross_source_connections()
        all_results["cross_source_connections"] = cross_source
        
        print_subsection("Entities by Source")
        print_table(cross_source["entities_by_source"], ["source_type", "count"])
        
        print_subsection("Bridge Entities (in both CSV and PDF)")
        print_table(cross_source["bridge_entities"], ["type", "count"])
        
        print_subsection("Cross-Source Relationships")
        print_table(cross_source["cross_source_relationships"], ["relationship_type", "count"])
        
        # Entity resolution stats
        print_section("Entity Resolution Statistics")
        resolution_stats = inspector.get_entity_resolution_stats()
        all_results["entity_resolution"] = resolution_stats
        
        print_subsection("Entities with Multiple Sources")
        print_table(resolution_stats["entities_with_multiple_sources"], ["type", "count"])
        
        print_subsection("Semantic Similarity Relationships")
        print_table(resolution_stats["semantic_relationships"], ["source_type", "target_type", "count"])
        
        print_subsection("Cross-Source Similar Relationships")
        print_table(resolution_stats["cross_source_relationships"], ["source_type", "target_type", "count"])
        
        # Advanced queries
        print_section("Advanced Query Examples")
        advanced_queries = inspector.execute_advanced_queries()
        all_results["advanced_queries"] = advanced_queries
        
        print_subsection("Cross-Source Paths")
        print_table(advanced_queries["cross_source_paths"])
        
        print_subsection("Vector Similarity Example")
        if "error" not in advanced_queries["vector_similarity_example"][0]:
            print_table(advanced_queries["vector_similarity_example"])
        else:
            print(f"{Colors.YELLOW}Error executing vector similarity query: {advanced_queries['vector_similarity_example'][0]['error']}{Colors.END}")
        
        # Sample entities and relationships (optional detailed view)
        if args.csv_only or args.pdf_only:
            print_section("Sample Entities")
            samples = inspector.get_sample_entities()
            
            for entity_type, entities in samples.items():
                # Filter based on command line arguments
                if args.csv_only and not any(e.get("csv", False) for e in entities):
                    continue
                if args.pdf_only and not any(e.get("pdf", False) for e in entities):
                    continue
                
                print_subsection(f"{entity_type} Samples")
                print_table(entities)
        
        # Save results to file
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n{Colors.GREEN}Results saved to {output_file}{Colors.END}")
    
    finally:
        inspector.close()

if __name__ == "__main__":
    main()