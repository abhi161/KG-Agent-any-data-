import os
import logging
import argparse
import json
from typing import List, Dict, Any
import time
from dotenv import load_dotenv

# Updated imports for LangChain
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_neo4j import Neo4jGraph

from unified_kg.config import Config
from unified_kg.core.kg_builder import LLMEnhancedKnowledgeGraph

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def build_knowledge_graph(csv_files: List[str], pdf_files: List[str], 
                          initial_schema_path: str = None, config_path: str = None) -> Dict[str, Any]:
    """
    Build a knowledge graph from structured and unstructured data with vector embeddings
    
    Args:
        csv_files: List of CSV file paths
        pdf_files: List of PDF file paths
        initial_schema_path: Path to initial schema JSON file
        config_path: Path to configuration JSON file
        
    Returns:
        Dictionary with processing statistics
    """
    # Load environment variables
    load_dotenv()
    
    # Load configuration
    config = Config()
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_data = json.load(f)
            for key, value in config_data.items():
                setattr(config, key, value)
    
    # Load initial schema
    initial_schema = None
    if initial_schema_path and os.path.exists(initial_schema_path):
        with open(initial_schema_path, 'r') as f:
            initial_schema = json.load(f)
    
    # Initialize LLM
    llm = config.get_llm()
    
    # Initialize Embeddings
    embeddings = config.get_embeddings()
    
    # Initialize Neo4j
    graph = Neo4jGraph(
        url=config.neo4j_uri,
        username=config.neo4j_user,
        password=config.neo4j_password
    )
    
    # Check Neo4j vector capabilities
    vector_support = check_neo4j_vector_capabilities(graph)
    if not vector_support["has_vector_support"]:
        logger.warning("Neo4j vector support not detected. Vector operations will be performed in-memory.")
        logger.warning(f"Reason: {vector_support['reason']}")
    else:
        logger.info(f"Neo4j vector support detected. Using {vector_support['details']} for vector operations.")
    
    # Initialize knowledge graph builder
    kg_builder = LLMEnhancedKnowledgeGraph(
        llm,
        graph,
        embeddings,  # Pass embeddings to the builder
        initial_schema=initial_schema,
        config={
            "batch_size": config.batch_size,
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
            "vector_enabled": vector_support["has_vector_support"]
        }
    )
    
    try:
        # Process CSV files
        csv_results = []
        for csv_file in csv_files:
            if os.path.exists(csv_file):
                try:
                    result = kg_builder.process_csv_file(csv_file)
                    csv_results.append(result)
                    logger.info(f"Processed CSV file: {csv_file}")
                except Exception as e:
                    logger.error(f"Error processing CSV file {csv_file}: {e}", exc_info=True)
        
        # Process PDF files
        pdf_results = []
        for pdf_file in pdf_files:
            if os.path.exists(pdf_file):
                try:
                    result = kg_builder.process_pdf_file(pdf_file)
                    pdf_results.append(result)
                    logger.info(f"Processed PDF file: {pdf_file}")
                except Exception as e:
                    logger.error(f"Error processing PDF file {pdf_file}: {e}", exc_info=True)
        
        # Cross-reference data sources
        if csv_results and pdf_results:
            cross_ref_results = kg_builder.cross_reference_data_sources()
            logger.info(f"Cross-referenced data sources: {cross_ref_results}")
        
        # Process schema changes
        kg_builder.schema_manager.process_pending_changes()
        
        # Get statistics
        stats = kg_builder.get_statistics()
        
        return stats
    
    except Exception as e:
        logger.error(f"Error building knowledge graph: {e}", exc_info=True)
        return {"error": str(e)}

def check_neo4j_vector_capabilities(graph: Neo4jGraph) -> Dict[str, Any]:
    """
    Check if Neo4j has vector capabilities enabled
    
    Args:
        graph: Neo4j graph connection
        
    Returns:
        Dictionary with vector support information
    """
    try:
        # Check for GDS plugin
        gds_query = "CALL gds.list() YIELD name RETURN count(*) > 0 as has_gds"
        gds_result = graph.query(gds_query)
        has_gds = gds_result[0]["has_gds"] if gds_result else False
        
        # Check for vector index capabilities
        vector_query = "CALL db.indexes() YIELD name, type WHERE type = 'VECTOR' RETURN count(*) > 0 as has_vector_indexes"
        
        try:
            vector_result = graph.query(vector_query)
            has_vector_indexes = vector_result[0]["has_vector_indexes"] if vector_result else False
        except Exception:
            has_vector_indexes = False
        
        # Check for APOC
        apoc_query = "CALL apoc.help('text') YIELD name RETURN count(*) > 0 as has_apoc"
        apoc_result = graph.query(apoc_query)
        has_apoc = apoc_result[0]["has_apoc"] if apoc_result else False
        
        # Determine overall status
        if has_vector_indexes:
            return {
                "has_vector_support": True,
                "details": "Neo4j Vector Index",
                "capabilities": {
                    "vector_indexes": has_vector_indexes,
                    "gds": has_gds,
                    "apoc": has_apoc
                }
            }
        elif has_gds:
            return {
                "has_vector_support": True,
                "details": "Graph Data Science library",
                "capabilities": {
                    "vector_indexes": has_vector_indexes,
                    "gds": has_gds,
                    "apoc": has_apoc
                }
            }
        else:
            return {
                "has_vector_support": False,
                "reason": "Neither Vector Indexes nor GDS library found",
                "capabilities": {
                    "vector_indexes": has_vector_indexes,
                    "gds": has_gds,
                    "apoc": has_apoc
                }
            }
    
    except Exception as e:
        return {
            "has_vector_support": False,
            "reason": f"Error checking capabilities: {e}",
            "capabilities": {
                "vector_indexes": False,
                "gds": False,
                "apoc": False
            }
        }

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Build a unified knowledge graph with vector embeddings')
    parser.add_argument('--csv', nargs='+', help='CSV files to process')
    parser.add_argument('--pdf', nargs='+', help='PDF files to process')
    parser.add_argument('--schema', help='Path to initial schema JSON file')
    parser.add_argument('--config', help='Path to configuration JSON file')
    parser.add_argument('--output', help='Path to output statistics JSON file')
    parser.add_argument('--vector', action='store_true', help='Enable vector embeddings (default if Azure OpenAI embeddings are configured)')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Build knowledge graph
    stats = build_knowledge_graph(
        csv_files=args.csv or [],
        pdf_files=args.pdf or [],
        initial_schema_path=args.schema,
        config_path=args.config
    )
    
    end_time = time.time()
    stats["processing_time_seconds"] = end_time - start_time
    
    # Print statistics
    print(json.dumps(stats, indent=2))
    
    # Save statistics to file
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(stats, f, indent=2)

if __name__ == "__main__":
    main()