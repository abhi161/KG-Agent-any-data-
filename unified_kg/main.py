import os
import logging
import argparse
import json
from typing import List, Dict, Any
import time
from dotenv import load_dotenv

# Updated imports to use langchain_openai instead of langchain.llms
from langchain_openai import AzureChatOpenAI
from langchain_community.graphs import Neo4jGraph

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
    Build a knowledge graph from structured and unstructured data
    
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
    
    # Initialize LLM - updated to use AzureChatOpenAI with correct parameters
    llm = config.get_llm()
    
    # Initialize Neo4j
    graph = Neo4jGraph(
        url=config.neo4j_uri,
        username=config.neo4j_user,
        password=config.neo4j_password
    )
    
    # Initialize knowledge graph builder
    # Initialize knowledge graph builder - pass Azure parameters directly
    kg_builder = LLMEnhancedKnowledgeGraph(
        llm,
        graph,
        initial_schema=initial_schema,
        config={
            "batch_size": config.batch_size,
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap
        }
    )
    
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

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Build a unified knowledge graph')
    parser.add_argument('--csv', nargs='+', help='CSV files to process')
    parser.add_argument('--pdf', nargs='+', help='PDF files to process')
    parser.add_argument('--schema', help='Path to initial schema JSON file')
    parser.add_argument('--config', help='Path to configuration JSON file')
    parser.add_argument('--output', help='Path to output statistics JSON file')
    
    args = parser.parse_args()
    
    # Build knowledge graph
    stats = build_knowledge_graph(
        csv_files=args.csv or [],
        pdf_files=args.pdf or [],
        initial_schema_path=args.schema,
        config_path=args.config
    )
    
    # Print statistics
    print(json.dumps(stats, indent=2))
    
    # Save statistics to file
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(stats, f, indent=2)

if __name__ == "__main__":
    main()