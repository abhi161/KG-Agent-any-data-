import os
import logging
import argparse
import json
from typing import List, Dict, Any
import time
from dotenv import load_dotenv

from unified_kg.config import Config
from unified_kg.core.kg_builder import LLMEnhancedKnowledgeGraph
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_neo4j import Neo4jGraph

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
    config_obj = Config()
    if config_path and os.path.exists(config_path):
        logger.info(f"Loading configuration from: {config_path}")
        with open(config_path, 'r') as f:
            config_data = json.load(f)
            config_obj.update_from_dict(config_data)
    else:
         logger.info("Using default configuration from environment variables.")

    # Load initial schema
    initial_schema = None
    if initial_schema_path and os.path.exists(initial_schema_path):
        logger.info(f"Loading initial schema from: {initial_schema_path}")
        try:
            with open(initial_schema_path, 'r') as f:
                initial_schema = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing initial schema file {initial_schema_path}: {e}", exc_info=True)
            raise ValueError(f"Invalid JSON in schema file: {initial_schema_path}") from e
        except Exception as e:
            logger.error(f"Error loading initial schema file {initial_schema_path}: {e}", exc_info=True)
            raise
    else:
        logger.warning("No initial schema file provided or found.")


    # Initialize LLM
    llm = config_obj.get_llm()

    # Initialize Embeddings
    embeddings = config_obj.get_embeddings()

    # Initialize Neo4j
    logger.info(f"Connecting to Neo4j at {config_obj.neo4j_uri}")
    try:
        graph = Neo4jGraph(
            url=config_obj.neo4j_uri,
            username=config_obj.neo4j_user,
            password=config_obj.neo4j_password,
            database= config_obj.neo4j_database
        )
        # Verify connection
        graph.query("RETURN 1")
        logger.info("Successfully connected to Neo4j.")
    except Exception as e:
        logger.error(f"Failed to connect to Neo4j: {e}", exc_info=True)
        raise ConnectionError(f"Could not connect to Neo4j at {config_obj.neo4j_uri}") from e


    # Check Neo4j vector capabilities
    vector_support = check_neo4j_vector_capabilities(graph)
    if not vector_support["has_vector_support"]:
        logger.warning("Neo4j vector support not detected or enabled via config. Vector operations might be limited or performed in-memory.")
        logger.warning(f"Reason: {vector_support.get('reason', 'Vector support check failed or disabled')}")
        vector_enabled_for_builder = False
    else:
        logger.info(f"Neo4j vector support detected. Using {vector_support.get('details', 'Native Indexing or GDS')} for vector operations.")
        vector_enabled_for_builder = True

    if not config_obj.vector_enabled:
         logger.warning("Vector embeddings explicitly disabled in application configuration.")
         vector_enabled_for_builder = False
         embeddings = None 




    # Initialize knowledge graph builder
    kg_builder = LLMEnhancedKnowledgeGraph(
        llm,
        graph,
        embeddings,  
        schema_path =initial_schema_path,
        initial_schema=initial_schema,
        config={
            "batch_size": config_obj.batch_size,
            "chunk_size": config_obj.chunk_size,
            "chunk_overlap": config_obj.chunk_overlap,
            "vector_enabled": vector_enabled_for_builder, 
            "vector_similarity_threshold": config_obj.vector_similarity_threshold 
        }
        
    )

    try:
        # Process CSV files
        csv_results = []
        for csv_file in csv_files:
            if os.path.exists(csv_file):
                try:
                    logger.info(f"Starting processing of CSV file: {csv_file}")
                    result = kg_builder.process_csv_file(csv_file)
                    csv_results.append(result)
                    logger.info(f"Finished processing CSV file: {csv_file}")
                except Exception as e:
                    logger.error(f"Error processing CSV file {csv_file}: {e}", exc_info=True)
            else:
                logger.warning(f"CSV file not found, skipping: {csv_file}")

        # Process PDF files
        pdf_results = []
        for pdf_file in pdf_files:
            if os.path.exists(pdf_file):
                try:
                    logger.info(f"Starting processing of PDF file: {pdf_file}")
                    result = kg_builder.process_pdf_file(pdf_file)
                    pdf_results.append(result)
                    logger.info(f"Finished processing PDF file: {pdf_file}")
                except Exception as e:
                    logger.error(f"Error processing PDF file {pdf_file}: {e}", exc_info=True)
            else:
                logger.warning(f"PDF file not found, skipping: {pdf_file}")

        # # Cross-reference data sources
        # if csv_results and pdf_results:
        #      logger.info("Starting cross-referencing between CSV and PDF data sources.")
        #      cross_ref_results = kg_builder.cross_reference_data_sources()
        #      logger.info(f"Finished cross-referencing data sources: {cross_ref_results}")
        # else:
        #      logger.info("Skipping cross-referencing as only one type of data source was processed.")

        # Process schema changes (if SchemaManager is implemented and used)
        # Assuming SchemaManager might be part of kg_builder
        if hasattr(kg_builder, 'schema_manager') and hasattr(kg_builder.schema_manager, 'process_pending_changes'):
             logger.info("Processing pending schema changes...")
             kg_builder.schema_manager.process_pending_changes()
             logger.info("Finished processing pending schema changes.")
        else:
             logger.warning("SchemaManager or process_pending_changes method not found. Skipping schema change processing.")


        # Get statistics
        stats = kg_builder.get_statistics()

        return stats

    except Exception as e:
        logger.error(f"Critical error during knowledge graph building: {e}", exc_info=True)
        return {"error": str(e)}




def check_neo4j_vector_capabilities(graph: Neo4jGraph) -> Dict[str, Any]:
    """
    Check if Neo4j has vector capabilities enabled (Vector Index or GDS)

    Args:
        graph: Neo4j graph connection

    Returns:
        Dictionary with vector support information
    """
    capabilities = {
        "has_vector_support": False,
        "details": "None",
        "reason": "",
        "capabilities": {
            "vector_indexes": False,
            "gds": False,
            "apoc": False 
        }
    }
    try:
        try:
            vector_query = "CALL db.indexes() YIELD name, type WHERE type = 'VECTOR' RETURN count(*) > 0 as has_vector_indexes"
            vector_result = graph.query(vector_query)
            capabilities["capabilities"]["vector_indexes"] = vector_result[0]["has_vector_indexes"] if vector_result else False
        except Exception as e:
            logger.debug(f"Could not check for vector indexes (may be expected): {e}")
            capabilities["capabilities"]["vector_indexes"] = False

        # Check for GDS plugin
        try:
            gds_query = "CALL gds.list() YIELD name RETURN count(*) > 0 as has_gds"
            gds_result = graph.query(gds_query)
            capabilities["capabilities"]["gds"] = gds_result[0]["has_gds"] if gds_result else False
        except Exception as e:
            logger.debug(f"Could not check for GDS plugin (may be expected): {e}")
            capabilities["capabilities"]["gds"] = False

         # Check for APOC plugin 
        try:
            apoc_query = "CALL apoc.help('text') YIELD name RETURN count(*) > 0 as has_apoc"
            apoc_result = graph.query(apoc_query)
            capabilities["capabilities"]["apoc"] = apoc_result[0]["has_apoc"] if apoc_result else False
        except Exception as e:
             logger.debug(f"Could not check for APOC plugin (may be expected): {e}")
             capabilities["capabilities"]["apoc"] = False


        # Determine overall status and details
        if capabilities["capabilities"]["vector_indexes"]:
            capabilities["has_vector_support"] = True
            capabilities["details"] = "Neo4j Vector Index"
            capabilities["reason"] = "Native vector index support found."
        elif capabilities["capabilities"]["gds"]:
            capabilities["has_vector_support"] = True
            capabilities["details"] = "Graph Data Science (GDS) library"
            capabilities["reason"] = "GDS library found, which can support vector operations."
            logger.warning("GDS found, but native Vector Index is preferred for optimal vector search performance.")
        else:
            capabilities["has_vector_support"] = False
            capabilities["reason"] = "Neither native Vector Index support nor GDS library found."

        # Report APOC status
        if capabilities["capabilities"]["apoc"]:
            logger.info("APOC library detected (enables features like fuzzy matching).")
        else:
            logger.warning("APOC library not detected. Features like fuzzy matching may be unavailable.")


    except Exception as e:
        logger.error(f"Error checking Neo4j capabilities: {e}", exc_info=True)
        capabilities["has_vector_support"] = False
        capabilities["reason"] = f"Error during capability check: {e}"

    return capabilities


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Build a unified knowledge graph with vector embeddings')
    parser.add_argument('--csv', nargs='+', help='List of CSV files to process', default=[])
    parser.add_argument('--pdf', nargs='+', help='List of PDF files to process', default=[])
    parser.add_argument('--schema', help='Path to initial schema JSON file')
    parser.add_argument('--config', help='Path to configuration JSON file')
    parser.add_argument('--output', help='Path to output statistics JSON file')
    # Keep --vector flag maybe for explicit override, though config is primary
    parser.add_argument('--vector', action=argparse.BooleanOptionalAction, help='Explicitly enable/disable vector embeddings (overrides config if set)', default=None)

    args = parser.parse_args()

    # Validate inputs
    if not args.csv and not args.pdf:
        parser.error("No input files specified. Please provide at least one --csv or --pdf file.")

    if args.vector is not None:
        logger.warning(f"Command line argument --vector={'enable' if args.vector else 'disable'} provided. This may override settings in the config file or environment variables.")
        # Optionally, you could update the config object here if needed based on args.vector
        # Example: config_obj.vector_enabled = args.vector

    start_time = time.time()

    # Build knowledge graph
    try:
        stats = build_knowledge_graph(
            csv_files=args.csv,
            pdf_files=args.pdf,
            initial_schema_path=args.schema,
            config_path=args.config
        )
    except (ConnectionError, ValueError, FileNotFoundError) as e:
         logger.critical(f"Failed to build knowledge graph due to setup error: {e}", exc_info=True)
         stats = {"error": f"Setup Error: {e}"}
    except Exception as e:
         logger.critical(f"An unexpected critical error occurred during graph building: {e}", exc_info=True)
         stats = {"error": f"Unexpected Error: {e}"}


    end_time = time.time()
    if "error" not in stats:
        stats["processing_time_seconds"] = round(end_time - start_time, 2)

    # Print statistics
    print("--- Processing Statistics ---")
    print(json.dumps(stats, indent=2))
    print("---------------------------")

    # Save statistics to file
    if args.output:
        logger.info(f"Saving statistics to: {args.output}")
        try:
            with open(args.output, 'w') as f:
                json.dump(stats, f, indent=2)
        except IOError as e:
            logger.error(f"Error saving statistics to file {args.output}: {e}", exc_info=True)

if __name__ == "__main__":
    main()