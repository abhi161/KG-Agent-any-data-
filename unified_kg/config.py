import os
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class Config:
    """Configuration for the Unified Knowledge Graph application with vector embedding support"""
    
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Azure OpenAI settings
        self.azure_openai_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_openai_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        self.azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")

        self.azure_openai_embedding_deployment = (
            os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT") or 
            os.getenv("AZURE_OPENAI_DEPLOYMENT") or
            "text-embedding-ada-002"  # Default if not specified
        )
        self.azure_openai_embedding_dimension = int(os.getenv("AZURE_OPENAI_EMBEDDING_DIMENSION", "1536"))
        self.temperature = float(os.getenv("TEMPERATURE", "0.0"))
        self.azure_openai_embedding_endpoint =os.getenv("AZURE_OPENAI_EMB_ENDPOINT")
        self.azure_openai_embedding_api_key = os.getenv("AZURE_OPENAI_EMB_API_KEY")
        self.azure_openai_embedding_api_version= os.getenv("AZURE_OPENAI_EMB_API_VERSION")

        # Neo4j settings
        self.neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD", "12345678")
        self.neo4j_database = os.getenv("NEO4J_DATABASE", "neo4j") # Default to 'neo4j' if not set

        # Vector settings
        self.vector_enabled = os.getenv("VECTOR_ENABLED", "true").lower() == "true"
        self.vector_similarity_threshold = float(os.getenv("VECTOR_SIMILARITY_THRESHOLD", "0.75"))
        
        # Processing settings
        self.batch_size = int(os.getenv("BATCH_SIZE", "100"))
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "3000"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
        
        # Validate required parameters
        self._validate_config()
        
        # Initialize services
        self.llm = None
        self.embeddings = None
        
    def _validate_config(self):
        """Validate the configuration parameters"""
        missing = []
        if not self.azure_openai_key: missing.append("AZURE_OPENAI_API_KEY")
        if not self.azure_openai_endpoint: missing.append("AZURE_OPENAI_ENDPOINT") 
        if not self.azure_openai_deployment: missing.append("AZURE_OPENAI_DEPLOYMENT")
        
        if missing:
            raise ValueError(f"Missing required Azure OpenAI configuration: {', '.join(missing)}")
    
    def initialize_llm(self):
        """Initialize the Azure OpenAI LLM"""
        if self.llm is not None:
            return self.llm
        
        try:
            from langchain_openai import AzureChatOpenAI
            
            self.llm = AzureChatOpenAI(
                temperature=self.temperature,
                api_key=self.azure_openai_key,
                azure_deployment=self.azure_openai_deployment,
                azure_endpoint=self.azure_openai_endpoint,
                api_version=self.azure_openai_api_version
            )
            
            logger.info("Successfully initialized Azure OpenAI LLM")
            return self.llm
        except Exception as e:
            logger.error(f"Error initializing Azure OpenAI LLM: {str(e)}")
            raise
    
    def initialize_embeddings(self):
        """Initialize the Azure OpenAI embeddings"""
        if self.embeddings is not None:
            return self.embeddings
        
        if not self.vector_enabled:
            logger.info("Vector embeddings are disabled in configuration")
            return None
        
        try:
            from langchain_openai import AzureOpenAIEmbeddings
            
            self.embeddings = AzureOpenAIEmbeddings(
                api_key=self.azure_openai_embedding_api_key,
                azure_deployment=self.azure_openai_embedding_deployment,
                azure_endpoint=self.azure_openai_embedding_endpoint,
                api_version=self.azure_openai_embedding_api_version,
            )
            
            logger.info(f"Successfully initialized Azure OpenAI embeddings with dimension {self.azure_openai_embedding_dimension}")
            return self.embeddings
        except Exception as e:
            logger.error(f"Error initializing Azure OpenAI embeddings: {str(e)}")
            logger.warning("Continuing without vector embeddings")
            self.vector_enabled = False
            return None
    
    def get_llm(self):
        """Get the initialized LLM or initialize it if needed"""
        if self.llm is None:
            self.initialize_llm()
        return self.llm
    
    def get_embeddings(self):
        """Get the initialized embeddings or initialize them if needed"""
        if not self.vector_enabled:
            return None
            
        if self.embeddings is None:
            self.initialize_embeddings()
        return self.embeddings
    
    def get_neo4j_config(self) -> Dict[str, Any]:
        """Returns the configuration for Neo4j"""
        return {
            "uri": self.neo4j_uri,
            "user": self.neo4j_user,
            "password": self.neo4j_password,
            "database": self.neo4j_database
        }
    
    def update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration from a dictionary"""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.debug(f"Updated config setting {key} = {value}")
            else:
                logger.warning(f"Unknown configuration setting: {key}")
        
        # Reinitialize services if settings have changed
        if any(key in config_dict for key in ['azure_openai_key', 'azure_openai_endpoint', 'azure_openai_deployment', 
                                              'azure_openai_api_version', 'temperature']):
            self.llm = None
            
        if any(key in config_dict for key in ['azure_openai_key', 'azure_openai_endpoint', 'azure_openai_embedding_deployment', 
                                             'azure_openai_api_version', 'vector_enabled']):
            self.embeddings = None