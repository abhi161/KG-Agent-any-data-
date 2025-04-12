# unified_kg/config.py
import os
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class Config:
    """Configuration for the Unified Knowledge Graph application"""
    
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
            self.azure_openai_deployment
        )
        self.temperature = float(os.getenv("TEMPERATURE", "0.0"))
        
        # Neo4j settings
        self.neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD", "12345678")
        
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
        
        try:
            from langchain_openai import AzureOpenAIEmbeddings
            
            self.embeddings = AzureOpenAIEmbeddings(
                api_key=self.azure_openai_key,
                azure_deployment=self.azure_openai_embedding_deployment,
                azure_endpoint=self.azure_openai_endpoint,
                api_version=self.azure_openai_api_version
            )
            
            logger.info("Successfully initialized Azure OpenAI embeddings")
            return self.embeddings
        except Exception as e:
            logger.error(f"Error initializing Azure OpenAI embeddings: {str(e)}")
            raise
    
    def get_llm(self):
        """Get the initialized LLM or initialize it if needed"""
        if self.llm is None:
            self.initialize_llm()
        return self.llm
    
    def get_embeddings(self):
        """Get the initialized embeddings or initialize them if needed"""
        if self.embeddings is None:
            self.initialize_embeddings()
        return self.embeddings
    
    def get_neo4j_config(self) -> Dict[str, Any]:
        """Returns the configuration for Neo4j"""
        return {
            "uri": self.neo4j_uri,
            "user": self.neo4j_user,
            "password": self.neo4j_password
        }