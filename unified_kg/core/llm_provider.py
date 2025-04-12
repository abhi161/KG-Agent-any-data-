# unified_kg/core/llm_provider.py
import os
import logging
from typing import Optional
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class AzureOpenAIProvider:
    """
    Provider for Azure OpenAI services (LLM and embeddings)
    """
    
    def __init__(self,
                 api_key: Optional[str] = None,
                 endpoint: Optional[str] = None,
                 api_version: Optional[str] = None,
                 deployment: Optional[str] = None,
                 embedding_deployment: Optional[str] = None,
                 temperature: float = 0.0):
        """
        Initialize Azure OpenAI provider
        
        Args:
            api_key: Azure OpenAI API key (defaults to AZURE_OPENAI_API_KEY env var)
            endpoint: Azure OpenAI endpoint (defaults to AZURE_OPENAI_ENDPOINT env var)
            api_version: Azure OpenAI API version (defaults to AZURE_OPENAI_API_VERSION env var)
            deployment: Azure OpenAI deployment for chat (defaults to AZURE_OPENAI_DEPLOYMENT env var)
            embedding_deployment: Azure OpenAI deployment for embeddings 
                                 (defaults to AZURE_OPENAI_EMBEDDING_DEPLOYMENT or AZURE_OPENAI_DEPLOYMENT env var)
            temperature: Temperature parameter for LLM (0.0 to 1.0)
        """
        # Load environment variables if not already loaded
        load_dotenv()
        
        # Set credentials from parameters or environment variables
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_version = api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")
        self.deployment = deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT")
        
        # For embedding deployment, use the specified value, or the environment variable,
        # or fall back to the chat deployment
        self.embedding_deployment = (
            embedding_deployment or 
            os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT") or 
            self.deployment
        )
        
        self.temperature = temperature
        
        # Validate required parameters
        if not all([self.api_key, self.endpoint, self.deployment]):
            missing = []
            if not self.api_key: missing.append("AZURE_OPENAI_API_KEY")
            if not self.endpoint: missing.append("AZURE_OPENAI_ENDPOINT")
            if not self.deployment: missing.append("AZURE_OPENAI_DEPLOYMENT")
            
            raise ValueError(f"Missing required Azure OpenAI configuration: {', '.join(missing)}")
        
        logger.info(f"Initializing Azure OpenAI with deployment: {self.deployment} and API version: {self.api_version}")
        logger.info(f"Using embedding deployment: {self.embedding_deployment}")
        
        # Initialize LLM and embeddings
        self._initialize_llm()
        self._initialize_embeddings()
    
    def _initialize_llm(self):
        """Initialize Azure OpenAI LLM"""
        try:
            from langchain_openai import AzureChatOpenAI
            
            self.llm = AzureChatOpenAI(
                temperature=self.temperature,
                api_key=self.api_key,
                azure_deployment=self.deployment,
                azure_endpoint=self.endpoint,
                api_version=self.api_version
            )
            
            logger.info("Successfully initialized Azure OpenAI LLM")
        except Exception as e:
            logger.error(f"Error initializing Azure OpenAI LLM: {str(e)}")
            raise
    
    def _initialize_embeddings(self):
        """Initialize Azure OpenAI embeddings"""
        try:
            from langchain_openai import AzureOpenAIEmbeddings
            
            self.embeddings = AzureOpenAIEmbeddings(
                api_key=self.api_key,
                azure_deployment=self.embedding_deployment,
                azure_endpoint=self.endpoint,
                api_version=self.api_version
            )
            
            logger.info("Successfully initialized Azure OpenAI embeddings")
        except Exception as e:
            logger.error(f"Error initializing Azure OpenAI embeddings: {str(e)}")
            raise
    
    def get_llm(self):
        """Get the initialized LLM"""
        return self.llm
    
    def get_embeddings(self):
        """Get the initialized embeddings"""
        return self.embeddings