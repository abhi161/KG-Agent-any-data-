import os
import logging
import json
import re
from typing import Dict, Any, List, Tuple, Optional

from langchain_openai import AzureChatOpenAI
from langchain_neo4j import Neo4jGraph
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

logger = logging.getLogger(__name__)

# Updated Cypher generation template - explicitly asking for raw queries without markdown
CYPHER_GENERATION_TEMPLATE = """
Task: Generate a Cypher query to retrieve information from a Neo4j graph database to answer the user's question.
Schema:
{schema}

Instructions:
- Use only the provided schema (node labels, relationship types, properties).
- Use the shortest possible query that answers the question.
- Pay attention to capitalization and use backticks (`) for properties with spaces or special characters.
- Node properties are case-sensitive. Match Relationship types and Node labels exactly.
- Return only the raw Cypher query with no surrounding formatting, code blocks, or markdown.
- Do not use triple backticks or any other formatting - just return the plain query.

Question: {question}
Cypher Query:
"""

QA_TEMPLATE = """You are a helpful AI assistant answering questions based on a knowledge graph database.

Context from the knowledge graph:
{context}

Important guidelines for interpreting the data:
- Knowledge graph entities often use ID codes as identifiers (like 'P123', 'ORG456')
- Property names may appear with prefixes (like 'n.name' or 'r.date')
- Relationship data shows connections between different entities
- Some values represent IDs while others contain actual information
- Always provide the information found in the results, even if it consists primarily of IDs
- If the context contains any results, use that information to construct your answer
- Never respond with "I couldn't find information" if any data was returned
- Explain relationships between entities when they appear in the results

Question: {question}
Answer:
"""

class KnowledgeGraphRAGAgent:
    def __init__(self, llm, graph: Neo4jGraph, graph_schema: str):
        """
        Initializes the RAG Agent.

        Args:
            llm: The Language Model instance.
            graph: The Neo4jGraph connection instance.
            graph_schema: A string representation of the graph schema.
        """
        self.llm = llm
        self.graph = graph
        self.graph_schema_string = graph_schema
        
        try:
            # Create separate LLM chains instead of using GraphCypherQAChain
            self.cypher_prompt = PromptTemplate(
                template=CYPHER_GENERATION_TEMPLATE,
                input_variables=["schema", "question"]
            )
            
            self.qa_prompt = PromptTemplate(
                template=QA_TEMPLATE,
                input_variables=["context", "question"]
            )
            
            # Initialize LLM chains
            self.cypher_chain = LLMChain(llm=llm, prompt=self.cypher_prompt)
            self.qa_chain = LLMChain(llm=llm, prompt=self.qa_prompt)
            
            logger.info("KnowledgeGraphRAGAgent initialized with custom LLM chains")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG components: {e}", exc_info=True)
            raise RuntimeError(f"Could not initialize RAG components: {e}")

    def _format_context_for_qa(self, context, question):
        """
        Transform Neo4j results into natural language the QA model can understand.
        Handles cleaning property names, formatting results, and adding context.
        """
        # Handle non-list contexts
        if not isinstance(context, list):
            return str(context)
            
        # Handle empty results
        if len(context) == 0:
            return "The query was executed but returned no results from the knowledge graph."
            
        # Create a natural language description
        formatted_text = [f"Found {len(context)} results in the knowledge graph:"]
        
        # Extract all unique keys across all results
        all_keys = set()
        for item in context:
            all_keys.update(item.keys())
        
        # Determine entity types involved in the results
        entity_types = set()
        for key in all_keys:
            if '.' in key:
                prefix = key.split('.')[0]
                if prefix in ['d', 'p', 'm', 'h', 'pr', 'man']:  # Common prefixes in Neo4j queries
                    entity_type = self._prefix_to_entity_type(prefix)
                    if entity_type:
                        entity_types.add(entity_type)
        
        # Format each result
        for i, item in enumerate(context):
            result_text = [f"Result {i+1}:"]
            
            for key in sorted(all_keys):  # Sort keys for consistent output
                if key in item:
                    # Clean up property names by removing prefixes
                    clean_key = key.split('.')[-1] if '.' in key else key
                    value = item[key]
                    
                    # Handle embedding vectors (truncate or remove)
                    if clean_key == "embedding" and isinstance(value, (list, str)) and len(str(value)) > 100:
                        continue  # Skip embeddings entirely
                    
                    # Add the property and value
                    result_text.append(f"- {clean_key}: {value}")
                    
            formatted_text.append("\n".join(result_text))
        
        # Add interpretation guidance based on the data and question
        guidance_text = []
        
        # Add ID interpretation if relevant
        if any(k.endswith('.id') or k.endswith('_id') or 'id' in k.lower() for k in all_keys):
            guidance_text.append("Note: Values like 'D001' are doctor IDs, 'P001' are patient IDs, and 'M001' are medication IDs.")
        
        # Add specific explanation based on entity types
        if entity_types:
            entity_type_str = ", ".join(entity_types)
            guidance_text.append(f"The results contain information about: {entity_type_str}.")
        
        # Context about the data model
        guidance_text.append("The knowledge graph uses ID values as names in some cases.")
        
        # Add query-specific explanation
        if "doctor" in question.lower() and "patient" in question.lower():
            if any("d.name" in k or "doctor" in k.lower() for k in all_keys):
                guidance_text.append("The doctor IDs represent the physicians who treat the specified patient(s).")
        
        # Add the guidance text to the formatted output
        formatted_text.extend(guidance_text)
        
        return "\n\n".join(formatted_text)
    
    def _prefix_to_entity_type(self, prefix):
        """Map common Neo4j query prefixes to entity types for better explanations."""
        mapping = {
            'd': 'Doctors',
            'p': 'Patients',
            'm': 'Medications',
            'h': 'Hospitals',
            'pr': 'Prescriptions',
            'man': 'Manufacturers'
        }
        return mapping.get(prefix.lower())
    
    def _clean_cypher_query(self, raw_response):
        """Extract and clean Cypher query from LLM response."""
        cypher_query = raw_response.strip()
        
        # Extract from code blocks if needed
        if "```" in raw_response:
            code_blocks = re.findall(r'```(?:cypher)?([^`]+)```', raw_response, re.DOTALL)
            if code_blocks:
                cypher_query = code_blocks[0].strip()
        
        # Ensure the query has a LIMIT clause for safety
        if "LIMIT" not in cypher_query.upper() and "RETURN" in cypher_query.upper():
            if ";" in cypher_query:
                cypher_query = cypher_query.replace(";", " LIMIT 20;")
            else:
                cypher_query = cypher_query + " LIMIT 20"
        
        return cypher_query

    def run_query(self, question: str) -> Dict[str, Any]:
        """
        Runs the RAG process with improved context formatting for better question answering.
        """
        logger.info(f"Processing question: {question}")
        
        result = {
            "query": question,
            "result": "",
            "intermediate_steps": []
        }
        
        try:
            # Step 1: Generate Cypher query
            cypher_response = self.cypher_chain.invoke({
                "schema": self.graph_schema_string,
                "question": question
            })
            
            # Extract and clean the query
            raw_response = cypher_response.get("text", "").strip()
            cypher_query = self._clean_cypher_query(raw_response)
            
            result["intermediate_steps"].append({"type": "Generated Cypher", "content": cypher_query})
            
            # Step 2: Execute the query
            try:
                if cypher_query and any(cmd in cypher_query.upper() for cmd in ["MATCH", "RETURN", "CALL"]):
                    query_result = self.graph.query(cypher_query)
                    
                    # Filter out large fields like embeddings
                    if isinstance(query_result, list):
                        for item in query_result:
                            if isinstance(item, dict):
                                # Remove embedding vectors
                                keys_to_remove = []
                                for key, value in item.items():
                                    if key.endswith("embedding") or (isinstance(value, (list, str)) and len(str(value)) > 100):
                                        keys_to_remove.append(key)
                                
                                for key in keys_to_remove:
                                    item.pop(key, None)
                    
                    context = query_result
                else:
                    context = f"Error: Invalid or empty Cypher query: '{cypher_query}'"
            except Exception as query_error:
                context = f"Error executing query: {str(query_error)}"
                logger.error(f"Query execution error: {query_error}", exc_info=True)
            
            result["intermediate_steps"].append({"type": "Retrieved Context", "content": context})
            
            # Step 3: Format context for the QA model
            context_str = self._format_context_for_qa(context, question)

            print(f"Formatted context for QA: {context_str[:500]}...")
            
            # Step 4: Generate final answer
            qa_response = self.qa_chain.invoke({
                "context": context_str,
                "question": question
            })
            
            result["result"] = qa_response.get("text", "No answer generated")
            
        except Exception as e:
            logger.error(f"Error processing question '{question}': {e}", exc_info=True)
            result["result"] = f"An error occurred while processing your question: {e}"
            
        return result