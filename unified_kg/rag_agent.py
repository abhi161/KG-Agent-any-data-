import os
import logging
import json
import re
from typing import Dict, Any, List, Tuple, Optional

# Assuming Embeddings class is importable if needed for type hints,
# but we primarily interact via Neo4j vector index calls.
# from langchain.embeddings.base import Embeddings # Optional for type hint
from langchain_openai import AzureChatOpenAI
from langchain_neo4j import Neo4jGraph
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

logger = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_SIMILARITY_TOP_K = 3 # How many similar nodes to fetch per seed node
DEFAULT_VECTOR_INDEX_NAME = "global_embedding_index" # Needs to match index in Neo4j

# --- Prompts (Keep CYPHER_GENERATION_TEMPLATE as is, update QA_TEMPLATE) ---

CYPHER_GENERATION_TEMPLATE = """
Task: Generate a Cypher query to retrieve information from a Neo4j graph database to answer the user's question.
Schema:
{schema}

Instructions:
- Use only the provided schema (node labels, relationship types, properties).
- Use the shortest possible query that answers the question. Return node properties like name, description, dates, IDs, etc., but *exclude* the 'embedding' property unless specifically asked for similarity.
- Pay attention to capitalization and use backticks (`) for properties with spaces or special characters.
- Node properties are case-sensitive. Match Relationship types and Node labels exactly.
- Always include a LIMIT clause (e.g., LIMIT 10 or LIMIT 25) to avoid overly large results. If generating an aggregation (like COUNT), a limit is not needed after the aggregation.
- Return only the raw Cypher query with no surrounding formatting, code blocks, or markdown.
- Do not use triple backticks or any other formatting - just return the plain query.

Question: {question}
Cypher Query:
"""

# Enhanced QA_TEMPLATE to understand different context types
QA_TEMPLATE = """You are a helpful AI assistant answering questions based on a knowledge graph database.
You will be given context retrieved from the graph. This context might include:
1.  Direct Results: Data explicitly matching the generated query.
2.  Related Information: Data found through semantic similarity to the direct results (if applicable).

Context from the knowledge graph:
{context}

Important guidelines for interpreting the data:
- Node properties might have prefixes (e.g., 'n.name'). Use the property value.
- Clearly distinguish between Direct Results and Related Information if both are present.
- Explain relationships simply if they appear in the results (e.g., "Dr. Smith WORKS_AT Central Hospital").
- If the query returned no direct results but related information was found, mention that.
- Synthesize the information from all context sections to provide a comprehensive answer.
- If the context contains *any* information (direct or related), use it. Do not state you couldn't find information if data is present.
- Prioritize information from 'Direct Results' if it directly answers the question, but use 'Related Information' to add valuable context or connections.

Question: {question}
Answer:
"""

class KnowledgeGraphRAGAgent:
    # Add embeddings parameter, default similarity threshold
    def __init__(self, llm, graph: Neo4jGraph, graph_schema: str, similarity_threshold: float = 0.85):
        """
        Initializes the RAG Agent with embedding support.

        Args:
            llm: The Language Model instance.
            graph: The Neo4jGraph connection instance.
            graph_schema: A string representation of the graph schema.
            similarity_threshold: The threshold for vector similarity searches.
        """
        self.llm = llm
        self.graph = graph
        self.graph_schema_string = graph_schema
        self.similarity_threshold = similarity_threshold
        # Check if vector index exists (optional but good practice)
        self.vector_index_exists = self._check_vector_index()

        try:
            self.cypher_prompt = PromptTemplate(
                template=CYPHER_GENERATION_TEMPLATE,
                input_variables=["schema", "question"]
            )
            self.qa_prompt = PromptTemplate(
                template=QA_TEMPLATE,
                input_variables=["context", "question"]
            )
            self.cypher_chain = LLMChain(llm=llm, prompt=self.cypher_prompt)
            self.qa_chain = LLMChain(llm=llm, prompt=self.qa_prompt)

            logger.info("KnowledgeGraphRAGAgent initialized with custom LLM chains and embedding capability check.")

        except Exception as e:
            logger.error(f"Failed to initialize RAG components: {e}", exc_info=True)
            raise RuntimeError(f"Could not initialize RAG components: {e}")

    def _check_vector_index(self) -> bool:
        """Checks if the assumed vector index exists."""
        try:
            # This query lists indexes; adjust if using GDS or different index type
            result = self.graph.query(f"SHOW INDEXES WHERE name = '{DEFAULT_VECTOR_INDEX_NAME}' AND type = 'VECTOR'")
            if result and len(result) > 0:
                logger.info(f"Vector index '{DEFAULT_VECTOR_INDEX_NAME}' found.")
                return True
            else:
                logger.warning(f"Vector index '{DEFAULT_VECTOR_INDEX_NAME}' not found. Semantic similarity search will be disabled.")
                return False
        except Exception as e:
            logger.warning(f"Could not check for vector index '{DEFAULT_VECTOR_INDEX_NAME}' (may indicate incompatible Neo4j version or index missing): {e}")
            return False

    def _extract_node_info(self, result_list: List[Dict]) -> List[Tuple[str, str]]:
        """
        Extracts potential node IDs and labels from query results.
        Looks for elementId() patterns or standard 'id'/'label' keys if available.
        Returns a list of tuples: [(node_id, node_label), ...].
        """
        nodes_found = []
        processed_ids = set()

        for item in result_list:
            if not isinstance(item, dict):
                continue

            node_id = None
            node_label = None

            # Try to find ID using common patterns or elementId convention
            potential_id_keys = [k for k in item if k.endswith('.id') or k == 'id' or 'elementId' in k or k.endswith('Id')]
            if potential_id_keys:
                # Prioritize keys that look like elementId results if graph store returns them
                element_id_keys = [k for k in potential_id_keys if isinstance(item[k], str) and ':' in item[k]] # Simple check for elementId format like '4:xxx:yyy'
                if element_id_keys:
                    node_id = item[element_id_keys[0]]
                else:
                    # Fallback to other ID keys
                    # Check if value looks like an elementId even if key name is just 'id'
                    id_key = potential_id_keys[0]
                    if isinstance(item[id_key], str) and ':' in item[id_key]:
                         node_id = item[id_key]
                    # Add more specific ID logic if needed based on your query results

            # Try to find Label (less reliable this way, better to fetch label with ID)
            # Placeholder: In a real scenario, you might need another query to get label from ID.
            # For now, we'll assume the ID is enough to fetch embedding.
            potential_label_keys = [k for k in item if 'label' in k.lower()]
            if potential_label_keys:
                 node_label = str(item[potential_label_keys[0]]) # Very basic guess

            if node_id and node_id not in processed_ids:
                # We mainly need the ID to fetch the embedding
                nodes_found.append((node_id, node_label or "Unknown")) # Store ID, use Unknown label if not found
                processed_ids.add(node_id)

        # Limit number of nodes for similarity search
        max_nodes_for_similarity = 3
        return nodes_found[:max_nodes_for_similarity]


    def _run_similarity_search(self, node_id: str) -> List[Dict]:
        """
        Runs vector similarity search for a given node ID.
        Fetches the embedding first, then queries the index.
        """
        if not self.vector_index_exists:
            return []

        try:
            # 1. Fetch the embedding for the node ID
            embedding_query = """
            MATCH (n) WHERE elementId(n) = $node_id
            RETURN n.embedding AS embedding
            LIMIT 1
            """
            embedding_result = self.graph.query(embedding_query, {"node_id": node_id})

            if not embedding_result or not embedding_result[0].get('embedding'):
                logger.warning(f"Could not retrieve embedding for node {node_id}. Skipping similarity search.")
                return []

            node_embedding = embedding_result[0]['embedding']

            # 2. Run the vector query
            similarity_query = f"""
            CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
            YIELD node, score
            WHERE score >= $threshold AND elementId(node) <> $node_id
            RETURN elementId(node) AS id, labels(node)[0] AS type, node.name AS name, score,
                   // Optionally return a few key properties, exclude embedding again
                   node {{ .name, .description, .specialty, .classification }} AS properties
            ORDER BY score DESC
            LIMIT $limit
            """
            params = {
                "index_name": DEFAULT_VECTOR_INDEX_NAME,
                "top_k": DEFAULT_SIMILARITY_TOP_K * 2, # Query more initially
                "embedding": node_embedding,
                "threshold": self.similarity_threshold,
                "node_id": node_id, # Exclude self
                "limit": DEFAULT_SIMILARITY_TOP_K
            }
            similar_nodes = self.graph.query(similarity_query, params)
            logger.info(f"Found {len(similar_nodes)} similar nodes for node {node_id}")
            return similar_nodes if similar_nodes else []

        except Exception as e:
            logger.error(f"Error during similarity search for node {node_id}: {e}", exc_info=True)
            # Potentially disable vector search if index query consistently fails
            if "NoSuchIndexException" in str(e) or "index does not exist" in str(e):
                logger.error(f"Vector index '{DEFAULT_VECTOR_INDEX_NAME}' query failed. Disabling vector search.")
                self.vector_index_exists = False
            return []


    
    

    def _format_context_for_qa(self, direct_results: List[Dict], similarity_results: Dict[str, List[Dict]], question: str) -> str:
        """
        Formats direct query results and similarity search results for the QA LLM,
        applying truncation and prioritization to manage context length.
        """
        # Add constants for context limits
        MAX_DIRECT_RESULTS_TO_FORMAT = 10  # Max direct results items to include
        MAX_SIMILAR_NODES_TO_FORMAT = 5   # Max total similar nodes across all sources
        MAX_CHARS_PER_PROPERTY_VALUE = 150 # Max characters for any property value string
        MAX_TOTAL_CONTEXT_CHARS = 100000 # Target character limit (approximate, less than token limit)
        
        context_parts = []
        current_char_count = 0

        # --- Format Direct Results (Limited) ---
        direct_results_formatted_count = 0
        if isinstance(direct_results, list) and len(direct_results) > 0:
            context_parts.append("Direct Results from Query:")
            formatted_direct = []
            all_keys = set() # Optional: Extract keys for potential consistency

            items_to_process = direct_results[:MAX_DIRECT_RESULTS_TO_FORMAT] # Limit items processed
            logger.info(f"Formatting max {MAX_DIRECT_RESULTS_TO_FORMAT} direct results (out of {len(direct_results)}).")

            # --- Pre-calculate keys if needed (optional) ---
            # for item in items_to_process:
            #      if isinstance(item, dict): all_keys.update(item.keys())

            for i, item in enumerate(items_to_process):
                item_str_parts = []
                if not isinstance(item, dict):
                    item_str = f"- Result {i+1}: {str(item)}"
                    if current_char_count + len(item_str) < MAX_TOTAL_CONTEXT_CHARS:
                        item_str_parts.append(item_str)
                        current_char_count += len(item_str)
                    else: break # Stop adding results if limit approached
                else:
                    result_header = f"- Result {i+1}:"
                    item_str_parts.append(result_header)
                    temp_item_char_count = len(result_header)

                    # Using item's keys directly for simplicity now
                    keys_to_format = sorted(item.keys())

                    for key in keys_to_format:
                        clean_key = key.split('.')[-1] if '.' in key else key
                        value = item[key]
                        # Basic string conversion and truncation
                        try:
                            value_str = str(value)
                        except Exception:
                            value_str = "[Unrepresentable Value]"

                        if len(value_str) > MAX_CHARS_PER_PROPERTY_VALUE:
                            value_str = value_str[:MAX_CHARS_PER_PROPERTY_VALUE] + "..."

                        # Exclude embeddings explicitly if they somehow got here
                        if clean_key == "embedding": continue

                        prop_str = f"  - {clean_key}: {value_str}"

                        # Check if adding this property exceeds limits
                        if current_char_count + temp_item_char_count + len(prop_str) < MAX_TOTAL_CONTEXT_CHARS:
                            item_str_parts.append(prop_str)
                            temp_item_char_count += len(prop_str)
                        else:
                            # Stop adding properties for *this item* if limit reached
                            item_str_parts.append("  - ... (truncated)")
                            break

                # Add the formatted item string if it contains more than just the header
                if len(item_str_parts) > 1:
                    formatted_item = "\n".join(item_str_parts)
                    formatted_direct.append(formatted_item)
                    current_char_count += temp_item_char_count # Add item's char count
                    direct_results_formatted_count += 1


            if formatted_direct:
                context_parts.append("\n".join(formatted_direct))
            else:
                # Remove the header if no results were actually added
                context_parts.pop() # Remove "Direct Results from Query:"
                context_parts.append("Direct Results from Query: The query returned no displayable direct results (or results were truncated).")


        elif isinstance(direct_results, str): # Handle error messages
            error_str = f"Direct Results from Query:\n{direct_results}"
            context_parts.append(error_str[:MAX_TOTAL_CONTEXT_CHARS]) # Truncate error too
            current_char_count += len(error_str)
        else:
            context_parts.append("Direct Results from Query: The query returned no direct results.")


        # --- Format Similarity Results (Limited and Conditional) ---
        if similarity_results and current_char_count < MAX_TOTAL_CONTEXT_CHARS * 0.9: # Only add if space remaining
            context_parts.append("\nRelated Information (Semantic Similarity):")
            formatted_similar = []
            similar_nodes_added_count = 0

            for source_node_id, similar_nodes in similarity_results.items():
                if similar_nodes_added_count >= MAX_SIMILAR_NODES_TO_FORMAT: break # Stop if overall limit reached

                if similar_nodes:
                    # Header for this source node's similar items (optional, adds length)
                    # similar_header = f"\nNodes similar to item related to ID {source_node_id}:"
                    # context_parts.append(similar_header)
                    # current_char_count += len(similar_header)

                    for node in similar_nodes:
                        if similar_nodes_added_count >= MAX_SIMILAR_NODES_TO_FORMAT: break

                        # Format properties concisely, applying truncation
                        props_str_parts = []
                        node_props = node.get('properties', {})
                        if isinstance(node_props, dict): # Ensure it's a dictionary
                            for k, v in node_props.items():
                                if v: # Skip empty properties
                                    try:
                                        v_str = str(v)
                                    except Exception: v_str = "[Unrepresentable]"
                                    if len(v_str) > MAX_CHARS_PER_PROPERTY_VALUE:
                                            v_str = v_str[:MAX_CHARS_PER_PROPERTY_VALUE] + "..."
                                    props_str_parts.append(f"{k}: {v_str}")
                        props_str = ", ".join(props_str_parts) if props_str_parts else 'N/A'

                        # Format the main node info
                        node_info_str = f"- Found '{node.get('name', 'N/A')}' (Type: {node.get('type', 'N/A')}, Score: {node.get('score', 0):.3f}). Properties: {props_str}."

                        # Check length before adding
                        if current_char_count + len(node_info_str) < MAX_TOTAL_CONTEXT_CHARS:
                            formatted_similar.append(node_info_str)
                            current_char_count += len(node_info_str)
                            similar_nodes_added_count += 1
                        else:
                            # Stop adding similar nodes if limit is reached
                            break # Break inner loop

            if formatted_similar:
                context_parts.append("\n".join(formatted_similar))
            else:
                # Remove the header if no similar results were added
                context_parts.pop() # Remove "Related Information..." header

        # --- Combine ---
        final_context = "\n".join(context_parts)
        logger.info(f"Final formatted context character count: {len(final_context)}")
        if len(final_context) >= MAX_TOTAL_CONTEXT_CHARS:
            logger.warning("Formatted context potentially still too long despite truncation.")
        return final_context


    def _clean_cypher_query(self, raw_response):
        """Extract and clean Cypher query from LLM response."""
        # Handles markdown code blocks and adds LIMIT if missing
        logger.debug(f"Raw Cypher response: {raw_response}")
        cypher_query = raw_response.strip()

        # Remove potential markdown backticks
        if cypher_query.startswith("```cypher"):
            cypher_query = cypher_query[len("```cypher"):].strip()
        if cypher_query.startswith("```"):
            cypher_query = cypher_query[3:].strip()
        if cypher_query.endswith("```"):
            cypher_query = cypher_query[:-3].strip()

        # Remove common conversational text sometimes added by models
        cypher_query = re.sub(r"^\s*Here'?s? the Cypher query:?\s*", "", cypher_query, flags=re.IGNORECASE)

        # Ensure the query has a LIMIT clause for safety, unless it's clearly an aggregation
        is_aggregation = any(agg in cypher_query.upper() for agg in ["COUNT(", "SUM(", "AVG(", "MIN(", "MAX("])
        if "LIMIT" not in cypher_query.upper() and "RETURN" in cypher_query.upper() and not is_aggregation:
            if ";" in cypher_query:
                # Attempt to insert before a final semicolon
                parts = cypher_query.rsplit(";", 1)
                cypher_query = parts[0].strip() + " LIMIT 25;" + parts[1] # Added limit 25
            else:
                cypher_query = cypher_query + " LIMIT 25" # Added limit 25
            logger.info(f"Added default LIMIT clause to query.")

        logger.info(f"Cleaned Cypher Query: {cypher_query}")
        return cypher_query

    def run_query(self, question: str) -> Dict[str, Any]:
        """
        Runs the RAG process: Cypher generation -> Query Execution -> Similarity Search -> QA Generation.
        """
        logger.info(f"Processing question: {question}")
        result = {
            "query": question,
            "result": "",
            "intermediate_steps": []
        }
        direct_results = "No direct results." # Default
        similarity_results_map = {} # Stores {source_node_id: [similar_nodes_list]}

        try:
            # Step 1: Generate Cypher query
            cypher_response = self.cypher_chain.invoke({
                "schema": self.graph_schema_string,
                "question": question
            })
            raw_cypher_response = cypher_response.get("text", "").strip()
            cypher_query = self._clean_cypher_query(raw_cypher_response)
            result["intermediate_steps"].append({"type": "Generated Cypher", "content": cypher_query})

            # Step 2: Execute the initial Cypher query
            query_error = None
            if cypher_query and any(cmd in cypher_query.upper() for cmd in ["MATCH", "RETURN", "CALL", "SHOW"]): # Allow SHOW
                try:
                    direct_results = self.graph.query(cypher_query) # Execute
                except Exception as e:
                    query_error = f"Error executing generated Cypher: {e}"
                    logger.error(f"Cypher execution error for query '{cypher_query}': {e}", exc_info=True)
                    direct_results = query_error # Pass error message as context
            else:
                direct_results = f"Invalid or non-executing Cypher generated: '{cypher_query}'"
                logger.warning(direct_results)

            result["intermediate_steps"].append({"type": "Direct Query Results", "content": direct_results if not query_error else query_error})

            # Step 3: Perform Similarity Search (if direct results are valid and contain nodes)
            if not query_error and isinstance(direct_results, list) and len(direct_results) > 0 and self.vector_index_exists:
                # Identify nodes from the direct results that might have embeddings
                nodes_to_search = self._extract_node_info(direct_results)
                logger.info(f"Found {len(nodes_to_search)} potential nodes in direct results for similarity search.")

                for node_id, _ in nodes_to_search: # We only need the ID here
                     similar_nodes = self._run_similarity_search(node_id)
                     if similar_nodes:
                         similarity_results_map[node_id] = similar_nodes # Store results keyed by source node ID

                if similarity_results_map:
                     result["intermediate_steps"].append({"type": "Similarity Search Results", "content": similarity_results_map})


            # Step 4: Format combined context for the QA model
            # Pass both direct results and the similarity results map
            context_str = self._format_context_for_qa(direct_results, similarity_results_map, question)
            logger.debug(f"Formatted context for QA (truncated): {context_str[:500]}...") # Log truncated context

            # Step 5: Generate final answer using QA chain
            qa_response = self.qa_chain.invoke({
                "context": context_str,
                "question": question
            })
            result["result"] = qa_response.get("text", "No answer generated.")

        except Exception as e:
            logger.error(f"Unhandled error in RAGAgent.run_query for question '{question}': {e}", exc_info=True)
            result["result"] = f"An unexpected error occurred: {e}"

        return result

# --- Updated app.py (No changes needed if it already uses KnowledgeGraphRAGAgent.run_query) ---
# The Streamlit app code you provided already calls `rag_agent.run_query` and displays
# the `result` and `intermediate_steps`, so it should automatically benefit from the
# enhanced context and show the similarity results in the expander if they are generated.
# Ensure the initialization part in app.py correctly passes the necessary arguments (llm, graph, schema string).
# The current app.py initialization looks okay.