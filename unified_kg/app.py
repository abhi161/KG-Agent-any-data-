# app.py (or wherever initialize_rag_agent is defined)
import streamlit as st
import json
import os
from dotenv import load_dotenv
import logging
from unified_kg.rag_agent import KnowledgeGraphRAGAgent
from unified_kg.config import Config
# Removed SchemaManager import as it's not directly used here for schema *string* formatting
from langchain_neo4j import Neo4jGraph

logger = logging.getLogger(__name__)

# --- Page Configuration ---
st.set_page_config(page_title="Knowledge Graph RAG Chat", layout="wide")
st.title("⚕️ Knowledge Graph RAG Assistant")
st.caption("Ask questions about the data integrated into the Neo4j knowledge graph.")

# --- Initialization (Cached) ---
@st.cache_resource # Cache resources like LLM and graph connection
def initialize_rag_agent():
    load_dotenv()
    try:
        config_obj = Config()
        llm = config_obj.get_llm()
        # embedding = config_obj.get_embeddings() # Embedding unused in RAG agent init directly
        if llm is None:
            st.error("LLM initialization failed. Cannot proceed.")
            st.stop()
        graph = Neo4jGraph(
            url=config_obj.neo4j_uri,
            username=config_obj.neo4j_user,
            password=config_obj.neo4j_password,
            database=config_obj.neo4j_database
        )
        graph.query("RETURN 1") # Verify connection

        # --- Load and Process Schema ---
        schema_path = config_obj.initial_schema_path
        schema_definition_for_display = "Error: Schema could not be processed." # Default error message
        schema_str_for_agent = "Schema not available" # Fallback for agent init
        st.info(f"Attempting to load schema from: {schema_path}")

        if schema_path and os.path.exists(schema_path):
            try:
                with open(schema_path, 'r', encoding='utf-8') as f:
                    schema_data = json.load(f)

                if not isinstance(schema_data, dict):
                    st.error(f"Schema file '{schema_path}' did not contain valid JSON (expected a dictionary, got {type(schema_data)}).")
                    st.stop()

                # --- Safer Schema String Formatting ---
                schema_parts = []
                schema_parts.append("Node Types & Properties:")
                for et in schema_data.get("entity_types", []):
                    if isinstance(et, dict):
                        name = et.get('name', 'UnknownType')
                        props_list = et.get("properties") # Get value first

                        # Check if props_list is iterable (list/tuple) and join strings
                        if isinstance(props_list, (list, tuple)):
                            props_str = ", ".join(map(str, props_list)) if props_list else "None"
                        else:
                            props_str = "N/A (Invalid Format)" # Handle non-iterable or None

                        schema_parts.append(f"- {name} ({props_str})")
                    else:
                        logger.warning(f"Skipping invalid entity type format in schema: {et}")

                schema_parts.append("\nRelationship Types (Source -> Target):")
                for rt in schema_data.get("relation_types", []):
                     if isinstance(rt, dict):
                        rel_name = rt.get('name', 'UnknownRelation')
                        src_list = rt.get("source_types") # Get value first
                        tgt_list = rt.get("target_types") # Get value first

                        # Check if src_list is iterable and join strings
                        if isinstance(src_list, (list, tuple)):
                            src_str = ",".join(map(str, src_list)) if src_list else "Any"
                        else:
                            src_str = "Any/None" # Handle None or non-list

                        # Check if tgt_list is iterable and join strings
                        if isinstance(tgt_list, (list, tuple)):
                            tgt_str = ",".join(map(str, tgt_list)) if tgt_list else "Any"
                        else:
                            tgt_str = "Any/None" # Handle None or non-list

                        schema_parts.append(f"- {rel_name} ({src_str} -> {tgt_str})")
                     else:
                         logger.warning(f"Skipping invalid relation type format in schema: {rt}")

                # --- Final Schema String for Agent ---
                schema_str_for_agent = "\n".join(schema_parts)
                schema_definition_for_display = schema_str_for_agent # Use the same for display now

            except json.JSONDecodeError as json_e:
                st.error(f"Error parsing schema file '{schema_path}' as JSON: {json_e}. Please ensure the file is valid JSON.")
                st.stop()
            except Exception as e:
                # Catch the specific TypeError here if needed, or generic Exception
                st.error(f"An unexpected error occurred loading/processing schema from {schema_path}: {e}")
                logger.error(f"Schema processing error detail:", exc_info=True) # Log traceback
                st.stop()
        else:
            st.error(f"Schema file not found at configured path: {schema_path}. Cannot initialize RAG agent correctly.")
            st.stop()

        # --- Instantiate Agent ---
        # Pass the processed schema string
        agent = KnowledgeGraphRAGAgent(llm, graph, schema_str_for_agent) # Pass the formatted string
        return agent

    except Exception as e:
        st.error(f"Initialization Failed during setup: {e}")
        logger.error(f"Full Initialization Traceback:", exc_info=True)
        st.stop()
        return None

rag_agent = initialize_rag_agent()

# --- Chat Interface (Keep as is) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "intermediate_steps" in message:
             with st.expander("Show Retrieval Process"):
                for step in message["intermediate_steps"]:
                     st.write(f"**{step['type']}:**")
                     if isinstance(step['content'], list):
                          st.json(step['content'], expanded=False)
                     elif isinstance(step['content'], dict) and step['type'] == 'Similarity Search Results': # Specific handling for similarity dict
                         st.json(step['content'], expanded=False)
                     elif isinstance(step['content'], str) and step['type'] == 'Generated Cypher':
                          st.code(step['content'], language='cypher')
                     else:
                          st.write(step['content']) # Display other string content

# Accept user input
if prompt := st.chat_input("Ask a question about the knowledge graph..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")

        if rag_agent:
            response_data = rag_agent.run_query(prompt)
            assistant_response = response_data["result"]
            intermediate_steps = response_data["intermediate_steps"]

            message_placeholder.markdown(assistant_response)

            st.session_state.messages.append({
                "role": "assistant",
                "content": assistant_response,
                "intermediate_steps": intermediate_steps
            })
            # Rerun to immediately display expander (Streamlit idiom)
            # Consider if immediate display is desired or just on next interaction
            # st.rerun() # <-- Uncomment this if you want the expander below to show immediately

            # Display intermediate steps (this might appear below input on first run without rerun)
            with st.expander("Show Retrieval Process", expanded=False): # Start collapsed
                 for step in intermediate_steps:
                      st.write(f"**{step['type']}:**")
                      if isinstance(step['content'], list):
                           st.json(step['content'], expanded=False)
                      elif isinstance(step['content'], dict) and step['type'] == 'Similarity Search Results':
                           st.json(step['content'], expanded=False)
                      elif isinstance(step['content'], str) and step['type'] == 'Generated Cypher':
                           st.code(step['content'], language='cypher')
                      else:
                           st.write(step['content'])
        else:
             message_placeholder.markdown("Error: RAG Agent not initialized.")
             st.session_state.messages.append({
                "role": "assistant",
                "content": "Error: RAG Agent not initialized."
             })