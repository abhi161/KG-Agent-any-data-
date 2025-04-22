# app.py
import streamlit as st
import json
import os
from dotenv import load_dotenv
import logging
# Import your RAG agent and initialization code
from rag_agent import KnowledgeGraphRAGAgent # Assuming rag_agent.py is in the same directory

# Assuming your config setup is reusable
from unified_kg.config import Config
from unified_kg.core.schema_manager import SchemaManager
from langchain_neo4j import Neo4jGraph

logger =logging.getLogger(__name__)
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
        if llm is None: # Check if LLM failed init
            st.error("LLM initialization failed. Cannot proceed.")
            st.stop()
        graph = Neo4jGraph(
            url=config_obj.neo4j_uri,
            username=config_obj.neo4j_user,
            password=config_obj.neo4j_password,
            database=config_obj.neo4j_database
        )
        graph.query("RETURN 1") # Verify connection

        # --- Load Schema ---
        schema_path = config_obj.initial_schema_path
        schema_definition = "Error: Schema could not be processed." # Default error message
        schema_str_for_agent = "Schema not available" # Fallback for agent init
        st.info(f"Attempting to load schema from: {schema_path}")

        if schema_path and os.path.exists(schema_path):
            try:
                with open(schema_path, 'r', encoding='utf-8') as f: # Added encoding
                    schema_data = json.load(f) # Load JSON data

                # --- Check if loaded data is a dictionary ---
                if not isinstance(schema_data, dict):
                    st.error(f"Schema file '{schema_path}' did not contain valid JSON (expected a dictionary, got {type(schema_data)}).")
                    schema_definition = "Error: Schema file is not a valid JSON object."
                    st.stop() # Stop if schema is fundamentally wrong

                # --- Format the schema dictionary into a string ---
                # Change this section in your app.py
                schema_str = "Node Types & Properties:\n"
                for et in schema_data.get("entity_types", []):
                    if isinstance(et, dict):
                        # Handle properties as simple strings, not dictionaries
                        props = ", ".join(et.get("properties", [])) if et.get("properties") else "None"
                        schema_str += f"- {et.get('name', 'Type Name')} ({props})\n"
                    else: 
                        logger.warning(f"Invalid entity type format in schema: {et}")


                schema_str += "\nRelationship Types (Source -> Target):\n"
                for rt in schema_data.get("relation_types", []):
                     if isinstance(rt, dict):
                         src = ",".join(rt.get("source_types", ["Any"]))
                         tgt = ",".join(rt.get("target_types", ["Any"]))
                         schema_str += f"- {rt.get('name', 'Rel Name')} ({src} -> {tgt})\n"
                     else: logger.warning(f"Invalid relation type format in schema: {rt}")

                schema_definition = schema_str # Store the formatted string for display/logging
                schema_str_for_agent = schema_definition # Use the formatted string for the agent

            except json.JSONDecodeError as json_e:
                st.error(f"Error parsing schema file '{schema_path}' as JSON: {json_e}. Please ensure the file is valid JSON.")
                schema_definition = f"Error: Invalid JSON in schema file: {json_e}"
                st.stop() # Stop if JSON is invalid
            except Exception as e:
                st.error(f"An unexpected error occurred loading/processing schema from {schema_path}: {e}")
                schema_definition = f"Error loading/processing schema: {e}"
                st.stop() # Stop on other unexpected errors
        else:
            st.error(f"Schema file not found at configured path: {schema_path}. Cannot initialize RAG agent correctly.")
            st.stop() # Stop if schema file is missing

        # --- Instantiate Agent ---
        # Pass the graph object. The schema_str_for_agent is now just for reference in the agent class if needed.
        agent = KnowledgeGraphRAGAgent(llm, graph, schema_str_for_agent)
        return agent

    except Exception as e:
        st.error(f"Initialization Failed during setup: {e}")
        logger.error(f"Full Initialization Traceback:", exc_info=True) # Log full trace
        st.stop()
        return None

rag_agent = initialize_rag_agent()

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Display intermediate steps if they exist (for assistant messages)
        if message["role"] == "assistant" and "intermediate_steps" in message:
             with st.expander("Show Retrieval Process"):
                for step in message["intermediate_steps"]:
                     st.write(f"**{step['type']}:**")
                     if isinstance(step['content'], list):
                          st.json(step['content'], expanded=False) # Display lists/dicts as JSON
                     elif isinstance(step['content'], str) and step['type'] == 'Generated Cypher':
                          st.code(step['content'], language='cypher')
                     else:
                          st.write(step['content'])


# Accept user input
if prompt := st.chat_input("Ask a question about the knowledge graph..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display thinking indicator
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")

        # Get assistant response
        if rag_agent:
            response_data = rag_agent.run_query(prompt)
            assistant_response = response_data["result"]
            intermediate_steps = response_data["intermediate_steps"]

            # Update the placeholder with the final response
            message_placeholder.markdown(assistant_response)

            # Add assistant response to chat history (including intermediates)
            st.session_state.messages.append({
                "role": "assistant",
                "content": assistant_response,
                "intermediate_steps": intermediate_steps # Store for display
            })
            # Display intermediate steps right after the response is shown
            with st.expander("Show Retrieval Process", expanded=True): # Expand immediately
                 for step in intermediate_steps:
                      st.write(f"**{step['type']}:**")
                      if isinstance(step['content'], list):
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