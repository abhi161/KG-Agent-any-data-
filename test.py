from neo4j import GraphDatabase

# Test connection
uri = "bolt://localhost:7687"
username = "neo4j"
password = "12345678"  # Change to your actual password

try:
    with GraphDatabase.driver(uri, auth=(username, password)) as driver:
        with driver.session() as session:
            result = session.run("RETURN 'Connection successful!' AS message")
            print(result.single()["message"])
except Exception as e:
    print(f"Connection failed: {e}")


# import os
# import sys
# import json
# from unittest.mock import MagicMock, patch

# # Make sure we can import from the unified_kg package
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# # Set environment variables for testing
# os.environ["AZURE_OPENAI_KEY"] = "test_key"
# os.environ["AZURE_OPENAI_ENDPOINT"] = "https://test.openai.azure.com"
# os.environ["AZURE_OPENAI_DEPLOYMENT"] = "test-deployment"
# os.environ["AZURE_OPENAI_API_VERSION"] = "2023-05-15"
# os.environ["NEO4J_URI"] = "bolt://localhost:7687"
# os.environ["NEO4J_USER"] = "neo4j"
# os.environ["NEO4J_PASSWORD"] = "password"

# # Create mock data
# os.makedirs("data/csv", exist_ok=True)
# os.makedirs("data/pdf", exist_ok=True)
# os.makedirs("schemas", exist_ok=True)

# # Create a sample CSV
# with open("data/csv/test.csv", "w") as f:
#     f.write("name,company,title\nJohn Doe,Acme Inc,CEO\nJane Smith,XYZ Corp,CTO")

# # Create a sample text file (representing a PDF)
# with open("data/pdf/test.txt", "w") as f:
#     f.write("Acme Inc is a leading company in the industry. John Doe is the CEO.")

# # Create a sample schema
# schema = {
#   "entity_types": [
#     {
#       "name": "Person",
#       "description": "A person entity",
#       "properties": ["name", "title"]
#     },
#     {
#       "name": "Company",
#       "description": "A company entity",
#       "properties": ["name", "industry"]
#     }
#   ],
#   "relation_types": [
#     {
#       "name": "WORKS_FOR",
#       "description": "Person works for a company",
#       "source_types": ["Person"],
#       "target_types": ["Company"]
#     }
#   ]
# }

# with open("schemas/test_schema.json", "w") as f:
#     json.dump(schema, f, indent=2)

# # Import our config class first to avoid the mock patching issues
# from unified_kg.config import Config

# # Now mock the dependencies - use the correct import paths!
# with patch("langchain_openai.AzureChatOpenAI") as mock_azure_openai, \
#      patch("langchain_community.graphs.Neo4jGraph") as mock_neo4j:
    
#     # Mock Azure OpenAI
#     mock_llm = MagicMock()
#     mock_llm.predict.return_value = "TEST_RESPONSE"
#     mock_azure_openai.return_value = mock_llm
    
#     # Mock Neo4j
#     mock_graph = MagicMock()
#     mock_graph.query.return_value = [{"id": 1, "n": {"name": "test"}}]
#     mock_neo4j.return_value = mock_graph
    
#     # Now import the function we want to test
#     from unified_kg.main import build_knowledge_graph
    
#     # Run the test
#     try:
#         result = build_knowledge_graph(
#             csv_files=["data/csv/test.csv"],
#             pdf_files=["data/pdf/test.txt"],
#             initial_schema_path="schemas/test_schema.json"
#         )
        
#         print("Test completed successfully!")
#         print(json.dumps(result, indent=2))
#     except Exception as e:
#         print(f"Test failed: {e}")
#         import traceback
#         traceback.print_exc()