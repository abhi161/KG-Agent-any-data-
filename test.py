from neo4j import GraphDatabase

# Test connection
uri = "bolt://localhost:7"
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


import os
from pathlib import Path
import argparse

def generate_tree(directory_path, prefix="", ignore_patterns=None, ignore_extensions=None):
    """
    Generate a tree structure of the given directory.
    
    Args:
        directory_path (str): Path to the directory
        prefix (str): Prefix for the current item (used for indentation)
        ignore_patterns (list): List of directories/files to ignore
        ignore_extensions (list): List of file extensions to ignore (e.g., ['.pyc', '.log'])
    """
    if ignore_patterns is None:
        ignore_patterns = ['.git', 'node_modules', '__pycache__', '.pytest_cache', '.vscode']
    if ignore_extensions is None:
        ignore_extensions = []
        
    # Get the directory contents
    path = Path(directory_path)
    
    # Get directories and files separately and sort them
    try:
        items = list(path.iterdir())
        directories = sorted([item for item in items if item.is_dir()])
        files = sorted([item for item in items if item.is_file()])
    except PermissionError:
        return f"{prefix}├── [Permission Denied]\n"
    
    tree = ""
    
    # Process directories
    for i, directory in enumerate(directories):
        # Skip if directory matches any ignore pattern
        if (directory.name.startswith('.') or 
            any(pattern.lower() in str(directory).lower() for pattern in ignore_patterns)):
            continue
            
        is_last_dir = (i == len(directories) - 1 and len(files) == 0)
        connector = "└── " if is_last_dir else "├── "
        
        tree += f"{prefix}{connector}{directory.name}/\n"
        extension = "    " if is_last_dir else "│   "
        tree += generate_tree(directory, prefix + extension, ignore_patterns, ignore_extensions)
    
    # Process files
    for i, file in enumerate(files):
        # Skip if file matches any ignore pattern or extension
        if (file.name.startswith('.') or 
            any(pattern.lower() in file.name.lower() for pattern in ignore_patterns) or
            any(file.name.lower().endswith(ext.lower()) for ext in ignore_extensions)):
            continue
            
        is_last = (i == len(files) - 1)
        connector = "└── " if is_last else "├── "
        tree += f"{prefix}{connector}{file.name}\n"
    
    return tree

def main():
    parser = argparse.ArgumentParser(description='Generate a tree structure of a directory')
    parser.add_argument('path', nargs='?', default='.', help='Path to the directory (default: current directory)')
    parser.add_argument('--ignore', nargs='*', help='Additional patterns to ignore (folders or files)')
    parser.add_argument('--ignore-ext', nargs='*', help='File extensions to ignore (e.g., .pyc .log)')
    # python -m unified_kg.main --csv test_data/csv/patients.csv test_data/csv/medications.csv test_data/csv/prescriptions.csv test_data/csv/doctors.csv --pdf  test_data/pdf/research_paper.pdf test_data/pdf/drug_interactions.pdf --schema test_data/medical_schema.json --output results.json
    args = parser.parse_args()
    
    # Default ignore patterns
    ignore_patterns = ['llm_provider.py','test.py','schemas','kenv','.git', 'node_modules', '__pycache__', '.pytest_cache','Dockerfile','docker-compose.yml','.dockerignore', '.vscode']
    
    # Add user-provided patterns
    if args.ignore:
        ignore_patterns.extend(args.ignore)
    
    # Process ignore extensions
    ignore_extensions = []
    if args.ignore_ext:
        ignore_extensions = [ext if ext.startswith('.') else f'.{ext}' for ext in args.ignore_ext]
    
    # Get the absolute path
    directory_path = os.path.abspath(args.path)
    
    # Print the root directory name
    print(f"\nProject Structure for: {directory_path}")
    # print(f"Ignoring: {', '.join(ignore_patterns)}")
    if ignore_extensions:
        print(f"Ignoring extensions: {', '.join(ignore_extensions)}")
    print()
    
    # Generate and print the tree
    tree = generate_tree(directory_path, ignore_patterns=ignore_patterns, ignore_extensions=ignore_extensions)
    print(tree)

if __name__ == "__main__":
    main()