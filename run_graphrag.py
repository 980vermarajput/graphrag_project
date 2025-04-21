import os
from pathlib import Path
from dotenv import load_dotenv
from graph_rag import GraphRAG
import questionary

# Load environment variables
load_dotenv()

# Get API key from .env
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")

# Initialize GraphRAG
graph_rag = GraphRAG(
    llm_model="llama3-70b-8192",
    embed_model="nomic-ai/nomic-embed-text-v1.5",
    chroma_dir="./chroma_db",
    persist_dir="./storage"
)

# Scan data directory for projects
data_dir = Path("./data")
projects = [d.name for d in data_dir.iterdir() if d.is_dir()]

if not projects:
    print("No project folders found in ./data directory!")
    exit(1)

print("Available projects:", projects)

# Process each project
for project in projects:
    print(f"\nProcessing project: {project}")
    try:
        graph_rag.process_and_index_data("./data", project)
    except Exception as e:
        print(f"Error processing project {project}: {e}")
        print("Skipping to next project...")
        continue

# Interactive query loop
print("\nGraphRAG system is ready! Type 'exit' to quit.")
while True:
    # Project selection using questionary
    choices = projects + ['exit']
    project = questionary.select(
        "Select project:",
        choices=choices,
        use_indicator=True,
    ).ask()
    
    if project == 'exit' or project is None:
        break
    
    try:
        # Set current project context and load its data
        graph_rag.set_current_project(project)
        
        # Query loop for selected project
        while True:
            query = input(f"\n[{project}] Enter your question (or 'back' to switch project): ")
            if query.lower() == 'back':
                break
            if query.lower() == 'exit':
                exit(0)
            
            try:
                response = graph_rag.query(query)
                print("\nResponse:", response["response"])
                # if response.get("source_nodes"):
                #     print("\nSources:")
                #     for i, source in enumerate(response["source_nodes"], 1):
                #         print(f"{i}. {source[:200]}...")
            except Exception as e:
                print(f"Error processing query: {e}")
                
    except Exception as e:
        print(f"Error switching to project {project}: {e}")
        continue