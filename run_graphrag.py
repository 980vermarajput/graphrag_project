import os
from dotenv import load_dotenv
from graph_rag import GraphRAG

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

# Process data - will skip if already processed
graph_rag.process_and_index_data("./data")

# Interactive query loop
print("GraphRAG system is ready! Type 'exit' to quit.")
while True:
    query = input("\nEnter your question: ")
    if query.lower() == 'exit':
        break
    
    response = graph_rag.query(query)
    
    print("\nResponse:", response["response"])
    # print("\nSources:")
    # for i, source in enumerate(response["source_nodes"][:3]):  # Show first 3 sources
    #     print(f"{i+1}. {source[:150]}...")  # First 150 chars of each source