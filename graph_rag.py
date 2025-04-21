# GraphRAG Implementation using LlamaIndex with Groq and ChromaDB
# With ability to skip embedding for already processed documents

import os
import logging
import hashlib
import json
from typing import List, Dict, Any, Optional
from pathlib import Path

# LlamaIndex imports
from llama_index.core import (
    SimpleDirectoryReader,
    Document,
    Settings,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage
)
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.indices.knowledge_graph import KnowledgeGraphIndex
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.indices.composability import ComposableGraph
from llama_index.core.query_engine import (
    RetrieverQueryEngine,
    TransformQueryEngine
)

# Groq integration
from llama_index.llms.groq.base import Groq

# ChromaDB integration
from llama_index.vector_stores.chroma.base import ChromaVectorStore
import chromadb

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

class GraphRAG:
    """
    GraphRAG: A class implementing Graph-based Retrieval Augmented Generation
    Flow of operations:
    1. Initialize -> Sets up LLM, embedding model, and storage
    2. process_and_index_data -> Main entry point for processing documents
    3. load_data_from_directory -> Reads documents
    4. build_knowledge_graph -> Creates knowledge graph from documents
    5. build_vector_index -> Creates vector embeddings in ChromaDB
    6. create_composable_graph -> Combines KG and vector indices
    7. query -> Used for actual querying after setup
    """
    
    def __init__(
        self,
        llm_model: str = "llama3-70b-8192",  # Groq model
        embed_model: str = "nomic-ai/nomic-embed-text-v1.5",  # Compatible embedding model
        persist_dir: str = "./storage",
        temperature: float = 0.1,
        chroma_dir: str = "./chroma_db",
        system_prompt: str = "You are a helpful AI assistant that provides accurate, relevant information based on the provided context."
    ):
        """
        Step 1: Initialization
        - Sets up Groq LLM for text generation
        - Initializes embedding model for vector representations
        - Creates storage directories and ChromaDB connection
        - Loads any existing document fingerprints
        """
        # Initialize Groq LLM
        self.llm = Groq(model=llm_model, temperature=temperature)
        self.system_prompt = system_prompt
        
        # We'll use a compatible embedding model
        from llama_index.embeddings.huggingface.base import HuggingFaceEmbedding
        self.embed_model = HuggingFaceEmbedding()
        
        # Set global settings
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        
        # Storage settings
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(exist_ok=True, parents=True)
        
        # Data fingerprint file to track processed documents
        self.fingerprint_file = self.persist_dir / "data_fingerprint.json"
        self.data_fingerprint = {}
        if self.fingerprint_file.exists():
            try:
                with open(self.fingerprint_file, 'r') as f:
                    self.data_fingerprint = json.load(f)
                logger.info(f"Loaded data fingerprints for {len(self.data_fingerprint)} documents")
            except Exception as e:
                logger.warning(f"Could not load data fingerprints: {e}")
        
        # ChromaDB setup
        self.chroma_dir = Path(chroma_dir)
        self.chroma_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path=str(self.chroma_dir))
        
        # Initialize graph store for knowledge graph
        self.graph_store = SimpleGraphStore()
        
        # Create storage context with ChromaDB vector store and graph store
        self.storage_context = None
        
        # Initialize text splitter for document processing
        self.node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
        
        # Initialize indices
        self.kg_index = None
        self.vector_index = None
        self.composable_graph = None
        
        # Flag to track if data has been loaded
        self.data_loaded = False
        
        # Add project tracking
        self.current_project = None
        self.projects = set()
        self.project_collections = {}  # Add this to track sanitized collection names
        self._load_existing_projects()
        
        logger.info("GraphRAG system initialized with Groq LLM and ChromaDB")
    
    def _load_existing_projects(self):
        """
        Step: Project Loading
        - Scans storage directory for existing projects
        - Checks for kg_index subdirectory to validate project
        - Populates self.projects set with valid project names
        - Called during initialization to track existing projects
        """
        if self.persist_dir.exists():
            for item in self.persist_dir.iterdir():
                if item.is_dir() and (item / "kg_index").exists():
                    self.projects.add(item.name)
        logger.info(f"Found existing projects: {self.projects}")
    
    def _sanitize_collection_name(self, project_name: str) -> str:
        """
        Step: Collection Name Sanitization
        - Converts project name to valid ChromaDB collection name
        - Removes special characters and spaces
        - Ensures name starts with a letter
        - Returns sanitized name compliant with ChromaDB requirements
        """
        sanitized = ''.join(c if c.isalnum() else '_' for c in project_name)
        if not sanitized[0].isalpha():
            sanitized = 'p_' + sanitized
        return sanitized

    def set_current_project(self, project_name: str):
        """
        Step: Project Context Management
        - Switches active project context
        - Creates/loads project-specific ChromaDB collection
        - Initializes fresh storage context for the project
        - Attempts to load existing indices for the project
        - Sets up vector store and graph store for the project
        """
        self.current_project = project_name
        if project_name not in self.projects:
            self.projects.add(project_name)
        
        # Update storage paths for current project
        project_persist_dir = self.persist_dir / project_name
        project_persist_dir.mkdir(exist_ok=True, parents=True)
        
        # Create or get project-specific ChromaDB collection
        sanitized_name = self._sanitize_collection_name(project_name)
        self.project_collections[project_name] = sanitized_name
        
        try:
            # Reset ChromaDB client connection
            self.chroma_client = chromadb.PersistentClient(path=str(self.chroma_dir))
            self.chroma_collection = self.chroma_client.get_or_create_collection(
                name=sanitized_name
            )
            self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
            
            # Reset graph store for new project
            self.graph_store = SimpleGraphStore()
            
            # Create fresh storage context
            self.storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store,
                graph_store=self.graph_store
            )
            
            # Try to load existing indices for this project
            try:
                self.load_indices_from_disk()
                self.data_loaded = True
            except Exception as e:
                logger.warning(f"No existing indices found for project {project_name}: {e}")
                self.data_loaded = False
                self.kg_index = None
                self.vector_index = None
                self.composable_graph = None
                
        except Exception as e:
            logger.error(f"Error setting up project storage: {e}")
            raise
        
        logger.info(f"Set current project to: {project_name}")
    
    def calculate_data_fingerprint(self, data_dir: str) -> str:
        """
        Step: Data Change Detection
        - Generates unique hash for project data directory
        - Includes file paths, sizes, and modification times
        - Used to detect changes in project documents
        - Prevents unnecessary reprocessing of unchanged data
        - Returns MD5 hash string of project state
        """
        data_path = Path(data_dir)
        if not data_path.exists() or not data_path.is_dir():
            return ""
        
        file_info = []
        for file_path in sorted(data_path.glob("**/*")):
            if file_path.is_file():
                # Collect file path, size and modification time
                relative_path = file_path.relative_to(data_path)
                stat = file_path.stat()
                file_info.append((
                    str(relative_path),
                    stat.st_size,
                    stat.st_mtime
                ))
        
        # Create a hash of all file information
        fingerprint = hashlib.md5(json.dumps(file_info).encode()).hexdigest()
        return fingerprint
    
    def load_data_from_directory(self, data_dir: str) -> List[Document]:
        """
        Step 2: Document Loading
        - Called by process_and_index_data
        - Reads all documents from specified directory
        - Converts them into LlamaIndex Document objects
        - First step in the document processing pipeline
        """
        logger.info(f"Loading data from directory: {data_dir}")
        reader = SimpleDirectoryReader(data_dir)
        documents = reader.load_data()
        logger.info(f"Loaded {len(documents)} documents")
        return documents
    
    def build_knowledge_graph(self, documents: List[Document]) -> KnowledgeGraphIndex:
        """
        Step 3: Knowledge Graph Creation
        - Called after load_data_from_directory
        - Extracts relationships and entities from documents
        - Creates a graph structure for contextual understanding
        - Stores graph data in SimpleGraphStore
        """
        logger.info("Building knowledge graph...")
        
        # Create knowledge graph index
        self.kg_index = KnowledgeGraphIndex.from_documents(
            documents,
            storage_context=self.storage_context,
            max_triplets_per_chunk=10,
            include_embeddings=True,
        )
        
        return self.kg_index
    
    def build_vector_index(self, documents: List[Document]) -> VectorStoreIndex:
        """
        Step 4: Vector Index Creation
        - Runs parallel to knowledge graph creation
        - Splits documents into chunks using SentenceSplitter
        - Creates embeddings for each chunk
        - Stores vectors in ChromaDB for similarity search
        """
        logger.info("Building vector index with ChromaDB...")
        
        # Process documents into nodes
        nodes = self.node_parser.get_nodes_from_documents(documents)
        
        # Create vector index with ChromaDB storage
        self.vector_index = VectorStoreIndex(
            nodes,
            storage_context=self.storage_context,
        )
        
        logger.info("Vector index built in ChromaDB")
        return self.vector_index
    
    def create_composable_graph(self) -> ComposableGraph:
        """
        Step 5: Graph Composition
        - Called after both KG and vector indices are built
        - Combines knowledge graph and vector indices
        - Enables hybrid search using both semantic and graph-based retrieval
        - Creates foundation for query processing
        """
        if not self.kg_index or not self.vector_index:
            raise ValueError("Both knowledge graph and vector indices must be built first")
        
        logger.info("Creating composable graph...")
        
        # Use vector index as root and connect KG index as a child
        root_index = self.vector_index
        
        # Create a composable graph
        self.composable_graph = ComposableGraph(
            root_id="vector_index",
            all_indices={
                "vector_index": root_index,
                "kg_index": self.kg_index
            }
        )
        
        logger.info("Composable graph created")
        return self.composable_graph
    
    def create_query_engine(self, hyde_enabled: bool = True) -> RetrieverQueryEngine:
        """
        Step 6: Query Engine Setup
        - Called during query processing
        - Sets up hybrid retrieval system
        - Configures HyDE for better query understanding
        - Creates the engine that will process actual queries
        """
        logger.info("Creating query engine...")
        
        # Get hybrid retriever that combines KG and vector retrieval
        kg_retriever = self.kg_index.as_retriever(
            similarity_top_k=3,
            include_text=True
        )
        
        # Create base query engine with system prompt
        query_engine = RetrieverQueryEngine.from_args(
            retriever=kg_retriever,
            node_postprocessors=[],
            response_mode="compact",
            system_prompt=self.system_prompt
        )
        
        # Add HyDE query transformation if enabled
        if hyde_enabled:
            hyde_transform = HyDEQueryTransform(
                include_original=True,
                llm=self.llm
            )
            query_engine = TransformQueryEngine(
                query_engine=query_engine,
                query_transform=hyde_transform
            )
            logger.info("HyDE query transformation enabled")
        
        logger.info("Query engine created")
        return query_engine
    
    def process_and_index_data(self, data_dir: str, project_name: str, force_reindex: bool = False) -> bool:
        """Modified to handle project-specific processing"""
        self.set_current_project(project_name)
        project_data_dir = Path(data_dir) / project_name
        
        if not project_data_dir.exists():
            raise ValueError(f"Project directory not found: {project_data_dir}")
        
        # Calculate fingerprint of data directory
        current_fingerprint = self.calculate_data_fingerprint(str(project_data_dir))
        previous_fingerprint = self.data_fingerprint.get(str(project_data_dir), "")
        
        # Check if we need to reprocess the data
        if current_fingerprint == previous_fingerprint and not force_reindex:
            logger.info("Project data unchanged. Loading existing indices.")
            try:
                self.load_indices_from_disk()
                self.data_loaded = True
                return False
            except Exception as e:
                logger.warning(f"Failed to load existing indices: {e}")

        # Load and process documents
        documents = self.load_data_from_directory(str(project_data_dir))

        # Build indices
        self.build_knowledge_graph(documents)
        self.build_vector_index(documents)
        self.create_composable_graph()
        
        # Update and save fingerprint
        self.data_fingerprint[str(project_data_dir)] = current_fingerprint
        with open(self.fingerprint_file, 'w') as f:
            json.dump(self.data_fingerprint, f)
        
        logger.info("Data processing and indexing complete")
        self.data_loaded = True
        
        # Persist the indices
        self.persist_indices()
        
        return True
    
    def query(self, query_text: str, response_mode: str = "compact") -> Dict[str, Any]:
        """
        Main Query Function: Called by end users
        - Entry point for all queries after system is set up
        - Ensures indices are loaded
        - Creates query engine if needed
        - Processes query and returns structured response
        """
        logger.info(f"Processing query: {query_text}")
        
        # Ensure data is loaded
        if not self.data_loaded:
            logger.info("No data loaded yet. Attempting to load from disk...")
            try:
                self.load_indices_from_disk()
                self.data_loaded = True
            except Exception as e:
                logger.error(f"Failed to load indices: {e}")
                return {
                    "query": query_text,
                    "response": "ERROR: No data has been loaded or indexed. Please run process_and_index_data() first.",
                    "source_nodes": []
                }
        
        # Create query engine
        query_engine = self.create_query_engine()
        
        # Execute query
        response = query_engine.query(query_text)
        
        logger.info("Query processed")
        
        # Format response
        result = {
            "query": query_text,
            "response": str(response),
            "source_nodes": [node.node.get_content() for node in response.source_nodes] if hasattr(response, "source_nodes") else []
        }
        
        return result
    
    def persist_indices(self) -> None:
        """
        Step: Index Persistence
        - Saves project indices to disk
        - Creates project-specific storage directory
        - Persists knowledge graph index with unique ID
        - Persists vector index with unique ID
        - Ensures project data can be reloaded later
        """
        if not self.current_project:
            raise ValueError("No project selected")
        
        project_dir = self.persist_dir / self.current_project
        project_dir.mkdir(exist_ok=True, parents=True)
        
        logger.info(f"Persisting indices for project {self.current_project}")
        
        if self.kg_index:
            kg_path = project_dir / "kg_index"
            kg_path.mkdir(exist_ok=True, parents=True)
            self.kg_index.set_index_id(f"kg_index_{self.current_project}")
            self.kg_index.storage_context.persist(persist_dir=str(kg_path))
        
        if self.vector_index:
            vector_path = project_dir / "vector_index"
            vector_path.mkdir(exist_ok=True, parents=True)
            self.vector_index.set_index_id(f"vector_index_{self.current_project}")
            self.vector_index.storage_context.persist(persist_dir=str(vector_path))
        
        logger.info(f"Indices persisted for project {self.current_project}")

    def load_indices_from_disk(self) -> None:
        """
        Step: Index Loading
        - Loads project-specific indices from disk
        - Reconnects to project ChromaDB collection
        - Initializes fresh graph store
        - Loads knowledge graph and vector indices
        - Recreates composable graph for querying
        - Validates all required components are loaded
        """
        if not self.current_project:
            raise ValueError("No project selected")
        
        project_dir = self.persist_dir / self.current_project
        
        logger.info(f"Loading indices for project {self.current_project}")
        
        # Reset and reconnect to project-specific ChromaDB collection
        sanitized_name = self.project_collections[self.current_project]
        self.chroma_collection = self.chroma_client.get_collection(name=sanitized_name)
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        
        # Reset graph store
        self.graph_store = SimpleGraphStore()
        
        try:
            kg_path = project_dir / "kg_index"
            if not kg_path.exists():
                raise FileNotFoundError(f"Knowledge graph index not found at {kg_path}")
            
            # Load knowledge graph with fresh storage context
            storage_context = StorageContext.from_defaults(
                persist_dir=str(kg_path),
                vector_store=self.vector_store,
                graph_store=self.graph_store
            )
            
            self.kg_index = load_index_from_storage(
                storage_context,
                index_id=f"kg_index_{self.current_project}"
            )
            
            # Load vector index
            vector_path = project_dir / "vector_index"
            vector_storage_context = StorageContext.from_defaults(
                persist_dir=str(vector_path) if vector_path.exists() else None,
                vector_store=self.vector_store
            )
            
            self.vector_index = load_index_from_storage(
                vector_storage_context,
                index_id=f"vector_index_{self.current_project}"
            )
            
            # Update main storage context
            self.storage_context = storage_context
            
            # Recreate composable graph
            if self.kg_index and self.vector_index:
                self.create_composable_graph()
            
            logger.info(f"Successfully loaded indices for project {self.current_project}")
            self.data_loaded = True
            
        except Exception as e:
            logger.error(f"Error loading indices: {e}")
            raise


