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
    GraphRAG: A class implementing Graph-based Retrieval Augmented Generation using LlamaIndex
    with Groq LLM and ChromaDB vector store
    
    This class provides functionality to:
    1. Load and process documents from various sources
    2. Build a knowledge graph from the documents
    3. Create vector indices with ChromaDB for efficient retrieval
    4. Query the system using graph-based context with Groq LLM
    5. Skip embedding process if documents were already processed
    """
    
    def __init__(
        self,
        llm_model: str = "llama3-70b-8192",  # Groq model
        embed_model: str = "nomic-ai/nomic-embed-text-v1.5",  # Compatible embedding model
        persist_dir: str = "./storage",
        temperature: float = 0.1,
        chroma_collection_name: str = "graph_rag",
        chroma_dir: str = "./chroma_db"
    ):
        """
        Initialize the GraphRAG system
        
        Args:
            llm_model: The Groq LLM model to use for generation
            embed_model: The embedding model to use for vector representations
            persist_dir: Directory to persist graph indices
            temperature: Temperature parameter for the LLM
            chroma_collection_name: Name of the ChromaDB collection
            chroma_dir: Directory to persist ChromaDB
        """
        # Initialize Groq LLM
        self.llm = Groq(model=llm_model, temperature=temperature)
        
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
        self.chroma_collection_name = chroma_collection_name
        self.chroma_dir = Path(chroma_dir)
        self.chroma_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize ChromaDB client and collection
        self.chroma_client = chromadb.PersistentClient(path=str(self.chroma_dir))
        try:
            self.chroma_collection = self.chroma_client.get_or_create_collection(
                name=self.chroma_collection_name
            )
            logger.info(f"ChromaDB collection '{self.chroma_collection_name}' initialized")
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {e}")
            raise
        
        # Create ChromaVectorStore using the collection
        self.vector_store = ChromaVectorStore(
            chroma_collection=self.chroma_collection
        )
        
        # Initialize graph store for knowledge graph
        self.graph_store = SimpleGraphStore()
        
        # Create storage context with ChromaDB vector store and graph store
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store,
            graph_store=self.graph_store
        )
        
        # Initialize text splitter for document processing
        self.node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
        
        # Initialize indices
        self.kg_index = None
        self.vector_index = None
        self.composable_graph = None
        
        # Flag to track if data has been loaded
        self.data_loaded = False
        
        logger.info("GraphRAG system initialized with Groq LLM and ChromaDB")
    
    def calculate_data_fingerprint(self, data_dir: str) -> str:
        """
        Calculate a fingerprint for the data directory to detect changes
        
        Args:
            data_dir: Path to data directory
            
        Returns:
            Fingerprint string representing the data state
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
        Load documents from a directory
        
        Args:
            data_dir: Path to the directory containing documents
            
        Returns:
            List of Document objects
        """
        logger.info(f"Loading data from directory: {data_dir}")
        reader = SimpleDirectoryReader(data_dir)
        documents = reader.load_data()
        logger.info(f"Loaded {len(documents)} documents")
        return documents
    
    def build_knowledge_graph(self, documents: List[Document]) -> KnowledgeGraphIndex:
        """
        Build a knowledge graph from documents
        
        Args:
            documents: List of Document objects
            
        Returns:
            Knowledge Graph Index
        """
        logger.info("Building knowledge graph...")
        
        # Create knowledge graph index
        self.kg_index = KnowledgeGraphIndex.from_documents(
            documents,
            storage_context=self.storage_context,
            max_triplets_per_chunk=10,
            include_embeddings=True,
        )
        
        # logger.info(f"Knowledge graph built with {len(self.graph_store.get_all_nodes())} nodes")
        return self.kg_index
    
    def build_vector_index(self, documents: List[Document]) -> VectorStoreIndex:
        """
        Build a vector index from documents for efficient retrieval using ChromaDB
        
        Args:
            documents: List of Document objects
            
        Returns:
            Vector Store Index
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
        Create a composable graph combining KG and vector indices
        
        Returns:
            Composable Graph for hybrid retrieval
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
        Create a query engine for graph-based retrieval
        
        Args:
            hyde_enabled: Whether to enable HyDE (Hypothetical Document Embeddings)
            
        Returns:
            Query engine for retrieval
        """
        logger.info("Creating query engine...")
        
        # Get hybrid retriever that combines KG and vector retrieval
        kg_retriever = self.kg_index.as_retriever(
            similarity_top_k=3,
            include_text=True
        )
        
        # Create base query engine
        query_engine = RetrieverQueryEngine.from_args(
            retriever=kg_retriever,
            node_postprocessors=[],
            response_mode="compact"
        )
        
        # Add HyDE query transformation if enabled
        if hyde_enabled:
            hyde_transform = HyDEQueryTransform(
                include_original=True,
                llm=self.llm
            )
            query_engine = TransformQueryEngine(
                query_engine=query_engine,
                query_transform=hyde_transform  # Changed from 'transform' to 'query_transform'
            )
            logger.info("HyDE query transformation enabled")
        
        logger.info("Query engine created")
        return query_engine
    
    def process_and_index_data(self, data_dir: str, force_reindex: bool = False) -> bool:
        """
        Process data from a directory and build all indices.
        Skip processing if data hasn't changed since last run.
        
        Args:
            data_dir: Path to directory containing documents
            force_reindex: Force reindexing even if fingerprint matches
            
        Returns:
            Boolean indicating if new indexing was performed
        """
        # Calculate fingerprint of data directory
        current_fingerprint = self.calculate_data_fingerprint(data_dir)
        previous_fingerprint = self.data_fingerprint.get(data_dir, "")
        
        # Check if we need to reprocess the data
        if current_fingerprint == previous_fingerprint and not force_reindex:
            logger.info("Data unchanged since last indexing. Skipping processing.")
            # Try to load existing indices
            try:
                self.load_indices_from_disk()
                self.data_loaded = True
                return False
            except Exception as e:
                logger.warning(f"Failed to load existing indices, will reindex: {e}")
        
        # Load and process documents
        documents = self.load_data_from_directory(data_dir)
        
        # Build indices only if they are not already loaded
        if not self.kg_index:
            self.build_knowledge_graph(documents)
        else:
            logger.info("Knowledge graph already built. Skipping.")
        
        if not self.vector_index:
            self.build_vector_index(documents)
        else:
            logger.info("Vector index already built. Skipping.")
        
        if not self.composable_graph:
            self.create_composable_graph()
        else:
            logger.info("Composable graph already created. Skipping.")
        
        # Update and save fingerprint
        self.data_fingerprint[data_dir] = current_fingerprint
        with open(self.fingerprint_file, 'w') as f:
            json.dump(self.data_fingerprint, f)
        
        logger.info("Data processing and indexing complete")
        self.data_loaded = True
        
        # Persist the indices
        self.persist_indices()
        
        return True
    
    def query(self, query_text: str, response_mode: str = "compact") -> Dict[str, Any]:
        """
        Query the system with graph-based retrieval using Groq LLM.
        Ensures indices are loaded before querying.
        
        Args:
            query_text: The query text
            response_mode: Response formatting mode
            
        Returns:
            Response dictionary containing answer and context
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
        Persist graph indices to disk
        Note: ChromaDB is already persisted automatically
        """
        logger.info(f"Persisting graph indices to {self.persist_dir}")
        
        # Save KG index
        if self.kg_index:
            kg_path = self.persist_dir / "kg_index"
            kg_path.mkdir(exist_ok=True, parents=True)
            self.kg_index.set_index_id("kg_index")  # Set explicit index ID
            self.kg_index.storage_context.persist(persist_dir=str(kg_path))
            logger.info("Knowledge graph index persisted")
        
        # Save vector index metadata (ChromaDB handles the vectors)
        if self.vector_index:
            vector_path = self.persist_dir / "vector_index"
            vector_path.mkdir(exist_ok=True, parents=True)
            self.vector_index.set_index_id("vector_index")  # Set explicit index ID
            self.vector_index.storage_context.persist(persist_dir=str(vector_path))
            logger.info("Vector index metadata persisted")
        
        logger.info("Graph indices persisted successfully")
        logger.info("ChromaDB vector store is automatically persisted")
    
    def load_indices_from_disk(self) -> None:
        """
        Load indices from disk and reconnect to ChromaDB
        """
        logger.info(f"Loading indices from {self.persist_dir}")
        
        # Reconnect to ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=str(self.chroma_dir))
        try:
            self.chroma_collection = self.chroma_client.get_collection(name=self.chroma_collection_name)
            self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
            logger.info(f"Reconnected to ChromaDB collection '{self.chroma_collection_name}'")
        except Exception as e:
            logger.error(f"Error reconnecting to ChromaDB: {e}")
            raise
        
        try:
            # Load knowledge graph index
            kg_path = self.persist_dir / "kg_index"
            if kg_path.exists():
                # Create storage context with both stores
                storage_context = StorageContext.from_defaults(
                    persist_dir=str(kg_path),
                    vector_store=self.vector_store,
                    graph_store=self.graph_store
                )
                
                # Load KG index with explicit index ID
                self.kg_index = load_index_from_storage(
                    storage_context,
                    index_id="kg_index"
                )
                logger.info("Knowledge graph index loaded")
            else:
                raise FileNotFoundError(f"Knowledge graph index not found at {kg_path}")
            
            # Load vector index from ChromaDB
            vector_path = self.persist_dir / "vector_index"
            if vector_path.exists():
                # Create storage context for vector index
                vector_storage_context = StorageContext.from_defaults(
                    persist_dir=str(vector_path),
                    vector_store=self.vector_store
                )
                self.vector_index = load_index_from_storage(
                    vector_storage_context,
                    index_id="vector_index"
                )
            else:
                # If no persisted metadata, create fresh from vector store
                self.vector_index = VectorStoreIndex.from_vector_store(
                    self.vector_store,
                    index_id="vector_index"
                )
            logger.info("Vector index loaded")
            
            # Update storage context
            self.storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store,
                graph_store=self.graph_store
            )
            
            # Recreate composable graph
            if self.kg_index and self.vector_index:
                self.create_composable_graph()
            
            logger.info("All indices loaded successfully")
            self.data_loaded = True
            
        except Exception as e:
            logger.error(f"Error loading indices: {e}")
            raise


