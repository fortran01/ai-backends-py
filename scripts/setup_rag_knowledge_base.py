"""
RAG Knowledge Base Setup Script for Phase 6
 
This script demonstrates:
1. Document loading and preprocessing for RAG
2. Text chunking strategies for optimal retrieval
3. Vector embedding generation using HuggingFace models
4. Vector database setup with ChromaDB
5. Knowledge base indexing and storage

Following the coding guidelines:
- Explicit type annotations for all functions
- Comprehensive documentation and error handling
- Production-ready implementation patterns
"""

import os
import sys
import logging
import uuid
from typing import List, Dict, Any, Optional
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# RAG framework imports
try:
    from llama_index.core import Document, VectorStoreIndex, Settings
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.core.embeddings import BaseEmbedding
    from llama_index.llms.ollama import Ollama
    import chromadb
    from fastembed import TextEmbedding
    from typing import Any, List
    RAG_AVAILABLE = True
except ImportError as e:
    print(f"RAG dependencies not available: {e}")
    print("Please install with: pip install llama-index llama-index-llms-ollama llama-index-vector-stores-chroma chromadb fastembed")
    RAG_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FastEmbedWrapper(BaseEmbedding):
    """
    LlamaIndex wrapper for FastEmbed TextEmbedding model.
    """
    
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name)
    
    @classmethod
    def class_name(cls) -> str:
        return "FastEmbedWrapper"
    
    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)
    
    def _get_query_embedding(self, query: str) -> List[float]:
        embeddings = list(self._model.embed([query]))
        return embeddings[0].tolist()
    
    def _get_text_embedding(self, text: str) -> List[float]:
        return self._get_query_embedding(text)
    
    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = list(self._model.embed(texts))
        return [embedding.tolist() for embedding in embeddings]

class RAGKnowledgeBaseSetup:
    """
    Comprehensive RAG knowledge base setup and management.
    
    Handles document loading, chunking, embedding generation,
    and vector database storage using LlamaIndex and ChromaDB.
    """
    
    def __init__(self, 
                 knowledge_base_path: str = "knowledge_base.md",
                 chroma_db_path: str = "./chroma_db",
                 collection_name: str = "ai_ml_knowledge_base",
                 embedding_model: str = "BAAI/bge-small-en-v1.5",
                 chunk_size: int = 512,
                 chunk_overlap: int = 50) -> None:
        """
        Initialize RAG knowledge base setup.
        
        Args:
            knowledge_base_path: Path to the knowledge base document
            chroma_db_path: Directory for ChromaDB storage
            collection_name: Name for the vector collection
            embedding_model: FastEmbed model name
            chunk_size: Maximum tokens per chunk
            chunk_overlap: Overlap between consecutive chunks
        """
        self.knowledge_base_path: str = knowledge_base_path
        self.chroma_db_path: str = chroma_db_path
        self.collection_name: str = collection_name
        self.embedding_model_name: str = embedding_model
        self.chunk_size: int = chunk_size
        self.chunk_overlap: int = chunk_overlap
        
        # Initialize components
        self.embedding_model: Optional[FastEmbedWrapper] = None
        self.chroma_client: Optional[chromadb.PersistentClient] = None
        self.vector_store: Optional[ChromaVectorStore] = None
        self.index: Optional[VectorStoreIndex] = None
        
        logger.info(f"Initializing RAG setup with knowledge base: {knowledge_base_path}")
        
    def initialize_components(self) -> bool:
        """
        Initialize all RAG components.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            logger.info("Initializing FastEmbed model...")
            self.embedding_model = FastEmbedWrapper(model_name=self.embedding_model_name)
            
            logger.info("Setting up ChromaDB client...")
            self.chroma_client = chromadb.PersistentClient(path=self.chroma_db_path)
            
            # Create or get collection
            try:
                collection = self.chroma_client.get_collection(name=self.collection_name)
                logger.info(f"Found existing collection: {self.collection_name}")
            except Exception:
                collection = self.chroma_client.create_collection(name=self.collection_name)
                logger.info(f"Created new collection: {self.collection_name}")
            
            # Initialize vector store
            self.vector_store = ChromaVectorStore(chroma_collection=collection)
            
            # Configure LlamaIndex global settings with FastEmbed wrapper
            Settings.embed_model = self.embedding_model
            Settings.llm = Ollama(model="tinyllama", base_url="http://localhost:11434")
            
            logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            return False
    
    def load_knowledge_base(self) -> List[Document]:
        """
        Load and preprocess the knowledge base document.
        
        Returns:
            List[Document]: Processed documents ready for chunking
        """
        try:
            knowledge_path = Path(self.knowledge_base_path)
            if not knowledge_path.exists():
                raise FileNotFoundError(f"Knowledge base not found: {self.knowledge_base_path}")
            
            logger.info(f"Loading knowledge base from: {knowledge_path}")
            
            with open(knowledge_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Create document with metadata
            document = Document(
                text=content,
                metadata={
                    "filename": knowledge_path.name,
                    "file_type": "markdown",
                    "source": "comprehensive_ai_ml_knowledge_base",
                    "doc_id": str(uuid.uuid4()),
                    "created_date": knowledge_path.stat().st_mtime
                }
            )
            
            logger.info(f"Loaded document with {len(content)} characters")
            return [document]
            
        except Exception as e:
            logger.error(f"Failed to load knowledge base: {e}")
            return []
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into optimized chunks for retrieval.
        
        Args:
            documents: List of documents to chunk
            
        Returns:
            List[Document]: Chunked documents with metadata
        """
        try:
            logger.info(f"Chunking {len(documents)} documents...")
            
            # Initialize sentence splitter
            splitter = SentenceSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separator=" ",
                paragraph_separator="\n\n",
                secondary_chunking_regex="[.!?]+",
            )
            
            # Process all documents
            all_chunks = []
            for doc in documents:
                chunks = splitter.split_text(doc.text)
                
                # Create Document objects for each chunk
                for i, chunk_text in enumerate(chunks):
                    chunk_doc = Document(
                        text=chunk_text,
                        metadata={
                            **doc.metadata,
                            "chunk_id": i,
                            "total_chunks": len(chunks),
                            "chunk_size": len(chunk_text),
                            "chunk_type": "sentence_split"
                        }
                    )
                    all_chunks.append(chunk_doc)
            
            logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
            
            # Log chunk statistics
            chunk_sizes = [len(chunk.text) for chunk in all_chunks]
            avg_size = sum(chunk_sizes) / len(chunk_sizes)
            min_size = min(chunk_sizes)
            max_size = max(chunk_sizes)
            
            logger.info(f"Chunk statistics - Avg: {avg_size:.1f}, Min: {min_size}, Max: {max_size}")
            
            return all_chunks
            
        except Exception as e:
            logger.error(f"Failed to chunk documents: {e}")
            return []
    
    def create_vector_index(self, chunks: List[Document]) -> bool:
        """
        Create vector index from document chunks.
        
        Args:
            chunks: List of document chunks to index
            
        Returns:
            bool: True if indexing successful, False otherwise
        """
        try:
            logger.info(f"Creating vector index from {len(chunks)} chunks...")
            
            # Create storage context with vector store
            from llama_index.core import StorageContext
            storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            
            # Create index from documents with explicit storage context
            self.index = VectorStoreIndex.from_documents(
                chunks,
                storage_context=storage_context,
                show_progress=True
            )
            
            logger.info("Vector index created successfully")
            
            # Verify documents were actually added to ChromaDB
            if self.chroma_client:
                try:
                    collection = self.chroma_client.get_collection(self.collection_name)
                    doc_count = collection.count()
                    logger.info(f"Verification: ChromaDB collection now has {doc_count} documents")
                    
                    if doc_count == 0:
                        logger.warning("Warning: ChromaDB collection is empty after indexing")
                        return False
                        
                except Exception as verify_e:
                    logger.warning(f"Could not verify ChromaDB document count: {verify_e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create vector index: {e}")
            logger.error(f"Error details: {type(e).__name__}: {str(e)}")
            return False
    
    def test_retrieval(self, test_queries: List[str]) -> Dict[str, Any]:
        """
        Test the retrieval system with sample queries.
        
        Args:
            test_queries: List of test questions
            
        Returns:
            Dict[str, Any]: Test results and performance metrics
        """
        if not self.index:
            logger.error("No index available for testing")
            return {"error": "No index available"}
        
        try:
            logger.info("Testing retrieval system...")
            
            # Create query engine
            query_engine = self.index.as_query_engine(
                similarity_top_k=5,
                response_mode="compact"
            )
            
            results = {}
            total_time = 0
            
            for i, query in enumerate(test_queries):
                logger.info(f"Testing query {i+1}: {query}")
                
                import time
                start_time = time.time()
                
                response = query_engine.query(query)
                
                end_time = time.time()
                query_time = end_time - start_time
                total_time += query_time
                
                results[f"query_{i+1}"] = {
                    "question": query,
                    "response": str(response),
                    "response_time_ms": query_time * 1000,
                    "source_nodes": len(response.source_nodes) if hasattr(response, 'source_nodes') else 0
                }
                
                logger.info(f"Query completed in {query_time:.3f}s")
            
            # Calculate performance metrics
            avg_time = (total_time / len(test_queries)) * 1000
            
            test_summary = {
                "total_queries": len(test_queries),
                "average_response_time_ms": avg_time,
                "total_time_seconds": total_time,
                "queries_per_second": len(test_queries) / total_time if total_time > 0 else 0,
                "results": results
            }
            
            logger.info(f"Retrieval testing completed - Avg time: {avg_time:.1f}ms")
            return test_summary
            
        except Exception as e:
            logger.error(f"Failed to test retrieval: {e}")
            return {"error": str(e)}
    
    def get_index_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the created index.
        
        Returns:
            Dict[str, Any]: Index statistics and metadata
        """
        try:
            if not self.chroma_client:
                return {"error": "No ChromaDB client available"}
            
            collection = self.chroma_client.get_collection(self.collection_name)
            
            stats = {
                "collection_name": self.collection_name,
                "total_documents": collection.count(),
                "embedding_model": self.embedding_model_name,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "database_path": self.chroma_db_path,
                "status": "ready"
            }
            
            # Get sample documents for inspection
            if collection.count() > 0:
                sample_data = collection.peek(limit=3)
                stats["sample_documents"] = [
                    {
                        "id": doc_id,
                        "text_preview": text[:100] + "..." if len(text) > 100 else text,
                        "metadata": metadata
                    }
                    for doc_id, text, metadata in zip(
                        sample_data["ids"],
                        sample_data["documents"],
                        sample_data["metadatas"]
                    )
                ]
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get index statistics: {e}")
            return {"error": str(e)}
    
    def setup_complete_rag_system(self) -> Dict[str, Any]:
        """
        Complete RAG system setup from start to finish.
        
        Returns:
            Dict[str, Any]: Setup results and system status
        """
        logger.info("Starting complete RAG system setup...")
        
        setup_results = {
            "steps": {},
            "success": False,
            "error": None
        }
        
        try:
            # Step 1: Initialize components
            logger.info("Step 1: Initializing components...")
            if not self.initialize_components():
                raise Exception("Component initialization failed")
            setup_results["steps"]["initialize_components"] = True
            
            # Step 2: Load knowledge base
            logger.info("Step 2: Loading knowledge base...")
            documents = self.load_knowledge_base()
            if not documents:
                raise Exception("Knowledge base loading failed")
            setup_results["steps"]["load_knowledge_base"] = True
            setup_results["documents_loaded"] = len(documents)
            
            # Step 3: Chunk documents
            logger.info("Step 3: Chunking documents...")
            chunks = self.chunk_documents(documents)
            if not chunks:
                raise Exception("Document chunking failed")
            setup_results["steps"]["chunk_documents"] = True
            setup_results["chunks_created"] = len(chunks)
            
            # Step 4: Create vector index
            logger.info("Step 4: Creating vector index...")
            if not self.create_vector_index(chunks):
                raise Exception("Vector index creation failed")
            setup_results["steps"]["create_vector_index"] = True
            
            # Step 5: Test retrieval
            logger.info("Step 5: Testing retrieval system...")
            test_queries = [
                "What is machine learning?",
                "Explain the difference between supervised and unsupervised learning",
                "What are the components of a RAG system?",
                "How do neural networks work?",
                "What is model drift in MLOps?"
            ]
            
            test_results = self.test_retrieval(test_queries)
            setup_results["steps"]["test_retrieval"] = True
            setup_results["test_results"] = test_results
            
            # Step 6: Get system statistics
            stats = self.get_index_statistics()
            setup_results["system_statistics"] = stats
            
            setup_results["success"] = True
            logger.info("RAG system setup completed successfully!")
            
        except Exception as e:
            error_msg = f"RAG setup failed: {e}"
            logger.error(error_msg)
            setup_results["error"] = error_msg
            setup_results["success"] = False
        
        return setup_results

def main() -> None:
    """
    Main function to run RAG knowledge base setup.
    """
    if not RAG_AVAILABLE:
        print("RAG dependencies not available. Please install required packages.")
        sys.exit(1)
    
    print("🚀 Starting RAG Knowledge Base Setup for Phase 6...")
    print("=" * 60)
    
    # Initialize setup
    rag_setup = RAGKnowledgeBaseSetup()
    
    # Run complete setup
    results = rag_setup.setup_complete_rag_system()
    
    # Display results
    print("\n📊 Setup Results:")
    print("=" * 40)
    
    if results["success"]:
        print("✅ RAG system setup completed successfully!")
        print(f"📚 Documents loaded: {results.get('documents_loaded', 0)}")
        print(f"📄 Chunks created: {results.get('chunks_created', 0)}")
        
        if "system_statistics" in results:
            stats = results["system_statistics"]
            print(f"🗃️ Total vectors: {stats.get('total_documents', 0)}")
            print(f"🔍 Embedding model: {stats.get('embedding_model', 'N/A')}")
        
        if "test_results" in results and "average_response_time_ms" in results["test_results"]:
            avg_time = results["test_results"]["average_response_time_ms"]
            print(f"⚡ Average query time: {avg_time:.1f}ms")
        
        print("\n🎯 RAG system is ready for use!")
        print("You can now use the /api/v1/rag-chat endpoint for intelligent Q&A.")
        
    else:
        print("❌ RAG system setup failed!")
        if results.get("error"):
            print(f"Error: {results['error']}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()