#!/usr/bin/env python3
"""
Quick RAG setup script to populate ChromaDB with knowledge base content
"""
import chromadb
from pathlib import Path

def setup_rag_quickly():
    """Setup RAG system with minimal configuration"""
    print("🚀 Quick RAG setup starting...")
    
    # Initialize ChromaDB client
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection_name = "ai_ml_knowledge_base"
    
    # Get or create collection
    try:
        collection = chroma_client.get_collection(name=collection_name)
        print(f"✅ Found existing collection: {collection_name}")
    except:
        collection = chroma_client.create_collection(name=collection_name)
        print(f"✅ Created new collection: {collection_name}")
    
    # Clear existing documents if any exist
    if collection.count() > 0:
        # Get all document IDs and delete them
        all_data = collection.get()
        if all_data['ids']:
            collection.delete(ids=all_data['ids'])
            print("🧹 Cleared existing documents")
        else:
            print("📭 Collection was already empty")
    
    # Load knowledge base
    kb_path = Path("knowledge_base.md")
    if not kb_path.exists():
        print("❌ knowledge_base.md not found!")
        return False
    
    with open(kb_path, 'r') as f:
        content = f.read()
    
    print(f"📚 Loaded knowledge base: {len(content)} characters")
    
    # Simple chunking - split by double newlines (paragraphs)
    chunks = [chunk.strip() for chunk in content.split('\n\n') if chunk.strip()]
    
    # Filter out very short chunks
    chunks = [chunk for chunk in chunks if len(chunk) > 100]
    
    print(f"📄 Created {len(chunks)} chunks")
    
    # Add documents to collection
    documents = []
    metadatas = []
    ids = []
    
    for i, chunk in enumerate(chunks):
        documents.append(chunk)
        metadatas.append({
            "chunk_id": i,
            "source": "knowledge_base.md",
            "chunk_length": len(chunk)
        })
        ids.append(f"chunk_{i}")
    
    # Add to ChromaDB
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    
    print(f"✅ Added {len(documents)} documents to ChromaDB")
    print(f"📊 Collection now has {collection.count()} documents")
    
    # Test retrieval
    results = collection.query(
        query_texts=["What is machine learning?"],
        n_results=2
    )
    
    if results['documents'] and results['documents'][0]:
        print("🔍 Test query successful:")
        print(f"   Found {len(results['documents'][0])} relevant documents")
        print(f"   Sample result: {results['documents'][0][0][:200]}...")
    else:
        print("⚠️  Test query returned no results")
    
    print("✅ Quick RAG setup completed!")
    return True

if __name__ == "__main__":
    setup_rag_quickly()