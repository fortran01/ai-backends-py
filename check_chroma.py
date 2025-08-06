#!/usr/bin/env python3
"""
Quick script to check ChromaDB collection status
"""
import chromadb

def check_chromadb():
    """Check ChromaDB collection status"""
    try:
        # Initialize ChromaDB client
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
        # Try to get the collection
        try:
            collection = chroma_client.get_collection(name="ai_ml_knowledge_base")
            document_count = collection.count()
            
            print(f"✅ Found collection 'ai_ml_knowledge_base' with {document_count} documents")
            
            if document_count > 0:
                # Get a sample document
                sample_data = collection.peek(limit=1)
                if sample_data['documents']:
                    print(f"📄 Sample document preview: {sample_data['documents'][0][:100]}...")
                    print(f"🏷️  Sample metadata: {sample_data['metadatas'][0]}")
                else:
                    print("⚠️  Collection exists but no documents found")
            else:
                print("❌ Collection is empty")
                
        except Exception as e:
            print(f"❌ Collection not found or error accessing it: {e}")
            
            # List available collections
            collections = chroma_client.list_collections()
            if collections:
                print(f"Available collections: {[c.name for c in collections]}")
            else:
                print("No collections found in database")
                
    except Exception as e:
        print(f"❌ Error connecting to ChromaDB: {e}")

if __name__ == "__main__":
    print("🔍 Checking ChromaDB Status")
    print("=" * 50)
    check_chromadb()