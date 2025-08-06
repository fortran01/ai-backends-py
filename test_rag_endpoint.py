#!/usr/bin/env python3
"""
Quick test script to verify RAG endpoint functionality
"""
import requests
import json
import time

def test_rag_endpoint():
    """Test the RAG endpoint with a sample query"""
    base_url = "http://localhost:5001"
    endpoint = f"{base_url}/api/v1/rag-chat"
    
    # Test query
    payload = {
        "query": "What is machine learning?"
    }
    
    print(f"Testing RAG endpoint: {endpoint}")
    print(f"Query: {payload['query']}")
    print("-" * 50)
    
    try:
        response = requests.post(
            endpoint,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ RAG endpoint working successfully!")
            print(f"Response: {result.get('response', 'No response field')}")
            print(f"Sources: {len(result.get('source_nodes', []))} documents retrieved")
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Flask server. Is it running?")
    except requests.exceptions.Timeout:
        print("❌ Request timed out (>30s)")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def check_server_health():
    """Check if Flask server is running"""
    try:
        response = requests.get("http://localhost:5001/health", timeout=5)
        print(f"✅ Server health check: {response.status_code}")
        return True
    except:
        print("❌ Server not responding to health check")
        return False

if __name__ == "__main__":
    print("🧪 RAG Endpoint Test")
    print("=" * 50)
    
    # Check server health first
    if check_server_health():
        time.sleep(1)
        test_rag_endpoint()
    else:
        print("Please start the Flask server first: python app.py")