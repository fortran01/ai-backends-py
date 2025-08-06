#!/usr/bin/env python3
"""
Phase 6 RAG Implementation Test Script

This script tests the Phase 6 RAG implementation by:
1. Checking if all dependencies are available
2. Testing the knowledge base setup script
3. Testing the RAG endpoint with sample queries
4. Generating a test report

Following the coding guidelines:
- Explicit type annotations for all functions
- Comprehensive error handling and logging
- Production-ready testing patterns
"""

import sys
import subprocess
import requests
import json
import time
from typing import Dict, List, Any, Optional
from pathlib import Path

def check_dependencies() -> Dict[str, bool]:
    """
    Check if all Phase 6 dependencies are available.
    
    Returns:
        Dict[str, bool]: Dictionary of dependency status
    """
    dependencies = {
        'llama-index': False,
        'chromadb': False,
        'requests': False,
        'flask': False
    }
    
    for dep in dependencies.keys():
        try:
            subprocess.run([sys.executable, '-c', f'import {dep.replace("-", "_")}'], 
                         check=True, capture_output=True)
            dependencies[dep] = True
        except subprocess.CalledProcessError:
            dependencies[dep] = False
    
    return dependencies

def test_knowledge_base_setup() -> Dict[str, Any]:
    """
    Test the knowledge base setup script.
    
    Returns:
        Dict[str, Any]: Test results
    """
    setup_script = Path('scripts/setup_rag_knowledge_base.py')
    
    if not setup_script.exists():
        return {
            'success': False,
            'error': 'Setup script not found',
            'path': str(setup_script)
        }
    
    knowledge_base = Path('knowledge_base.md')
    if not knowledge_base.exists():
        return {
            'success': False,
            'error': 'Knowledge base document not found',
            'path': str(knowledge_base)
        }
    
    return {
        'success': True,
        'setup_script_exists': True,
        'knowledge_base_exists': True,
        'knowledge_base_size': knowledge_base.stat().st_size
    }

def test_rag_endpoint(base_url: str = "http://localhost:5001") -> Dict[str, Any]:
    """
    Test the RAG chat endpoint with sample queries.
    
    Args:
        base_url: Base URL of the Flask application
        
    Returns:
        Dict[str, Any]: Test results
    """
    test_queries = [
        "What is machine learning?",
        "Explain the difference between supervised and unsupervised learning",
        "What are neural networks?",
        "How does RAG work?",
        "What is model drift?"
    ]
    
    results = {
        'server_running': False,
        'endpoint_available': False,
        'query_results': [],
        'average_response_time': 0,
        'success_rate': 0
    }
    
    # Check if server is running
    try:
        health_response = requests.get(f"{base_url}/health", timeout=5)
        results['server_running'] = health_response.status_code == 200
    except requests.RequestException:
        results['server_running'] = False
        return results
    
    # Test RAG endpoint
    successful_queries = 0
    total_time = 0
    
    for query in test_queries:
        query_result = {
            'query': query,
            'success': False,
            'response_time_ms': 0,
            'error': None,
            'response_preview': None
        }
        
        try:
            start_time = time.time()
            
            response = requests.post(
                f"{base_url}/api/v1/rag-chat",
                json={
                    'query': query,
                    'max_sources': 3,
                    'include_sources': True
                },
                timeout=30
            )
            
            end_time = time.time()
            query_result['response_time_ms'] = (end_time - start_time) * 1000
            total_time += query_result['response_time_ms']
            
            if response.status_code == 200:
                data = response.json()
                query_result['success'] = True
                query_result['response_preview'] = data.get('response', '')[:100] + '...'
                query_result['source_count'] = data.get('source_count', 0)
                successful_queries += 1
                results['endpoint_available'] = True
            else:
                query_result['error'] = f"HTTP {response.status_code}: {response.text[:100]}"
                
        except requests.RequestException as e:
            query_result['error'] = str(e)
        
        results['query_results'].append(query_result)
    
    if len(test_queries) > 0:
        results['success_rate'] = successful_queries / len(test_queries)
        results['average_response_time'] = total_time / len(test_queries) if len(test_queries) > 0 else 0
    
    return results

def generate_test_report(dep_results: Dict[str, bool], 
                        setup_results: Dict[str, Any], 
                        endpoint_results: Dict[str, Any]) -> str:
    """
    Generate a comprehensive test report.
    
    Args:
        dep_results: Dependency check results
        setup_results: Setup script test results  
        endpoint_results: Endpoint test results
        
    Returns:
        str: Formatted test report
    """
    report = []
    report.append("=" * 60)
    report.append("Phase 6 RAG Implementation Test Report")
    report.append("=" * 60)
    
    # Dependency Status
    report.append("\n📦 Dependency Status:")
    for dep, available in dep_results.items():
        status = "✅ Available" if available else "❌ Missing"
        report.append(f"  {dep}: {status}")
    
    missing_deps = [dep for dep, available in dep_results.items() if not available]
    if missing_deps:
        report.append(f"\n⚠️  Missing dependencies: {', '.join(missing_deps)}")
        report.append("   Install with: pip install llama-index llama-index-llms-ollama llama-index-embeddings-huggingface chromadb")
    
    # Setup Status
    report.append("\n🔧 Setup Status:")
    if setup_results['success']:
        report.append("  ✅ Knowledge base setup files found")
        report.append(f"  📄 Knowledge base size: {setup_results.get('knowledge_base_size', 0):,} bytes")
    else:
        report.append(f"  ❌ Setup issue: {setup_results.get('error', 'Unknown error')}")
    
    # Endpoint Testing
    report.append("\n🔗 Endpoint Testing:")
    if endpoint_results['server_running']:
        report.append("  ✅ Flask server is running")
        
        if endpoint_results['endpoint_available']:
            report.append("  ✅ RAG endpoint is functional")
            report.append(f"  📊 Success rate: {endpoint_results['success_rate']:.1%}")
            report.append(f"  ⏱️  Average response time: {endpoint_results['average_response_time']:.1f}ms")
            
            # Sample query results
            report.append("\n📝 Sample Query Results:")
            for result in endpoint_results['query_results'][:3]:  # Show first 3
                status = "✅" if result['success'] else "❌"
                report.append(f"  {status} \"{result['query'][:50]}...\"")
                if result['success']:
                    report.append(f"      Response: {result.get('response_preview', 'N/A')}")
                    report.append(f"      Sources: {result.get('source_count', 0)}, Time: {result['response_time_ms']:.1f}ms")
                elif result['error']:
                    report.append(f"      Error: {result['error']}")
        else:
            report.append("  ❌ RAG endpoint not available")
    else:
        report.append("  ❌ Flask server not running")
        report.append("      Start with: python app.py")
    
    # Next Steps
    report.append("\n🚀 Next Steps:")
    if missing_deps:
        report.append("  1. Install missing dependencies")
    if not setup_results['success']:
        report.append("  2. Fix setup script issues")
    if not endpoint_results['server_running']:
        report.append("  3. Start the Flask application")
    if not endpoint_results['endpoint_available'] and endpoint_results['server_running']:
        report.append("  4. Set up the RAG knowledge base: python scripts/setup_rag_knowledge_base.py")
    
    if all(dep_results.values()) and setup_results['success'] and endpoint_results['endpoint_available']:
        report.append("  🎉 Phase 6 RAG implementation is fully functional!")
    
    report.append("\n" + "=" * 60)
    
    return "\n".join(report)

def main() -> None:
    """
    Main function to run all Phase 6 tests.
    """
    print("🧪 Starting Phase 6 RAG Implementation Tests...\n")
    
    # Run tests
    print("Checking dependencies...")
    dep_results = check_dependencies()
    
    print("Testing setup...")
    setup_results = test_knowledge_base_setup()
    
    print("Testing RAG endpoint...")
    endpoint_results = test_rag_endpoint()
    
    # Generate and display report
    report = generate_test_report(dep_results, setup_results, endpoint_results)
    print(report)
    
    # Save report
    report_file = Path('phase6_test_report.txt')
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"\n📄 Test report saved to: {report_file}")

if __name__ == "__main__":
    main()