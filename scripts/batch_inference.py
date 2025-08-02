"""
Batch Inference Script for Ollama API

This script demonstrates offline (batch) inference by:
1. Reading prompts from input files (text or CSV format)
2. Calling the Ollama API for each prompt
3. Saving generated responses to output files
4. Demonstrating the difference between online and offline inference patterns

Following the coding guidelines:
- Explicit type annotations for all functions
- Comprehensive error handling and logging
- Educational comments about batch vs real-time inference
"""

import csv
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ollama API configuration
OLLAMA_BASE_URL: str = "http://localhost:11434"
OLLAMA_MODEL: str = "tinyllama"


def read_prompts_from_text(file_path: Path) -> List[str]:
    """
    Read prompts from a text file (one prompt per line).
    
    Args:
        file_path: Path to the text file containing prompts
        
    Returns:
        List of prompt strings
        
    Raises:
        FileNotFoundError: If the input file doesn't exist
        IOError: If there's an error reading the file
    """
    try:
        with file_path.open('r', encoding='utf-8') as f:
            prompts: List[str] = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Read {len(prompts)} prompts from {file_path}")
        return prompts
        
    except FileNotFoundError:
        logger.error(f"Input file not found: {file_path}")
        raise
    except IOError as e:
        logger.error(f"Error reading file {file_path}: {e}")
        raise


def read_prompts_from_csv(file_path: Path, prompt_column: str = 'prompt') -> List[Dict[str, Any]]:
    """
    Read prompts from a CSV file with metadata.
    
    Args:
        file_path: Path to the CSV file
        prompt_column: Name of the column containing prompts
        
    Returns:
        List of dictionaries containing prompt data and metadata
        
    Raises:
        FileNotFoundError: If the input file doesn't exist
        KeyError: If the prompt column is not found in CSV
        IOError: If there's an error reading the file
    """
    try:
        prompt_data: List[Dict[str, Any]] = []
        
        with file_path.open('r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            if prompt_column not in reader.fieldnames:
                raise KeyError(f"Column '{prompt_column}' not found in CSV. Available columns: {reader.fieldnames}")
            
            for row_idx, row in enumerate(reader):
                if row[prompt_column].strip():  # Skip empty prompts
                    prompt_data.append({
                        'id': row_idx + 1,
                        'prompt': row[prompt_column].strip(),
                        'metadata': {k: v for k, v in row.items() if k != prompt_column}
                    })
        
        logger.info(f"Read {len(prompt_data)} prompts from CSV {file_path}")
        return prompt_data
        
    except FileNotFoundError:
        logger.error(f"CSV file not found: {file_path}")
        raise
    except KeyError as e:
        logger.error(f"Error reading CSV {file_path}: {e}")
        raise
    except IOError as e:
        logger.error(f"Error reading CSV file {file_path}: {e}")
        raise


def call_ollama_api(prompt: str, timeout: int = 30) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Call the Ollama API for a single prompt.
    
    Args:
        prompt: The prompt to send to the LLM
        timeout: Request timeout in seconds
        
    Returns:
        Tuple of (success, response_text, metadata)
        
    Raises:
        requests.RequestException: If there's an error with the HTTP request
    """
    try:
        start_time: float = time.time()
        
        # Prepare request payload
        ollama_request: Dict[str, Any] = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False  # Use non-streaming for batch processing
        }
        
        # Make API request
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=ollama_request,
            timeout=timeout
        )
        
        response.raise_for_status()
        
        # Parse response
        response_data: Dict[str, Any] = response.json()
        generated_text: str = response_data.get('response', '')
        
        # Calculate processing time and other metadata
        processing_time: float = time.time() - start_time
        metadata: Dict[str, Any] = {
            'processing_time_seconds': round(processing_time, 3),
            'model_used': OLLAMA_MODEL,
            'prompt_length': len(prompt),
            'response_length': len(generated_text),
            'total_duration': response_data.get('total_duration', 0),
            'load_duration': response_data.get('load_duration', 0),
            'prompt_eval_count': response_data.get('prompt_eval_count', 0),
            'eval_count': response_data.get('eval_count', 0)
        }
        
        logger.debug(f"Generated response for prompt (length: {len(prompt)}) in {processing_time:.3f}s")
        
        return True, generated_text, metadata
        
    except requests.exceptions.ConnectionError:
        logger.error("Failed to connect to Ollama API - ensure Ollama is running")
        return False, "Error: Ollama API unavailable", {"error": "connection_error"}
    except requests.exceptions.Timeout:
        logger.error(f"Request timed out after {timeout} seconds")
        return False, "Error: Request timed out", {"error": "timeout"}
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {e}")
        return False, f"Error: {str(e)}", {"error": "request_failed"}
    except Exception as e:
        logger.error(f"Unexpected error calling Ollama API: {e}")
        return False, f"Error: {str(e)}", {"error": "unexpected_error"}


def process_prompts_batch(prompts: List[str], output_file: Path, delay_seconds: float = 1.0) -> Dict[str, Any]:
    """
    Process a list of prompts in batch mode.
    
    Args:
        prompts: List of prompt strings to process
        output_file: Path to save the results
        delay_seconds: Delay between requests to avoid overwhelming the API
        
    Returns:
        Dictionary containing batch processing statistics
    """
    logger.info(f"Starting batch processing of {len(prompts)} prompts")
    logger.info(f"Output will be saved to: {output_file}")
    
    results: List[Dict[str, Any]] = []
    successful_requests: int = 0
    failed_requests: int = 0
    total_processing_time: float = 0.0
    
    start_time: float = time.time()
    
    for idx, prompt in enumerate(prompts, 1):
        logger.info(f"Processing prompt {idx}/{len(prompts)}: {prompt[:100]}...")
        
        # Call Ollama API
        success, response_text, metadata = call_ollama_api(prompt)
        
        # Record result
        result: Dict[str, Any] = {
            'prompt_id': idx,
            'prompt': prompt,
            'response': response_text,
            'success': success,
            'metadata': metadata
        }
        results.append(result)
        
        # Update statistics
        if success:
            successful_requests += 1
            total_processing_time += metadata.get('processing_time_seconds', 0)
        else:
            failed_requests += 1
        
        # Add delay between requests (except for the last one)
        if idx < len(prompts) and delay_seconds > 0:
            logger.debug(f"Waiting {delay_seconds} seconds before next request...")
            time.sleep(delay_seconds)
    
    # Save results to output file
    try:
        with output_file.open('w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_file}")
        
    except IOError as e:
        logger.error(f"Failed to save results to {output_file}: {e}")
        raise
    
    # Calculate batch statistics
    total_elapsed_time: float = time.time() - start_time
    avg_processing_time: float = total_processing_time / successful_requests if successful_requests > 0 else 0
    
    statistics: Dict[str, Any] = {
        'total_prompts': len(prompts),
        'successful_requests': successful_requests,
        'failed_requests': failed_requests,
        'success_rate': successful_requests / len(prompts) if prompts else 0,
        'total_elapsed_time_seconds': round(total_elapsed_time, 3),
        'average_processing_time_seconds': round(avg_processing_time, 3),
        'requests_per_minute': round((len(prompts) / total_elapsed_time) * 60, 2) if total_elapsed_time > 0 else 0
    }
    
    logger.info("Batch processing completed!")
    logger.info(f"Statistics: {json.dumps(statistics, indent=2)}")
    
    return statistics


def create_sample_input_files() -> None:
    """
    Create sample input files for demonstration purposes.
    
    This creates both text and CSV input files with example prompts
    to demonstrate different batch processing scenarios.
    """
    logger.info("Creating sample input files for demonstration...")
    
    # Create sample text file
    text_prompts: List[str] = [
        "What is artificial intelligence?",
        "Explain the difference between machine learning and deep learning.",
        "How do neural networks work?",
        "What are the benefits of using AI in healthcare?",
        "Describe the concept of natural language processing."
    ]
    
    text_file: Path = Path("sample_prompts.txt")
    with text_file.open('w', encoding='utf-8') as f:
        for prompt in text_prompts:
            f.write(f"{prompt}\n")
    
    logger.info(f"Created sample text file: {text_file}")
    
    # Create sample CSV file with metadata
    csv_data: List[Dict[str, str]] = [
        {"id": "1", "category": "AI_Basics", "prompt": "What is the Turing test?", "priority": "high"},
        {"id": "2", "category": "ML_Algorithms", "prompt": "Explain decision trees in machine learning.", "priority": "medium"},
        {"id": "3", "category": "Ethics", "prompt": "What are the ethical concerns with AI?", "priority": "high"},
        {"id": "4", "category": "Applications", "prompt": "How is AI used in autonomous vehicles?", "priority": "medium"},
        {"id": "5", "category": "Future", "prompt": "What is the future of artificial general intelligence?", "priority": "low"}
    ]
    
    csv_file: Path = Path("sample_prompts.csv")
    with csv_file.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["id", "category", "prompt", "priority"])
        writer.writeheader()
        for row in csv_data:
            writer.writerow(row)
    
    logger.info(f"Created sample CSV file: {csv_file}")


def main() -> None:
    """
    Main function to handle command line arguments and orchestrate batch processing.
    """
    parser = argparse.ArgumentParser(
        description="Batch inference script for Ollama API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process prompts from text file
  python scripts/batch_inference.py --input sample_prompts.txt --output results.json

  # Process prompts from CSV file  
  python scripts/batch_inference.py --input sample_prompts.csv --csv --output results.json

  # Create sample input files
  python scripts/batch_inference.py --create-samples

  # Custom delay between requests
  python scripts/batch_inference.py --input prompts.txt --output results.json --delay 2.0
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Path to input file containing prompts (text or CSV format)'
    )
    
    parser.add_argument(
        '--output', '-o', 
        type=str,
        help='Path to output JSON file for results'
    )
    
    parser.add_argument(
        '--csv',
        action='store_true',
        help='Process input as CSV file (default: treat as text file)'
    )
    
    parser.add_argument(
        '--prompt-column',
        type=str,
        default='prompt',
        help='Column name containing prompts in CSV file (default: "prompt")'
    )
    
    parser.add_argument(
        '--delay',
        type=float,
        default=1.0,
        help='Delay in seconds between API requests (default: 1.0)'
    )
    
    parser.add_argument(
        '--create-samples',
        action='store_true',
        help='Create sample input files for demonstration'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create sample files if requested
    if args.create_samples:
        create_sample_input_files()
        return
    
    # Validate arguments
    if not args.input or not args.output:
        parser.error("Both --input and --output arguments are required (unless using --create-samples)")
    
    input_file: Path = Path(args.input)
    output_file: Path = Path(args.output)
    
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return
    
    try:
        # Read prompts based on file type
        if args.csv:
            logger.info(f"Reading prompts from CSV file: {input_file}")
            prompt_data = read_prompts_from_csv(input_file, args.prompt_column)
            prompts = [item['prompt'] for item in prompt_data]
        else:
            logger.info(f"Reading prompts from text file: {input_file}")
            prompts = read_prompts_from_text(input_file)
        
        if not prompts:
            logger.warning("No prompts found in input file")
            return
        
        # Process prompts in batch
        statistics = process_prompts_batch(prompts, output_file, args.delay)
        
        print("\n" + "="*80)
        print("🎯 BATCH PROCESSING COMPLETED")
        print("="*80)
        print(f"📊 Total prompts processed: {statistics['total_prompts']}")
        print(f"✅ Successful requests: {statistics['successful_requests']}")
        print(f"❌ Failed requests: {statistics['failed_requests']}")
        print(f"📈 Success rate: {statistics['success_rate']:.1%}")
        print(f"⏱️  Total time: {statistics['total_elapsed_time_seconds']:.3f} seconds")
        print(f"⚡ Average processing time: {statistics['average_processing_time_seconds']:.3f} seconds per prompt")
        print(f"🚀 Throughput: {statistics['requests_per_minute']:.1f} requests per minute")
        print(f"💾 Results saved to: {output_file}")
        
        print("\n📋 KEY INSIGHTS:")
        print("   • Batch processing enables efficient handling of multiple prompts")
        print("   • Offline inference allows for better resource utilization")
        print("   • Results are stored for later analysis and processing")
        print("   • Rate limiting (delay) prevents API overwhelming")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        raise


if __name__ == "__main__":
    main()