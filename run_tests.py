#!/usr/bin/env python3
"""
Test runner script for Flask API testing.

Following the coding guidelines:
- Explicit type annotations for all functions
- Comprehensive test execution with proper reporting
- Support for different test categories and options
"""

import sys
import subprocess
import argparse
from typing import List, Optional
import os


def run_command(command: List[str], description: str) -> bool:
    """
    Run a shell command and return success status.
    
    Args:
        command: List of command arguments
        description: Description of the command for logging
        
    Returns:
        True if command succeeded, False otherwise
    """
    print(f"\n🔍 {description}")
    print(f"Running: {' '.join(command)}")
    
    try:
        result = subprocess.run(command, check=True, capture_output=False)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"❌ Command not found: {command[0]}")
        return False


def install_playwright() -> bool:
    """
    Install Playwright browsers if needed.
    
    Returns:
        True if installation succeeded, False otherwise
    """
    print("\n🎭 Checking Playwright installation...")
    
    # Check if playwright is installed
    try:
        subprocess.run(["playwright", "--version"], check=True, capture_output=True)
        print("✅ Playwright is already installed")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("📥 Installing Playwright browsers...")
        return run_command(["playwright", "install"], "Playwright browser installation")


def main() -> int:
    """
    Main test runner function.
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(description="Run Flask API tests")
    parser.add_argument(
        "--type", 
        choices=["unit", "e2e", "all", "security"],
        default="all",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--coverage", 
        action="store_true",
        help="Run tests with coverage reporting"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Run tests in verbose mode"
    )
    parser.add_argument(
        "--install-deps", 
        action="store_true",
        help="Install test dependencies before running"
    )
    
    args = parser.parse_args()
    
    print("🚀 Flask API Test Runner")
    print("=" * 50)
    
    # Change to script directory
    script_dir: str = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Install dependencies if requested
    if args.install_deps:
        if not run_command(["pip", "install", "-r", "requirements.txt"], "Installing dependencies"):
            return 1
        
        if not install_playwright():
            return 1
    
    # Build pytest command
    pytest_cmd: List[str] = ["python", "-m", "pytest"]
    
    # Add coverage if requested
    if args.coverage:
        pytest_cmd.extend(["--cov=app", "--cov-report=html", "--cov-report=term-missing"])
    
    # Add verbosity
    if args.verbose:
        pytest_cmd.append("-v")
    else:
        pytest_cmd.append("-q")
    
    # Add test selection based on type
    if args.type == "unit":
        pytest_cmd.extend(["-m", "not e2e"])
        test_description = "unit and integration tests"
    elif args.type == "e2e":
        pytest_cmd.extend(["-m", "e2e"])
        test_description = "end-to-end tests"
    elif args.type == "security":
        pytest_cmd.extend(["-m", "security"])
        test_description = "security tests"
    else:
        test_description = "all tests"
    
    # Add test directory
    pytest_cmd.append("tests/")
    
    # Run the tests
    success: bool = run_command(pytest_cmd, f"Running {test_description}")
    
    if success:
        print("\n🎉 All tests passed!")
        if args.coverage:
            print("📊 Coverage report generated in htmlcov/index.html")
    else:
        print("\n💥 Some tests failed!")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())