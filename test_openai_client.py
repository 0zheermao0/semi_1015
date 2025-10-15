#!/usr/bin/env python3
"""
Test script for OpenAI-compatible client integration.
This script tests the basic functionality of the OpenAI client.
"""

import os
import sys
import argparse
from types import SimpleNamespace

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common.openai_client import create_openai_client

def test_openai_client():
    """Test the OpenAI client with a simple request."""

    # Create a mock config object
    config = SimpleNamespace(
        llm="gpt-3.5-turbo",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_base_url=os.getenv("OPENAI_BASE_URL"),
        temperature=0.0,
        max_tokens=100,
        timeout=30
    )

    # Test prompt
    test_prompt = """
    You are testing an OpenAI-compatible API client.
    Please respond with a simple JSON object:
    {
        "status": "working",
        "message": "The API client is functioning correctly"
    }
    """

    try:
        # Initialize the client
        print("Initializing OpenAI-compatible client...")
        client = create_openai_client(config)
        print(f"Client initialized successfully with model: {config.llm}")

        if config.openai_base_url:
            print(f"Using custom base URL: {config.openai_base_url}")

        # Test the client
        print("\nTesting API call...")
        response = client.generate(
            prompt=test_prompt,
            format='json'
        )

        print("API call successful!")
        print(f"Response: {response['response']}")

        if response.get('usage'):
            print(f"Token usage: {response['usage']}")

        print("\n✅ OpenAI client test PASSED!")
        return True

    except Exception as e:
        print(f"\n❌ OpenAI client test FAILED: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure OPENAI_API_KEY environment variable is set")
        print("2. If using a custom endpoint, make sure OPENAI_BASE_URL is set")
        print("3. Check your internet connection and API service status")
        print("4. Verify the model name is correct for your API provider")
        return False

def test_command_line_args():
    """Test parsing command line arguments like in main.py."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", type=str, default='gpt-3.5-turbo')
    parser.add_argument("--openai_api_key", type=str, default=None)
    parser.add_argument("--openai_base_url", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=int, default=60)

    args = parser.parse_args([])

    # Override with environment variables
    if not args.openai_api_key:
        args.openai_api_key = os.getenv("OPENAI_API_KEY")
    if not args.openai_base_url:
        args.openai_base_url = os.getenv("OPENAI_BASE_URL")

    print(f"Configuration from args: {args}")
    return args

if __name__ == "__main__":
    print("🧪 Testing OpenAI-compatible Client Integration")
    print("=" * 50)

    # Test command line argument parsing
    print("\n1. Testing command line argument parsing...")
    config = test_command_line_args()

    # Test actual API call
    print("\n2. Testing actual API functionality...")
    success = test_openai_client()

    if success:
        print("\n🎉 All tests passed! The OpenAI integration is ready to use.")
        print("\nTo run the main training script with OpenAI:")
        print("export OPENAI_API_KEY=your_key_here")
        print("export OPENAI_BASE_URL=your_endpoint_if_custom")
        print("python main.py --source dblpv7 --target citationv1 --expert_num 6")
    else:
        print("\n⚠️  Please fix the API configuration before running the main script.")
        sys.exit(1)