#!/usr/bin/env python3
"""
Test script for debugging main_enhanced.py with loss tracking.
"""

import os
import sys

# Set environment variables
os.environ['OPENAI_API_KEY'] = 'test-key'  # Mock key for testing
os.environ['OPENAI_BASE_URL'] = 'http://localhost:8000/v1'  # Example local endpoint

# Add mock LLM response for testing without real API calls
def mock_generate(self, *args, **kwargs):
    """Mock LLM response for testing"""
    return {
        'response': '{"expert": 1, "confidence": 0.8, "reasoning": "Test reasoning", "ranking": [1, 0, 2, 3, 4, 5]}'
    }

# Patch the OpenAI client before importing main_enhanced
try:
    from common.openai_client import OpenAIClient
    OpenAIClient.generate = mock_generate
    print("✅ Successfully patched OpenAI client with mock responses")
except ImportError:
    print("⚠️  Could not patch OpenAI client, using original implementation")

if __name__ == "__main__":
    print("🧪 Testing main_enhanced.py with debugging...")
    print("=" * 50)

    # Set command line arguments for testing
    sys.argv = [
        'test_main_enhanced.py',
        '--source', 'dblpv7',
        '--target', 'citationv1',
        '--expert_num', '6',
        '--epochs', '5',  # Short test run
        '--llm_interval', '1',  # Call LLM every epoch
        '--uncertainty_k', '50',  # More uncertain nodes
        '--moe_architecture', 'original',
        '--wandb_project', 'test-enhanced-moe'
    ]

    print(f"Running with args: {' '.join(sys.argv[1:])}")
    print("=" * 50)

    try:
        # Import and run main_enhanced
        import main_enhanced
        print("✅ Test completed successfully!")
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()