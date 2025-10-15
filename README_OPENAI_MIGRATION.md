# OpenAI-Compatible API Migration

This document describes the migration from Ollama to OpenAI-compatible API for LLM inference in the MoEDG project.

## Overview

The MoEDG project has been updated to use OpenAI-compatible APIs instead of Ollama for LLM inference. This provides better flexibility, reliability, and compatibility with various LLM providers.

## Changes Made

### 1. New OpenAI Client Module

Created `common/openai_client.py`:
- `OpenAIClient` class for handling OpenAI-compatible API calls
- `create_openai_client()` function for easy initialization
- Support for custom base URLs and API keys
- Configurable temperature, max tokens, and timeout parameters

### 2. Updated Main Scripts

Modified files:
- `main.py` - Main training script
- `ablation.py` - Ablation study script
- `main_enhanced.py` - Enhanced version (if needed)

Key changes:
- Replaced `import ollama` with `from common.openai_client import create_openai_client`
- Added new command-line arguments for OpenAI configuration
- Updated LLM generation calls to use OpenAI client
- Changed error messages and logging

### 3. New Configuration Parameters

Added the following command-line arguments to all scripts:

```bash
--openai_api_key     # OpenAI API key (can also use OPENAI_API_KEY env var)
--openai_base_url    # Custom API endpoint (can also use OPENAI_BASE_URL env var)
--temperature        # Sampling temperature (default: 0.0)
--max_tokens         # Maximum response tokens (default: 4096)
--timeout            # Request timeout in seconds (default: 60)
```

### 4. Updated Default Model

Changed default LLM model from `qwen2.5:7b` to `gpt-3.5-turbo`.

## Usage

### Using OpenAI API

1. Set your OpenAI API key:
```bash
export OPENAI_API_KEY=sk-your-openai-api-key-here
```

2. Run the training script:
```bash
python main.py --source dblpv7 --target citationv1 --expert_num 6 --llm gpt-3.5-turbo
```

### Using Custom OpenAI-Compatible Endpoints

1. Set your custom endpoint:
```bash
export OPENAI_BASE_URL=http://your-llm-server:8000/v1
export OPENAI_API_KEY=your-api-key-if-needed
```

2. Run with your model:
```bash
python main.py --source dblpv7 --target citationv1 --expert_num 6 --llm llama-2-7b-chat
```

### Using Command Line Arguments

You can also specify parameters directly:

```bash
python main.py \
    --source dblpv7 \
    --target citationv1 \
    --expert_num 6 \
    --llm gpt-3.5-turbo \
    --openai_api_key sk-your-key-here \
    --openai_base_url https://api.openai.com/v1 \
    --temperature 0.0 \
    --max_tokens 4096
```

## Testing

Run the test script to verify your OpenAI configuration:

```bash
# Install dependencies
pip install -r requirements.txt

# Run test
python test_openai_client.py
```

## Examples

### Example 1: OpenAI API
```bash
export OPENAI_API_KEY=sk-your-openai-key
python main.py --source dblpv7 --target citationv1 --llm gpt-4
```

### Example 2: Local Server with Ollama
```bash
# Start Ollama with OpenAI compatibility
ollama serve --host 0.0.0.0:11434

# Set environment variables
export OPENAI_BASE_URL=http://localhost:11434/v1
export OPENAI_API_KEY=not-needed

# Run training
python main.py --source dblpv7 --target citationv1 --llm qwen2.5:7b
```

### Example 3: Custom LLM Server
```bash
export OPENAI_BASE_URL=http://llm-server.company.com:8000/v1
export OPENAI_API_KEY=your-company-api-key
python main.py --source dblpv7 --target citationv1 --llm custom-llm-v1
```

## Configuration File

You can also use the provided `config_example.yaml` as a reference for configuration options.

## Environment Variables

The following environment variables are supported:

- `OPENAI_API_KEY` - Your API key
- `OPENAI_BASE_URL` - Custom API endpoint URL

## Troubleshooting

### Common Issues

1. **API Key Not Found**
   ```
   Error: OPENAI_API_KEY not found
   ```
   Solution: Set the `OPENAI_API_KEY` environment variable or use `--openai_api_key`

2. **Connection Timeout**
   ```
   Error: Request timeout after 60 seconds
   ```
   Solution: Increase timeout with `--timeout 120` or check your network connection

3. **Invalid Model Name**
   ```
   Error: Model 'invalid-model' not found
   ```
   Solution: Check available models for your API provider and use a valid model name

4. **Custom Endpoint Not Responding**
   ```
   Error: Connection refused
   ```
   Solution: Verify your custom endpoint is running and accessible at the specified URL

### Debug Mode

Enable debug logging by setting the environment variable:
```bash
export OPENAI_LOG=debug
python main.py --source dblpv7 --target citationv1
```

## Migration Checklist

- [ ] Install required dependencies: `pip install openai>=1.0.0`
- [ ] Set `OPENAI_API_KEY` environment variable
- [ ] (Optional) Set `OPENAI_BASE_URL` for custom endpoints
- [ ] Run `python test_openai_client.py` to verify configuration
- [ ] Update your training scripts with new command-line arguments
- [ ] Update any documentation or job scripts

## Backward Compatibility

This change is **not backward compatible** with the previous Ollama-only setup. You must:
1. Update your environment variables
2. Update your command-line arguments
3. Ensure you have the required dependencies installed

## Benefits

- **Better Reliability**: OpenAI API provides more stable connections
- **More Options**: Use any OpenAI-compatible LLM provider
- **Flexibility**: Easy switching between different LLM services
- **Better Error Handling**: Improved timeout and retry mechanisms
- **Standardization**: Using industry-standard OpenAI API format

## Support

If you encounter issues with the migration:

1. Check the troubleshooting section above
2. Run the test script to verify your configuration
3. Ensure all dependencies are installed: `pip install -r requirements.txt`
4. Check your API provider's documentation for specific requirements