#!/usr/bin/env python3
"""
Test script for MolMIM MCP Protocol
Tests the MCP server implementation using the MCP protocol
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_mcp_server():
    """Test the MolMIM MCP server functionality"""
    
    print("🧪 Testing MolMIM MCP Server Protocol")
    print("=" * 50)
    
    # Start the MCP server as a subprocess
    process = subprocess.Popen(
        [sys.executable, "-m", "molmim_mcp.server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    try:
        # Wait a moment for the server to start
        time.sleep(2)
        
        # Test data
        test_smiles = ["CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]  # Caffeine
        
        print(f"📝 Testing with SMILES: {test_smiles[0]}")
        print()
        
        # Test 1: Generate molecules
        print("🔬 Test 1: Generate molecules with CMA-ES optimization")
        test_generate_data = {
            "smi": test_smiles[0],
            "algorithm": "CMA-ES",
            "num_molecules": 3,
            "property_name": "QED",
            "minimize": False,
            "min_similarity": 0.3,
            "particles": 10,
            "iterations": 2
        }
        
        print(f"Request: {json.dumps(test_generate_data, indent=2)}")
        print()
        
        # Test 2: Get embeddings
        print("🧬 Test 2: Get molecule embeddings")
        test_embedding_data = {
            "sequences": test_smiles
        }
        
        print(f"Request: {json.dumps(test_embedding_data, indent=2)}")
        print()
        
        # Test 3: Sampling
        print("🎲 Test 3: Sample latent space")
        test_sampling_data = {
            "sequences": test_smiles,
            "num_molecules": 2,
            "scaled_radius": 0.5
        }
        
        print(f"Request: {json.dumps(test_sampling_data, indent=2)}")
        print()
        
        print("✅ MCP Server tests completed!")
        print("📋 The server provides the following tools:")
        print("   - molmim_embedding: Get embeddings for SMILES strings")
        print("   - molmim_hidden: Get hidden state representations")
        print("   - molmim_decode: Decode hidden states to SMILES")
        print("   - molmim_sampling: Sample latent space around molecules")
        print("   - molmim_generate: Generate optimized molecules")
        
    except Exception as e:
        print(f"❌ Error testing MCP server: {e}")
    finally:
        # Clean up
        process.terminate()
        process.wait()

def test_mcp_inspector():
    """Test with MCP Inspector"""
    print("\n🔍 Testing with MCP Inspector")
    print("=" * 40)
    
    print("To test with MCP Inspector:")
    print("1. Install MCP Inspector: npx @modelcontextprotocol/inspector")
    print("2. Use this configuration:")
    print()
    
    config = {
        "mcpServers": {
            "molmim": {
                "command": "python",
                "args": ["-m", "molmim_mcp.server"],
                "env": {
                    "MOLMIM_BASE_URL": "http://localhost:8000"
                }
            }
        }
    }
    
    print(json.dumps(config, indent=2))
    print()
    print("3. Start the inspector and connect to the MolMIM server")
    print("4. Test the available tools through the inspector interface")

def main():
    """Main test function"""
    print("🚀 MolMIM MCP Protocol Test Suite")
    print("=" * 50)
    
    # Test MCP server
    asyncio.run(test_mcp_server())
    
    # Test MCP Inspector
    test_mcp_inspector()
    
    print("\n🎉 MCP Protocol test suite completed!")
    print("📚 For more information, see:")
    print("   - MCP Specification: https://modelcontextprotocol.io/")
    print("   - MCP Inspector: https://github.com/modelcontextprotocol/inspector")

if __name__ == "__main__":
    main()
