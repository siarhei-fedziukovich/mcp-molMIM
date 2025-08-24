#!/usr/bin/env python3
"""
Test script for MolMIM MCP Server
Tests direct API calls to verify MolMIM server connectivity
"""

import asyncio
import json
import logging
import os
import sys
import time
from typing import Dict, Any
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MolMIM server configuration
MOLMIM_BASE_URL = os.getenv("MOLMIM_BASE_URL", "http://localhost:8000")

def test_direct_api_calls():
    """Test direct API calls to verify MolMIM server is working"""
    
    print("🔌 Testing Direct MolMIM API Calls")
    print("=" * 50)
    print(f"Target URL: {MOLMIM_BASE_URL}")
    print()
    
    test_smiles = ["CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]  # Caffeine
    
    try:
        # Test 1: Generate endpoint
        print("🔬 Test 1: /generate endpoint")
        print("-" * 30)
        generate_data = {
            "smi": test_smiles[0],
            "algorithm": "CMA-ES",
            "num_molecules": 2,
            "property_name": "QED",
            "minimize": False,
            "min_similarity": 0.3,
            "particles": 5,
            "iterations": 2
        }
        
        print(f"Request: {json.dumps(generate_data, indent=2)}")
        print()
        
        response = requests.post(f"{MOLMIM_BASE_URL}/generate", json=generate_data, timeout=30)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Generate successful! Generated {len(result.get('generated', []))} molecules")
            for i, mol in enumerate(result.get('generated', [])[:2]):
                print(f"   Molecule {i+1}: {mol.get('smiles', 'N/A')} (Score: {mol.get('score', 'N/A'):.3f})")
        else:
            print(f"❌ Generate failed: {response.status_code} - {response.text}")
        
        print()
        
        # Test 2: Embedding endpoint
        print("🧬 Test 2: /embedding endpoint")
        print("-" * 30)
        embedding_data = {"sequences": test_smiles}
        print(f"Request: {json.dumps(embedding_data, indent=2)}")
        print()
        
        response = requests.post(f"{MOLMIM_BASE_URL}/embedding", json=embedding_data, timeout=30)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Embedding successful! Got embeddings for {len(result.get('embeddings', []))} molecules")
            if result.get('embeddings'):
                print(f"   Embedding dimension: {len(result['embeddings'][0])}")
        else:
            print(f"❌ Embedding failed: {response.status_code} - {response.text}")
        
        print()
        
        # Test 3: Sampling endpoint
        print("🎲 Test 3: /sampling endpoint")
        print("-" * 30)
        sampling_data = {
            "sequences": test_smiles,
            "num_molecules": 2,
            "scaled_radius": 0.5
        }
        print(f"Request: {json.dumps(sampling_data, indent=2)}")
        print()
        
        response = requests.post(f"{MOLMIM_BASE_URL}/sampling", json=sampling_data, timeout=30)
        if response.status_code == 200:
            result = response.json()
            generated = result.get('generated', [])
            if isinstance(generated, list):
                print(f"✅ Sampling successful! Generated {len(generated)} molecules")
                for i, mol in enumerate(generated[:2]):
                    if isinstance(mol, dict):
                        print(f"   Molecule {i+1}: {mol.get('smiles', 'N/A')}")
                    else:
                        print(f"   Molecule {i+1}: {mol}")
            else:
                print(f"✅ Sampling successful! Result: {result}")
        else:
            print(f"❌ Sampling failed: {response.status_code} - {response.text}")
        
        print()
        
        # Test 4: Hidden endpoint
        print("🔍 Test 4: /hidden endpoint")
        print("-" * 30)
        hidden_data = {"sequences": test_smiles}
        print(f"Request: {json.dumps(hidden_data, indent=2)}")
        print()
        
        response = requests.post(f"{MOLMIM_BASE_URL}/hidden", json=hidden_data, timeout=30)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Hidden successful! Got hidden states for {len(result.get('hiddens', []))} molecules")
            if result.get('hiddens'):
                print(f"   Hidden state shape: {len(result['hiddens'][0])} x {len(result['hiddens'][0][0])}")
        else:
            print(f"❌ Hidden failed: {response.status_code} - {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to MolMIM server.")
        print(f"   Make sure it's running on {MOLMIM_BASE_URL}")
        print("   You can set MOLMIM_BASE_URL environment variable to change the URL")
        return False
    except requests.exceptions.Timeout:
        print("❌ Request timed out. The server might be overloaded or slow.")
        return False
    except Exception as e:
        print(f"❌ Error testing API: {e}")
        return False
    
    return True

def test_server_health():
    """Test basic server health"""
    print("🏥 Testing Server Health")
    print("=" * 30)
    
    try:
        # Try to connect to the server
        response = requests.get(f"{MOLMIM_BASE_URL}/", timeout=5)
        print(f"✅ Server is responding (Status: {response.status_code})")
        return True
    except requests.exceptions.ConnectionError:
        print("❌ Server is not accessible")
        return False
    except Exception as e:
        print(f"⚠️  Server health check failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 MolMIM MCP Server Test Suite")
    print("=" * 50)
    print()
    
    # Test server health first
    if not test_server_health():
        print("\n❌ Server health check failed. Please ensure MolMIM server is running.")
        sys.exit(1)
    
    print()
    
    # Test direct API calls
    success = test_direct_api_calls()
    
    print()
    print("=" * 50)
    if success:
        print("🎉 All tests completed successfully!")
        print("✅ MolMIM server is ready for MCP integration")
    else:
        print("❌ Some tests failed. Please check the MolMIM server configuration.")
        sys.exit(1)

if __name__ == "__main__":
    main()
