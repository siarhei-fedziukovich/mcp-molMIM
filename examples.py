#!/usr/bin/env python3
"""
Example usage scripts for MolMIM MCP Server
Demonstrates how to use the MCP server with different MolMIM endpoints
"""

import asyncio
import json
import logging
import os
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Example SMILES strings
EXAMPLE_MOLECULES = {
    "caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "paracetamol": "CC(=O)NC1=CC=C(O)C=C1",
    "morphine": "CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O"
}

async def example_generate_drug_like_molecules():
    """Example: Generate drug-like molecules using CMA-ES optimization"""
    print("🔬 Example: Generate Drug-like Molecules")
    print("=" * 50)
    
    # Example request for generating drug-like molecules
    request = {
        "smi": EXAMPLE_MOLECULES["caffeine"],
        "algorithm": "CMA-ES",
        "num_molecules": 5,
        "property_name": "QED",
        "minimize": False,  # Maximize QED (drug-likeness)
        "min_similarity": 0.3,
        "particles": 20,
        "iterations": 5
    }
    
    print(f"Seed molecule: {EXAMPLE_MOLECULES['caffeine']} (Caffeine)")
    print(f"Request: {json.dumps(request, indent=2)}")
    print()
    print("This would generate 5 drug-like molecules optimized for QED score.")
    print("The CMA-ES algorithm will search for molecules with high drug-likeness")
    print("while maintaining at least 30% similarity to caffeine.")

async def example_optimize_partition_coefficient():
    """Example: Optimize molecules for partition coefficient (plogP)"""
    print("\n🧪 Example: Optimize Partition Coefficient")
    print("=" * 50)
    
    # Example request for plogP optimization
    request = {
        "smi": EXAMPLE_MOLECULES["aspirin"],
        "algorithm": "CMA-ES",
        "num_molecules": 3,
        "property_name": "plogP",
        "minimize": True,  # Minimize plogP (more hydrophilic)
        "min_similarity": 0.4,
        "particles": 15,
        "iterations": 3
    }
    
    print(f"Seed molecule: {EXAMPLE_MOLECULES['aspirin']} (Aspirin)")
    print(f"Request: {json.dumps(request, indent=2)}")
    print()
    print("This would generate 3 molecules with optimized partition coefficients.")
    print("Minimizing plogP makes molecules more hydrophilic (water-soluble).")
    print("This is useful for improving drug absorption and bioavailability.")

async def example_random_sampling():
    """Example: Random sampling around seed molecules"""
    print("\n🎲 Example: Random Sampling")
    print("=" * 50)
    
    # Example request for random sampling
    request = {
        "smi": EXAMPLE_MOLECULES["ibuprofen"],
        "algorithm": "none",  # Random sampling
        "num_molecules": 10,
        "scaled_radius": 0.8
    }
    
    print(f"Seed molecule: {EXAMPLE_MOLECULES['ibuprofen']} (Ibuprofen)")
    print(f"Request: {json.dumps(request, indent=2)}")
    print()
    print("This would generate 10 molecules by random sampling around ibuprofen.")
    print("The scaled_radius of 0.8 controls how far to explore in chemical space.")
    print("Higher values explore more diverse molecules, lower values stay closer.")

async def example_get_embeddings():
    """Example: Get molecular embeddings for similarity analysis"""
    print("\n🧬 Example: Get Molecular Embeddings")
    print("=" * 50)
    
    # Example request for embeddings
    request = {
        "sequences": [
            EXAMPLE_MOLECULES["caffeine"],
            EXAMPLE_MOLECULES["aspirin"],
            EXAMPLE_MOLECULES["ibuprofen"],
            EXAMPLE_MOLECULES["paracetamol"]
        ]
    }
    
    print("Molecules to embed:")
    for i, (name, smiles) in enumerate(request["sequences"], 1):
        print(f"  {i}. {name}: {smiles}")
    print()
    print(f"Request: {json.dumps(request, indent=2)}")
    print()
    print("This would return embeddings for all 4 molecules.")
    print("Embeddings can be used for:")
    print("  - Similarity calculations")
    print("  - Clustering molecules")
    print("  - Property prediction")
    print("  - Chemical space visualization")

async def example_latent_space_sampling():
    """Example: Sample latent space around molecules"""
    print("\n🔍 Example: Latent Space Sampling")
    print("=" * 50)
    
    # Example request for latent space sampling
    request = {
        "sequences": [EXAMPLE_MOLECULES["morphine"]],
        "num_molecules": 5,
        "beam_size": 3,
        "scaled_radius": 0.6
    }
    
    print(f"Seed molecule: {EXAMPLE_MOLECULES['morphine']} (Morphine)")
    print(f"Request: {json.dumps(request, indent=2)}")
    print()
    print("This would sample 5 molecules from the latent space around morphine.")
    print("Beam size of 3 explores multiple generation paths.")
    print("Scaled radius of 0.6 provides moderate exploration distance.")
    print("Useful for finding structurally similar compounds.")

async def example_workflow():
    """Example: Complete workflow combining multiple tools"""
    print("\n🔄 Example: Complete Workflow")
    print("=" * 50)
    
    print("Complete workflow example:")
    print("1. Generate novel molecules with CMA-ES optimization")
    print("2. Get embeddings for all generated molecules")
    print("3. Sample latent space around the best molecules")
    print("4. Decode hidden states to get final structures")
    print()
    
    workflow_steps = [
        {
            "step": 1,
            "tool": "molmim_generate",
            "description": "Generate 10 drug-like molecules from caffeine",
            "request": {
                "smi": EXAMPLE_MOLECULES["caffeine"],
                "algorithm": "CMA-ES",
                "num_molecules": 10,
                "property_name": "QED",
                "minimize": False
            }
        },
        {
            "step": 2,
            "tool": "molmim_embedding",
            "description": "Get embeddings for generated molecules",
            "request": {
                "sequences": ["[generated_smiles_here]"]
            }
        },
        {
            "step": 3,
            "tool": "molmim_sampling",
            "description": "Sample around the best molecules",
            "request": {
                "sequences": ["[best_smiles_here]"],
                "num_molecules": 5,
                "scaled_radius": 0.7
            }
        }
    ]
    
    for step in workflow_steps:
        print(f"Step {step['step']}: {step['tool']}")
        print(f"  {step['description']}")
        print(f"  Request: {json.dumps(step['request'], indent=4)}")
        print()

def main():
    """Run all examples"""
    print("🚀 MolMIM MCP Server Examples")
    print("=" * 60)
    print()
    
    # Run all examples
    asyncio.run(example_generate_drug_like_molecules())
    asyncio.run(example_optimize_partition_coefficient())
    asyncio.run(example_random_sampling())
    asyncio.run(example_get_embeddings())
    asyncio.run(example_latent_space_sampling())
    asyncio.run(example_workflow())
    
    print("=" * 60)
    print("🎉 All examples completed!")
    print()
    print("To use these examples with the MCP server:")
    print("1. Start the MolMIM server: python -m molmim_mcp.server")
    print("2. Use an MCP client to call these tools")
    print("3. Or use the MCP Inspector for interactive testing")

if __name__ == "__main__":
    main()
