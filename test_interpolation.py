#!/usr/bin/env python3
"""
Test script for MolMIM molecular interpolation functionality
"""

import json
import base64
import os
import sys
import requests
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
from PIL import Image
import io

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# MolMIM server configuration
MOLMIM_BASE_URL = os.getenv("MOLMIM_BASE_URL", "http://localhost:8000")

def test_molecular_interpolation():
    """Test the molecular interpolation functionality"""
    
    # Test molecules (caffeine and aspirin)
    smiles1 = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"  # Caffeine
    smiles2 = "CC(=O)OC1=CC=CC=C1C(=O)O"      # Aspirin
    
    print(f"Testing molecular interpolation between:")
    print(f"  Molecule 1: {smiles1} (Caffeine)")
    print(f"  Molecule 2: {smiles2} (Aspirin)")
    print()
    
    try:
        # Step 1: Canonicalize SMILES strings
        smiles1_canon = Chem.CanonSmiles(smiles1)
        smiles2_canon = Chem.CanonSmiles(smiles2)
        
        print(f"Canonicalized SMILES:")
        print(f"  Molecule 1: {smiles1_canon}")
        print(f"  Molecule 2: {smiles2_canon}")
        print()
        
        # Step 2: Get hidden states for both molecules
        print("Step 1: Getting hidden states...")
        url = f"{MOLMIM_BASE_URL}/hidden"
        data = {"sequences": [smiles1_canon, smiles2_canon]}
        
        response = requests.post(url, json=data, headers={"Content-Type": "application/json"})
        response.raise_for_status()
        
        hiddens_response = response.json()
        hiddens = hiddens_response["hiddens"]
        
        print(f"✓ Retrieved hidden states for {len(hiddens)} molecules")
        
        # Step 3: Convert to numpy array and extract the two hidden state vectors
        hiddens_array = np.array(hiddens)
        hiddens_array = np.squeeze(hiddens_array)
        row1 = hiddens_array[0]  # First molecule
        row2 = hiddens_array[1]  # Second molecule
        
        print(f"✓ Hidden state shapes: {row1.shape}, {row2.shape}")
        
        # Step 4: Generate interpolated hidden states
        print("Step 2: Generating interpolated hidden states...")
        num_interpolations = 10  # Reduced for testing
        interpolated_rows = []
        
        for i in range(num_interpolations):
            t = i / (num_interpolations - 1)
            interpolated_row = (1 - t) * row1 + t * row2
            interpolated_rows.append(interpolated_row)
        
        # Convert interpolated vectors to array format expected by decoder
        interpolated_hiddens = np.expand_dims(np.array(interpolated_rows), axis=1)
        
        print(f"✓ Generated {num_interpolations} interpolated hidden states")
        
        # Step 5: Decode interpolated hidden states to SMILES
        print("Step 3: Decoding interpolated hidden states to SMILES...")
        url = f"{MOLMIM_BASE_URL}/decode"
        interpolated_hiddens_json = {
            "hiddens": interpolated_hiddens.tolist(),
            "mask": [[True] for _ in range(num_interpolations)]
        }
        
        response = requests.post(url, json=interpolated_hiddens_json, headers={"Content-Type": "application/json"})
        response.raise_for_status()
        
        decode_response = response.json()
        generated_molecules = decode_response['generated']
        
        print(f"✓ Decoded {len(generated_molecules)} molecules")
        
        # Step 6: Deduplicate and create final molecule list
        molecules = [smiles1_canon] + list(dict.fromkeys(generated_molecules)) + [smiles2_canon]
        legends = ['Caffeine'] + [f'Interpolated #{i+1}' for i in range(len(molecules) - 2)] + ['Aspirin']
        
        print(f"✓ Final molecule list: {len(molecules)} molecules (including endpoints)")
        
        # Step 7: Create visualization
        print("Step 4: Creating visualization...")
        mols = [Chem.MolFromSmiles(smile, sanitize=False) for smile in molecules]
        
        # Create the grid image
        img = Draw.MolsToGridImage(
            mols,
            legends=legends,
            molsPerRow=4,
            subImgSize=(300, 300),
            returnPNG=True
        )
        
        # Convert image to base64
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        
        print("✓ Generated visualization image")
        
        # Step 8: Save results
        result = {
            "molecules": molecules,
            "legends": legends,
            "interpolation_count": num_interpolations,
            "image_base64": img_base64,
            "input_molecules": {
                "smiles1": smiles1_canon,
                "smiles2": smiles2_canon
            }
        }
        
        # Save JSON result
        with open("interpolation_result.json", "w") as f:
            json.dump(result, f, indent=2)
        
        # Save image
        img.save("interpolation_visualization.png")
        
        print("\n✓ Test completed successfully!")
        print("✓ Results saved to:")
        print("  - interpolation_result.json")
        print("  - interpolation_visualization.png")
        
        # Display some sample molecules
        print(f"\nSample interpolated molecules:")
        for i, (mol, legend) in enumerate(zip(molecules[1:-1], legends[1:-1])):
            if i < 3:  # Show first 3 interpolated molecules
                print(f"  {legend}: {mol}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during interpolation test: {str(e)}")
        return False

def test_mcp_interpolation_tool():
    """Test the MCP interpolation tool via direct API call"""
    print("\n" + "="*60)
    print("Testing MCP Interpolation Tool")
    print("="*60)
    
    # This would test the actual MCP tool call
    # For now, we'll just show the expected format
    print("MCP Tool Call Format:")
    print(json.dumps({
        "name": "molmim_interpolate",
        "arguments": {
            "smiles1": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            "smiles2": "CC(=O)OC1=CC=CC=C1C(=O)O",
            "num_interpolations": 10,
            "mols_per_row": 4,
            "image_size": 300
        }
    }, indent=2))

if __name__ == "__main__":
    print("MolMIM Molecular Interpolation Test")
    print("="*40)
    
    # Test the core interpolation functionality
    success = test_molecular_interpolation()
    
    if success:
        # Test MCP tool format
        test_mcp_interpolation_tool()
        
        print("\n" + "="*60)
        print("All tests completed!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("Tests failed!")
        print("="*60)
        sys.exit(1)
