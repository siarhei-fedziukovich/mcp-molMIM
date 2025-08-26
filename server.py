#!/usr/bin/env python3
"""
MolMIM MCP Server
A Model Context Protocol server for NVIDIA MolMIM (Molecular Masked Image Modeling)
"""

import asyncio
import argparse
import json
import logging
import os
import sys
import base64
import io
from typing import Any, Dict, List, Optional, Sequence
import requests
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.server.lowlevel.server import NotificationOptions
from mcp.server.fastmcp import FastMCP
from mcp.types import (
    CallToolRequest,
    CallToolResult,
    ListToolsRequest,
    ListToolsResult,
    Tool,
    TextContent,
    LoggingLevel,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MolMIM server configuration
MOLMIM_BASE_URL = os.getenv("MOLMIM_BASE_URL", "http://localhost:8000")

class MolMIMServer:
    """MolMIM MCP Server implementation using low-level Server"""
    
    def __init__(self):
        self.server = Server("molmim")
        self.setup_tools()
    
    def setup_tools(self):
        """Setup all MolMIM tools"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> ListToolsResult:
            """List all available MolMIM tools"""
            tools = [
                Tool(
                    name="molmim_embedding",
                    description="Get embeddings for SMILES strings from MolMIM. Returns molecular embeddings that can be used for similarity analysis and molecular property prediction.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "sequences": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Array of SMILES strings to get embeddings for",
                                "minItems": 1
                            }
                        },
                        "required": ["sequences"]
                    }
                ),
                Tool(
                    name="molmim_hidden",
                    description="Get hidden state representations from MolMIM. Returns the internal hidden states that can be used for advanced molecular analysis and manipulation.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "sequences": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Array of SMILES strings to get hidden states for",
                                "minItems": 1
                            }
                        },
                        "required": ["sequences"]
                    }
                ),
                Tool(
                    name="molmim_decode",
                    description="Decode hidden states back to SMILES strings. Converts MolMIM's internal representations back to molecular structures.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "hiddens": {
                                "type": "array",
                                "items": {
                                    "type": "array",
                                    "items": {
                                        "type": "array",
                                        "items": {"type": "number"}
                                    }
                                },
                                "description": "Hidden state representations from MolMIM"
                            },
                            "mask": {
                                "type": "array",
                                "items": {
                                    "type": "array",
                                    "items": {"type": "boolean"}
                                },
                                "description": "Mask indicating which positions to decode"
                            }
                        },
                        "required": ["hiddens", "mask"]
                    }
                ),
                Tool(
                    name="molmim_sampling",
                    description="Sample latent space around seed molecules. Generates novel molecules by exploring the chemical space near the input molecules.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "sequences": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Array of SMILES strings to sample around",
                                "minItems": 1
                            },
                            "beam_size": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 10,
                                "default": 1,
                                "description": "Beam width for sampling (higher values explore more paths)"
                            },
                            "num_molecules": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 10,
                                "default": 1,
                                "description": "Number of molecules to generate"
                            },
                            "scaled_radius": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 2.0,
                                "default": 0.7,
                                "description": "Scaled radius for sampling (controls exploration distance)"
                            }
                        },
                        "required": ["sequences"]
                    }
                ),
                Tool(
                    name="molmim_generate",
                    description="Generate novel molecules with property optimization using CMA-ES or random sampling. Optimizes for drug-likeness (QED) or partition coefficient (plogP).",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "smi": {
                                "type": "string",
                                "description": "Seed molecule in SMILES format"
                            },
                            "algorithm": {
                                "type": "string",
                                "enum": ["CMA-ES", "none"],
                                "default": "CMA-ES",
                                "description": "Algorithm to use: CMA-ES for optimization, 'none' for random sampling"
                            },
                            "num_molecules": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 100,
                                "default": 10,
                                "description": "Number of molecules to generate"
                            },
                            "property_name": {
                                "type": "string",
                                "enum": ["QED", "plogP"],
                                "default": "QED",
                                "description": "Property to optimize: QED (drug-likeness) or plogP (partition coefficient)"
                            },
                            "minimize": {
                                "type": "boolean",
                                "default": False,
                                "description": "Whether to minimize (True) or maximize (False) the property"
                            },
                            "min_similarity": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 0.7,
                                "default": 0.7,
                                "description": "Minimum similarity threshold to the seed molecule"
                            },
                            "particles": {
                                "type": "integer",
                                "minimum": 2,
                                "maximum": 1000,
                                "default": 30,
                                "description": "Number of particles for CMA-ES optimization"
                            },
                            "iterations": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 1000,
                                "default": 10,
                                "description": "Number of optimization iterations"
                            },
                            "scaled_radius": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 2.0,
                                "default": 1.0,
                                "description": "Scaled radius for sampling (used with 'none' algorithm)"
                            }
                        },
                        "required": ["smi"]
                    }
                ),
                Tool(
                    name="molmim_interpolate",
                    description="Interpolate between two molecules by manipulating MolMIM hidden states. Generates intermediate molecules that share properties of each parent molecule, with either end of the spectrum being closer to respective starting molecule. Returns both a PNG visualization and JSON data.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "smiles1": {
                                "type": "string",
                                "description": "First molecule in SMILES format"
                            },
                            "smiles2": {
                                "type": "string",
                                "description": "Second molecule in SMILES format"
                            },
                            "num_interpolations": {
                                "type": "integer",
                                "minimum": 5,
                                "maximum": 100,
                                "default": 50,
                                "description": "Number of interpolated molecules to generate between the two input molecules"
                            },
                            "mols_per_row": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 10,
                                "default": 4,
                                "description": "Number of molecules per row in the visualization grid"
                            },
                            "image_size": {
                                "type": "integer",
                                "minimum": 200,
                                "maximum": 500,
                                "default": 300,
                                "description": "Size of each molecule image in pixels"
                            }
                        },
                        "required": ["smiles1", "smiles2"]
                    }
                )
            ]
            return ListToolsResult(tools=tools)

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> CallToolResult:
            """Handle tool calls to MolMIM endpoints"""
            try:
                logger.info(f"Calling tool: {name} with arguments: {arguments}")
                
                if name == "molmim_embedding":
                    return await self._call_embedding(arguments)
                elif name == "molmim_hidden":
                    return await self._call_hidden(arguments)
                elif name == "molmim_decode":
                    return await self._call_decode(arguments)
                elif name == "molmim_sampling":
                    return await self._call_sampling(arguments)
                elif name == "molmim_generate":
                    return await self._call_generate(arguments)
                elif name == "molmim_interpolate":
                    return await self._call_interpolate(arguments)
                else:
                    raise ValueError(f"Unknown tool: {name}")
            except Exception as e:
                logger.error(f"Error calling tool {name}: {str(e)}")
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Error: {str(e)}")]
                )

    async def _call_embedding(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Call the /embedding endpoint"""
        url = f"{MOLMIM_BASE_URL}/embedding"
        data = {"sequences": arguments["sequences"]}
        
        logger.info(f"Calling embedding endpoint with {len(arguments['sequences'])} sequences")
        
        response = requests.post(url, json=data, headers={"Content-Type": "application/json"})
        response.raise_for_status()
        
        result = response.json()
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(result, indent=2))]
        )

    async def _call_hidden(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Call the /hidden endpoint"""
        url = f"{MOLMIM_BASE_URL}/hidden"
        data = {"sequences": arguments["sequences"]}
        
        logger.info(f"Calling hidden endpoint with {len(arguments['sequences'])} sequences")
        
        response = requests.post(url, json=data, headers={"Content-Type": "application/json"})
        response.raise_for_status()
        
        result = response.json()
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(result, indent=2))]
        )

    async def _call_decode(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Call the /decode endpoint"""
        url = f"{MOLMIM_BASE_URL}/decode"
        data = {
            "hiddens": arguments["hiddens"],
            "mask": arguments["mask"]
        }
        
        logger.info(f"Calling decode endpoint with {len(arguments['hiddens'])} hidden states")
        
        response = requests.post(url, json=data, headers={"Content-Type": "application/json"})
        response.raise_for_status()
        
        result = response.json()
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(result, indent=2))]
        )

    async def _call_sampling(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Call the /sampling endpoint"""
        url = f"{MOLMIM_BASE_URL}/sampling"
        data = {
            "sequences": arguments["sequences"],
            "beam_size": arguments.get("beam_size", 1),
            "num_molecules": arguments.get("num_molecules", 1),
            "scaled_radius": arguments.get("scaled_radius", 0.7)
        }
        
        logger.info(f"Calling sampling endpoint with {len(arguments['sequences'])} sequences")
        
        response = requests.post(url, json=data, headers={"Content-Type": "application/json"})
        response.raise_for_status()
        
        result = response.json()
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(result, indent=2))]
        )

    async def _call_generate(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Call the /generate endpoint"""
        url = f"{MOLMIM_BASE_URL}/generate"
        data = {
            "smi": arguments["smi"],
            "algorithm": arguments.get("algorithm", "CMA-ES"),
            "num_molecules": arguments.get("num_molecules", 10),
            "property_name": arguments.get("property_name", "QED"),
            "minimize": arguments.get("minimize", False),
            "min_similarity": arguments.get("min_similarity", 0.7),
            "particles": arguments.get("particles", 30),
            "iterations": arguments.get("iterations", 10),
            "scaled_radius": arguments.get("scaled_radius", 1.0)
        }
        
        logger.info(f"Calling generate endpoint with algorithm: {data['algorithm']}, property: {data['property_name']}")
        
        response = requests.post(url, json=data, headers={"Content-Type": "application/json"})
        response.raise_for_status()
        
        result = response.json()
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(result, indent=2))]
        )

    async def _call_interpolate(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Call the molecular interpolation functionality"""
        try:
            smiles1 = arguments["smiles1"]
            smiles2 = arguments["smiles2"]
            num_interpolations = arguments.get("num_interpolations", 50)
            mols_per_row = arguments.get("mols_per_row", 4)
            image_size = arguments.get("image_size", 300)
            
            # Canonicalize SMILES strings
            smiles1_canon = Chem.CanonSmiles(smiles1)
            smiles2_canon = Chem.CanonSmiles(smiles2)
            
            logger.info(f"Interpolating between molecules: {smiles1_canon} and {smiles2_canon}")
            
            # Step 1: Get hidden states for both molecules
            url = f"{MOLMIM_BASE_URL}/hidden"
            data = {"sequences": [smiles1_canon, smiles2_canon]}
            
            response = requests.post(url, json=data, headers={"Content-Type": "application/json"})
            response.raise_for_status()
            
            hiddens_response = response.json()
            hiddens = hiddens_response["hiddens"]
            
            # Convert to numpy array and extract the two hidden state vectors
            hiddens_array = np.array(hiddens)
            hiddens_array = np.squeeze(hiddens_array)
            row1 = hiddens_array[0]  # First molecule
            row2 = hiddens_array[1]  # Second molecule
            
            # Step 2: Generate interpolated hidden states
            interpolated_rows = []
            for i in range(num_interpolations):
                t = i / (num_interpolations - 1)
                interpolated_row = (1 - t) * row1 + t * row2
                interpolated_rows.append(interpolated_row)
            
            # Convert interpolated vectors to array format expected by decoder
            interpolated_hiddens = np.expand_dims(np.array(interpolated_rows), axis=1)
            
            # Step 3: Decode interpolated hidden states to SMILES
            url = f"{MOLMIM_BASE_URL}/decode"
            interpolated_hiddens_json = {
                "hiddens": interpolated_hiddens.tolist(),
                "mask": [[True] for _ in range(num_interpolations)]
            }
            
            response = requests.post(url, json=interpolated_hiddens_json, headers={"Content-Type": "application/json"})
            response.raise_for_status()
            
            decode_response = response.json()
            generated_molecules = decode_response['generated']
            
            # Step 4: Deduplicate and create final molecule list
            molecules = [smiles1_canon] + list(dict.fromkeys(generated_molecules)) + [smiles2_canon]
            legends = ['Molecule 1'] + [f'Interpolated #{i+1}' for i in range(len(molecules) - 2)] + ['Molecule 2']
            
            # Step 5: Create visualization
            mols = [Chem.MolFromSmiles(smile, sanitize=False) for smile in molecules]
            
            # Create the grid image
            img = Draw.MolsToGridImage(
                mols,
                legends=legends,
                molsPerRow=mols_per_row,
                subImgSize=(image_size, image_size),
                returnPNG=True
            )
            
            # Convert image to base64
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            
            # Step 6: Prepare result
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
            
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(result, indent=2))]
            )
            
        except Exception as e:
            logger.error(f"Error in molecular interpolation: {str(e)}")
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]
            )

class MolMIMFastServer:
    """MolMIM MCP Server implementation using FastMCP for HTTP/SSE transports"""
    
    def __init__(self):
        self.mcp = FastMCP("MolMIM")
        self.setup_tools()
    
    def setup_tools(self):
        """Setup all MolMIM tools using FastMCP decorators"""
        
        @self.mcp.tool()
        async def molmim_embedding(sequences: List[str]) -> str:
            """Get embeddings for SMILES strings from MolMIM."""
            url = f"{MOLMIM_BASE_URL}/embedding"
            data = {"sequences": sequences}
            
            logger.info(f"Calling embedding endpoint with {len(sequences)} sequences")
            
            response = requests.post(url, json=data, headers={"Content-Type": "application/json"})
            response.raise_for_status()
            
            result = response.json()
            return json.dumps(result, indent=2)

        @self.mcp.tool()
        async def molmim_hidden(sequences: List[str]) -> str:
            """Get hidden state representations from MolMIM."""
            url = f"{MOLMIM_BASE_URL}/hidden"
            data = {"sequences": sequences}
            
            logger.info(f"Calling hidden endpoint with {len(sequences)} sequences")
            
            response = requests.post(url, json=data, headers={"Content-Type": "application/json"})
            response.raise_for_status()
            
            result = response.json()
            return json.dumps(result, indent=2)

        @self.mcp.tool()
        async def molmim_decode(hiddens: List[List[List[float]]], mask: List[List[bool]]) -> str:
            """Decode hidden states back to SMILES strings."""
            url = f"{MOLMIM_BASE_URL}/decode"
            data = {"hiddens": hiddens, "mask": mask}
            
            logger.info(f"Calling decode endpoint with {len(hiddens)} hidden states")
            
            response = requests.post(url, json=data, headers={"Content-Type": "application/json"})
            response.raise_for_status()
            
            result = response.json()
            return json.dumps(result, indent=2)

        @self.mcp.tool()
        async def molmim_sampling(
            sequences: List[str], 
            beam_size: int = 1, 
            num_molecules: int = 1, 
            scaled_radius: float = 0.7
        ) -> str:
            """Sample latent space around seed molecules."""
            url = f"{MOLMIM_BASE_URL}/sampling"
            data = {
                "sequences": sequences,
                "beam_size": beam_size,
                "num_molecules": num_molecules,
                "scaled_radius": scaled_radius
            }
            
            logger.info(f"Calling sampling endpoint with {len(sequences)} sequences")
            
            response = requests.post(url, json=data, headers={"Content-Type": "application/json"})
            response.raise_for_status()
            
            result = response.json()
            return json.dumps(result, indent=2)

        @self.mcp.tool()
        async def molmim_generate(
            smi: str,
            algorithm: str = "CMA-ES",
            num_molecules: int = 10,
            property_name: str = "QED",
            minimize: bool = False,
            min_similarity: float = 0.7,
            particles: int = 30,
            iterations: int = 10,
            scaled_radius: float = 1.0
        ) -> str:
            """Generate novel molecules with property optimization."""
            url = f"{MOLMIM_BASE_URL}/generate"
            data = {
                "smi": smi,
                "algorithm": algorithm,
                "num_molecules": num_molecules,
                "property_name": property_name,
                "minimize": minimize,
                "min_similarity": min_similarity,
                "particles": particles,
                "iterations": iterations,
                "scaled_radius": scaled_radius
            }
            
            logger.info(f"Calling generate endpoint with algorithm: {algorithm}, property: {property_name}")
            
            response = requests.post(url, json=data, headers={"Content-Type": "application/json"})
            response.raise_for_status()
            
            result = response.json()
            return json.dumps(result, indent=2)

        @self.mcp.tool()
        async def molmim_interpolate(
            smiles1: str,
            smiles2: str,
            num_interpolations: int = 50,
            mols_per_row: int = 4,
            image_size: int = 300
        ) -> str:
            """Interpolate between two molecules by manipulating MolMIM hidden states."""
            try:
                # Canonicalize SMILES strings
                smiles1_canon = Chem.CanonSmiles(smiles1)
                smiles2_canon = Chem.CanonSmiles(smiles2)
                
                logger.info(f"Interpolating between molecules: {smiles1_canon} and {smiles2_canon}")
                
                # Step 1: Get hidden states for both molecules
                url = f"{MOLMIM_BASE_URL}/hidden"
                data = {"sequences": [smiles1_canon, smiles2_canon]}
                
                response = requests.post(url, json=data, headers={"Content-Type": "application/json"})
                response.raise_for_status()
                
                hiddens_response = response.json()
                hiddens = hiddens_response["hiddens"]
                
                # Convert to numpy array and extract the two hidden state vectors
                hiddens_array = np.array(hiddens)
                hiddens_array = np.squeeze(hiddens_array)
                row1 = hiddens_array[0]  # First molecule
                row2 = hiddens_array[1]  # Second molecule
                
                # Step 2: Generate interpolated hidden states
                interpolated_rows = []
                for i in range(num_interpolations):
                    t = i / (num_interpolations - 1)
                    interpolated_row = (1 - t) * row1 + t * row2
                    interpolated_rows.append(interpolated_row)
                
                # Convert interpolated vectors to array format expected by decoder
                interpolated_hiddens = np.expand_dims(np.array(interpolated_rows), axis=1)
                
                # Step 3: Decode interpolated hidden states to SMILES
                url = f"{MOLMIM_BASE_URL}/decode"
                interpolated_hiddens_json = {
                    "hiddens": interpolated_hiddens.tolist(),
                    "mask": [[True] for _ in range(num_interpolations)]
                }
                
                response = requests.post(url, json=interpolated_hiddens_json, headers={"Content-Type": "application/json"})
                response.raise_for_status()
                
                decode_response = response.json()
                generated_molecules = decode_response['generated']
                
                # Step 4: Deduplicate and create final molecule list
                molecules = [smiles1_canon] + list(dict.fromkeys(generated_molecules)) + [smiles2_canon]
                legends = ['Molecule 1'] + [f'Interpolated #{i+1}' for i in range(len(molecules) - 2)] + ['Molecule 2']
                
                # Step 5: Create visualization
                mols = [Chem.MolFromSmiles(smile, sanitize=False) for smile in molecules]
                
                # Create the grid image
                img = Draw.MolsToGridImage(
                    mols,
                    legends=legends,
                    molsPerRow=mols_per_row,
                    subImgSize=(image_size, image_size),
                    returnPNG=True
                )
                
                # Convert image to base64
                img_buffer = io.BytesIO()
                img.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
                
                # Step 6: Prepare result
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
                
                return json.dumps(result, indent=2)
                
            except Exception as e:
                logger.error(f"Error in molecular interpolation: {str(e)}")
                return json.dumps({"error": str(e)}, indent=2)

async def run_stdio_server(molmim_server: MolMIMServer):
    """Run the MCP server using stdio transport"""
    logger.info("Starting MolMIM MCP Server with stdio transport")
    async with stdio_server() as (read_stream, write_stream):
        await molmim_server.server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="molmim",
                server_version="1.0.0",
                capabilities=molmim_server.server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

async def run_http_streamable_server(molmim_fast_server: MolMIMFastServer, host: str, port: int):
    """Run the MCP server using HTTP Streamable transport"""
    logger.info(f"Starting MolMIM MCP Server with HTTP Streamable transport on {host}:{port}")
    
    # Use the streamable_http_app() method and run with uvicorn
    import uvicorn
    app = molmim_fast_server.mcp.streamable_http_app()
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()

async def run_sse_server(molmim_fast_server: MolMIMFastServer, host: str, port: int):
    """Run the MCP server using Server-Sent Events (SSE) transport"""
    logger.info(f"Starting MolMIM MCP Server with SSE transport on {host}:{port}")
    
    # Use the sse_app() method and run with uvicorn
    import uvicorn
    app = molmim_fast_server.mcp.sse_app()
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="MolMIM MCP Server - A Model Context Protocol server for NVIDIA MolMIM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with stdio transport (default)
  python server.py

  # Run with HTTP Streamable transport
  python server.py --transport http-streamable --host 127.0.0.1 --port 8001

  # Run with Server-Sent Events transport
  python server.py --transport sse --host 0.0.0.0 --port 8002

  # Set MolMIM server URL (overrides environment variable)
  python server.py --molmim-url http://your-molmim-server:8000

Environment Variables:
  MOLMIM_BASE_URL: MolMIM server URL (default: http://localhost:8000)
  MCP_TRANSPORT: Transport type (stdio, http-streamable, sse)
  MCP_HOST: Host for HTTP Streamable/SSE transport
  MCP_PORT: Port for HTTP Streamable/SSE transport

Note: MolMIM requires NVIDIA NIM technology and enterprise subscription for official deployment.
      You may need to run your own MolMIM server instance if you have the model weights.
        """
    )
    
    parser.add_argument(
        "--transport", "-t",
        choices=["stdio", "http-streamable", "sse"],
        default=os.getenv("MCP_TRANSPORT", "stdio"),
        help="MCP transport mechanism (default: stdio)"
    )
    
    parser.add_argument(
        "--host", "-H",
        default=os.getenv("MCP_HOST", "127.0.0.1"),
        help="Host for HTTP Streamable/SSE transport (default: 127.0.0.1)"
    )
    
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=int(os.getenv("MCP_PORT", "8001")),
        help="Port for HTTP Streamable/SSE transport (default: 8001)"
    )
    
    parser.add_argument(
        "--molmim-url", "-u",
        default=None,  # Don't set default here, let environment variable take precedence
        help="MolMIM server URL (overrides MOLMIM_BASE_URL environment variable)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="MolMIM MCP Server 1.0.0"
    )
    
    return parser.parse_args()

async def main():
    """Main function to run the MCP server"""
    args = parse_arguments()
    
    # Update global MolMIM URL - prioritize command-line argument over environment variable
    global MOLMIM_BASE_URL
    if args.molmim_url is not None:
        # Command-line argument takes precedence
        MOLMIM_BASE_URL = args.molmim_url
    else:
        # Use environment variable, with fallback to default
        MOLMIM_BASE_URL = os.getenv("MOLMIM_BASE_URL", "http://localhost:8000")
    
    # Validate MOLMIM_BASE_URL
    if not MOLMIM_BASE_URL or MOLMIM_BASE_URL == "http://localhost:8000":
        # Check if we're running in Docker (common Docker environment variables)
        if os.getenv("DOCKER_CONTAINER") or os.getenv("KUBERNETES_SERVICE_HOST"):
            logger.error("MOLMIM_BASE_URL environment variable is required for Docker deployment")
            logger.error("Please set MOLMIM_BASE_URL to point to your MolMIM server")
            sys.exit(1)
        else:
            logger.warning("Using default MOLMIM_BASE_URL: http://localhost:8000")
            logger.warning("For production deployment, set MOLMIM_BASE_URL environment variable")
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info(f"MolMIM MCP Server Configuration:")
    logger.info(f"  MolMIM URL: {MOLMIM_BASE_URL}")
    logger.info(f"  Transport: {args.transport}")
    if args.transport in ["http-streamable", "sse"]:
        logger.info(f"  Host: {args.host}")
        logger.info(f"  Port: {args.port}")
    
    try:
        if args.transport == "stdio":
            # Use low-level Server for stdio transport
            molmim_server = MolMIMServer()
            await run_stdio_server(molmim_server)
        elif args.transport == "http-streamable":
            # Use FastMCP for HTTP Streamable transport
            molmim_fast_server = MolMIMFastServer()
            await run_http_streamable_server(molmim_fast_server, args.host, args.port)
        elif args.transport == "sse":
            # Use FastMCP for SSE transport
            molmim_fast_server = MolMIMFastServer()
            await run_sse_server(molmim_fast_server, args.host, args.port)
        else:
            logger.error(f"Unsupported transport: {args.transport}")
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
