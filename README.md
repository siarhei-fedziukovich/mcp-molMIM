# MolMIM MCP Server

A Model Context Protocol (MCP) server for NVIDIA MolMIM (Molecular Masked Image Modeling) that provides access to all MolMIM endpoints as MCP tools.

## ⚠️ Important Notice: NVIDIA NIM Technology

**MolMIM is deployed using NVIDIA NIM (NVIDIA Inference Microservices) technology, which requires an NVIDIA Enterprise subscription.** 

- **NVIDIA NIM**: A cloud-native microservice platform for deploying AI models
- **Enterprise Subscription**: Required to access and deploy MolMIM through NVIDIA NIM
- **Alternative**: You can run your own MolMIM server instance if you have the model weights and infrastructure

For more information about NVIDIA NIM and enterprise subscriptions, visit:
- [NVIDIA NIM Documentation](https://docs.nvidia.com/nim/)
- [NVIDIA Enterprise](https://www.nvidia.com/en-us/enterprise/)

## 🚀 Features

- **Complete MolMIM Integration**: All 5 MolMIM endpoints available as MCP tools
- **Multiple Transport Options**: Support for stdio, HTTP Streamable, and Server-Sent Events (SSE) transports
- **Property Optimization**: Support for CMA-ES optimization of QED and plogP properties
- **Molecular Generation**: Generate novel molecules with desired properties
- **Embedding Extraction**: Get molecular embeddings for similarity analysis
- **Latent Space Sampling**: Explore chemical space around seed molecules
- **Hidden State Manipulation**: Work with MolMIM's latent representations
- **Docker Support**: Containerized deployment with docker-compose
- **Flexible Configuration**: Environment variables and command-line options

## 📋 Available Tools

### 1. `molmim_embedding`
Get embeddings for SMILES strings from MolMIM.

**Parameters:**
- `sequences` (array of strings): Array of SMILES strings

### 2. `molmim_hidden`
Get hidden state representations from MolMIM.

**Parameters:**
- `sequences` (array of strings): Array of SMILES strings

### 3. `molmim_decode`
Decode hidden states back to SMILES strings.

**Parameters:**
- `hiddens` (array): Hidden state representations
- `mask` (array): Mask for the hidden states

### 4. `molmim_sampling`
Sample latent space around seed molecules.

**Parameters:**
- `sequences` (array of strings): Array of SMILES strings to sample around
- `beam_size` (integer, 1-10, default: 1): Beam width for sampling
- `num_molecules` (integer, 1-10, default: 1): Number of molecules to generate
- `scaled_radius` (number, 0.0-2.0, default: 0.7): Scaled radius for sampling

### 5. `molmim_generate`
Generate novel molecules with property optimization.

**Parameters:**
- `smi` (string): Seed molecule in SMILES format
- `algorithm` (string, "CMA-ES" or "none", default: "CMA-ES"): Algorithm to use
- `num_molecules` (integer, 1-100, default: 10): Number of molecules to generate
- `property_name` (string, "QED" or "plogP", default: "QED"): Property to optimize
- `minimize` (boolean, default: false): Whether to minimize the property
- `min_similarity` (number, 0.0-0.7, default: 0.7): Minimum similarity threshold
- `particles` (integer, 2-1000, default: 30): Number of particles for optimization
- `iterations` (integer, 1-1000, default: 10): Number of optimization iterations
- `scaled_radius` (number, 0.0-2.0, default: 1.0): Scaled radius for sampling

### 6. `molmim_interpolate`
Interpolate between two molecules by manipulating MolMIM hidden states. Generates intermediate molecules that share properties of each parent molecule, with either end of the spectrum being closer to respective starting molecule.

**Parameters:**
- `smiles1` (string): First molecule in SMILES format
- `smiles2` (string): Second molecule in SMILES format
- `num_interpolations` (integer, 5-100, default: 50): Number of interpolated molecules to generate
- `mols_per_row` (integer, 1-10, default: 4): Number of molecules per row in visualization grid
- `image_size` (integer, 200-500, default: 300): Size of each molecule image in pixels

**Returns:**
- **stdio transport**: JSON data + native MCP image content type (PNG)
- **HTTP/SSE transports**: JSON with molecules, legends, interpolation count, base64-encoded PNG image, and input molecules

## 🛠️ Installation

### From Source

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd molmim-mcp
   ```

2. **Install dependencies:**
   ```bash
   pip install -r molmim_mcp/requirements.txt
   ```

3. **Install the package:**
   ```bash
   pip install -e .
   ```

### Using pip

```bash
pip install molmim-mcp
```

## 🚀 Usage

### Running the MCP Server

```bash
# Using the installed script
molmim-mcp

# Or directly with Python
python -m molmim_mcp.server
```

### Configuration

Set the MolMIM server URL using environment variables:

```bash
export MOLMIM_BASE_URL="http://your-molmim-server:8000"
molmim-mcp
```

### MCP Client Configuration

Create an MCP configuration file (e.g., `mcp_config.json`):

```json
{
  "mcpServers": {
    "molmim": {
      "command": "molmim-mcp",
      "env": {
        "MOLMIM_BASE_URL": "http://localhost:8000"
      }
    }
  }
}
```

## 🌐 Transport Options

The MolMIM MCP server supports multiple transport mechanisms for different deployment scenarios:

### 1. stdio Transport (Default)

**Best for:** Local development, single client usage, native MCP image content types

```bash
# Run with stdio transport
python server.py

# Or with verbose logging
python server.py --verbose
```

**Configuration:**
```json
{
  "mcpServers": {
    "molmim": {
      "command": "python",
      "args": ["server.py"],
      "env": {
        "MOLMIM_BASE_URL": "http://localhost:8000"
      }
    }
  }
}
```

### 2. Server-Sent Events (SSE) Transport

**Best for:** Web-based clients, real-time applications, network deployment

```bash
# Run with SSE transport
python server.py --transport sse --host 0.0.0.0 --port 8002

# Or with environment variables
export MCP_TRANSPORT=sse
export MCP_HOST=0.0.0.0
export MCP_PORT=8002
python server.py
```

**Configuration:**
```json
{
  "mcpServers": {
    "molmim": {
      "command": "python",
      "args": ["server.py", "--transport", "sse", "--host", "0.0.0.0", "--port", "8002"],
      "env": {
        "MOLMIM_BASE_URL": "http://localhost:8000"
      }
    }
  }
}
```

### 3. HTTP Streamable Transport

**Best for:** Network deployment, multiple clients, production environments

```bash
# Run with HTTP Streamable transport
python server.py --transport http-streamable --host 0.0.0.0 --port 8001

# Or with environment variables
export MCP_TRANSPORT=http-streamable
export MCP_HOST=0.0.0.0
export MCP_PORT=8001
python server.py
```

**Configuration:**
```json
{
  "mcpServers": {
    "molmim": {
      "command": "python",
      "args": ["server.py", "--transport", "http-streamable", "--host", "0.0.0.0", "--port", "8001"],
      "env": {
        "MOLMIM_BASE_URL": "http://localhost:8000"
      }
    }
  }
}
```

### Environment Variables

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `MOLMIM_BASE_URL` | `http://localhost:8000` | Yes (Docker) | MolMIM server URL |
| `MCP_TRANSPORT` | `stdio` | No | Transport type: `stdio`, `http-streamable`, `sse` |
| `MCP_HOST` | `127.0.0.1` | No | Host for HTTP Streamable/SSE transport |
| `MCP_PORT` | `8001` | No | Port for HTTP Streamable/SSE transport |
| `PYTHONUNBUFFERED` | `1` (Docker) | No | Set to `1` for immediate log output |
| `VERBOSE` | - | No | Set to `true` or `1` to enable verbose logging |

### Docker Deployment

The MolMIM MCP server can be deployed using Docker with full environment variable support:

```bash
# Build the Docker image
docker build -t molmim-mcp .

# Run with required MolMIM server URL (uses entrypoint script)
docker run -p 8001:8001 -e MOLMIM_BASE_URL=http://your-molmim-server:8000 molmim-mcp

# Run with different transport
docker run -p 8002:8002 -e MCP_TRANSPORT=sse -e MCP_PORT=8002 -e MOLMIM_BASE_URL=http://your-molmim-server:8000 molmim-mcp

# Run with verbose logging
docker run -p 8001:8001 -e MOLMIM_BASE_URL=http://your-molmim-server:8000 -e VERBOSE=true molmim-mcp

# Override with custom command (bypasses entrypoint)
docker run -p 8001:8001 -e MOLMIM_BASE_URL=http://your-molmim-server:8000 molmim-mcp python server.py --help
```

**Required Environment Variables:**
- `MOLMIM_BASE_URL`: URL of the MolMIM server (required for Docker deployment)

**Optional Environment Variables:**
- `MCP_TRANSPORT`: Transport type (default: `http-streamable`)
- `MCP_HOST`: Host binding (default: `0.0.0.0`)
- `MCP_PORT`: Port binding (default: `8001`)
- `PYTHONUNBUFFERED`: Set to `1` for immediate log output (default: `1`)
- `VERBOSE`: Set to `true` or `1` to enable verbose logging

### Docker Compose Deployment

For complete deployment with MolMIM server:

```bash
# Start both MolMIM server and MCP server
docker-compose --profile with-molmim up -d

# Start only the MCP server (requires external MolMIM server)
docker-compose up -d molmim-mcp
```

**⚠️ Note**: The MolMIM server in the docker-compose example uses `nvidia/molmim:latest` which requires NVIDIA NIM technology and an enterprise subscription. You may need to replace this with your own MolMIM server implementation or use a different image.

**Docker Compose Environment Variables:**
- `MOLMIM_BASE_URL`: URL of the MolMIM server (default: `http://molmim-server:8000`)
- `MCP_TRANSPORT`: Transport type (default: `http-streamable`)
- `MCP_HOST`: Host binding (default: `0.0.0.0`)
- `MCP_PORT`: Port binding (default: `8001`)

### Command Line Options

```bash
python server.py --help
```

**Available options:**
- `--transport, -t`: Transport mechanism (`stdio`, `http-streamable`, `sse`)
- `--host, -H`: Host for HTTP Streamable/SSE transport
- `--port, -p`: Port for HTTP Streamable/SSE transport
- `--molmim-url, -u`: MolMIM server URL
- `--verbose, -v`: Enable verbose logging
- `--version`: Show version information

## 📊 Example Usage

### Generate Drug-like Molecules

```python
# Example: Generate 5 drug-like molecules from caffeine
generate_request = {
    "smi": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
    "algorithm": "CMA-ES",
    "num_molecules": 5,
    "property_name": "QED",
    "minimize": False,
    "min_similarity": 0.3,
    "particles": 20,
    "iterations": 5
}
```

### Get Molecular Embeddings

```python
# Example: Get embeddings for multiple molecules
embedding_request = {
    "sequences": [
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "CC(=O)OC1=CC=CC=C1C(=O)O"       # Aspirin
    ]
}
```

### Sample Chemical Space

```python
# Example: Sample around ibuprofen
sampling_request = {
    "sequences": ["CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"],  # Ibuprofen
    "num_molecules": 10,
    "scaled_radius": 0.8
}
```

### Interpolate Between Molecules

```python
# Example: Interpolate between caffeine and aspirin
interpolation_request = {
    "smiles1": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
    "smiles2": "CC(=O)OC1=CC=CC=C1C(=O)O",      # Aspirin
    "num_interpolations": 20,
    "mols_per_row": 5,
    "image_size": 250
}
```

## 🔧 Integration with DIAL

The MolMIM MCP server can be integrated with DIAL QuickApps using the MCP toolset configuration:

```json
{
  "mcp_toolset": [
    {
      "name": "molmim-mcp",
      "type": "mcp",
      "mcp_server_info": {
        "command": "molmim-mcp"
      },
             "allowed_tools": [
         "molmim_embedding",
         "molmim_hidden", 
         "molmim_decode",
         "molmim_sampling",
         "molmim_generate",
         "molmim_interpolate"
       ]
    }
  ]
}
```

## 🧪 Testing

### Test the Server

```bash
# Test direct API calls
python molmim_mcp/test_server.py

# Test MCP protocol
python molmim_mcp/test_mcp.py
```

### Test with MCP Inspector

```bash
# Install MCP Inspector
npx @modelcontextprotocol/inspector

# Connect to the MolMIM server
# Use the configuration from above
```

## 📚 API Reference

### MolMIM Endpoints

Based on the [NVIDIA MolMIM documentation](https://docs.nvidia.com/nim/bionemo/molmim/latest/endpoints.html):

- **`/embedding`**: Get molecular embeddings
- **`/hidden`**: Get hidden state representations  
- **`/decode`**: Decode hidden states to SMILES
- **`/sampling`**: Sample latent space
- **`/generate`**: Generate optimized molecules

### MCP Protocol

The server implements the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) specification:

- **Tool Listing**: `list_tools()` returns available MolMIM tools
- **Tool Execution**: `call_tool()` executes MolMIM API calls
- **Error Handling**: Comprehensive error handling and logging
- **Async Support**: Full async/await support for concurrent operations

## 🔍 Troubleshooting

### Common Issues

1. **Connection Error**: Ensure MolMIM server is running and accessible
2. **Import Error**: Install all required dependencies from `requirements.txt`
3. **Permission Error**: Ensure Python has execute permissions for the server script

### Debug Mode

Enable debug logging by setting the environment variable:

```bash
export PYTHONPATH=.
export LOG_LEVEL=DEBUG
molmim-mcp
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📞 Support

For issues related to:
- **MolMIM API**: Refer to [NVIDIA MolMIM documentation](https://docs.nvidia.com/nim/bionemo/molmim/latest/endpoints.html)
- **MCP Protocol**: Check the [MCP specification](https://modelcontextprotocol.io/)
- **This Server**: Open an issue in the repository

## 🔗 Related Links

- [Model Context Protocol](https://modelcontextprotocol.io/)
- [NVIDIA MolMIM Documentation](https://docs.nvidia.com/nim/bionemo/molmim/latest/endpoints.html)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [MCP Inspector](https://github.com/modelcontextprotocol/inspector)