#!/bin/bash
set -e

# MolMIM MCP Server Entrypoint Script
# This script handles environment variable validation and server startup

echo "=== MolMIM MCP Server Entrypoint ==="
echo "Environment:"
echo "  MOLMIM_BASE_URL: ${MOLMIM_BASE_URL:-not set}"
echo "  MCP_TRANSPORT: ${MCP_TRANSPORT:-stdio}"
echo "  MCP_HOST: ${MCP_HOST:-127.0.0.1}"
echo "  MCP_PORT: ${MCP_PORT:-8001}"
echo "  PYTHONUNBUFFERED: ${PYTHONUNBUFFERED:-not set}"
echo ""

# Validate required environment variables
if [ -z "$MOLMIM_BASE_URL" ]; then
    echo "ERROR: MOLMIM_BASE_URL environment variable is required for Docker deployment"
    echo "Please set MOLMIM_BASE_URL to point to your MolMIM server"
    echo "Example: docker run -e MOLMIM_BASE_URL=http://your-server:8000 molmim-mcp"
    exit 1
fi

# Set default values for optional environment variables
export MCP_TRANSPORT=${MCP_TRANSPORT:-stdio}
export MCP_HOST=${MCP_HOST:-127.0.0.1}
export MCP_PORT=${MCP_PORT:-8001}
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}

# Build command arguments based on transport
CMD_ARGS=("python" "server.py")

if [ "$MCP_TRANSPORT" != "stdio" ]; then
    CMD_ARGS+=("--transport" "$MCP_TRANSPORT")
    CMD_ARGS+=("--host" "$MCP_HOST")
    CMD_ARGS+=("--port" "$MCP_PORT")
fi

# Add MolMIM URL if provided
if [ -n "$MOLMIM_BASE_URL" ]; then
    CMD_ARGS+=("--molmim-url" "$MOLMIM_BASE_URL")
fi

# Add verbose flag if requested
if [ "$VERBOSE" = "true" ] || [ "$VERBOSE" = "1" ]; then
    CMD_ARGS+=("--verbose")
fi

echo "Starting MolMIM MCP Server with command:"
echo "  ${CMD_ARGS[*]}"
echo ""

# Execute the command
exec "${CMD_ARGS[@]}"
