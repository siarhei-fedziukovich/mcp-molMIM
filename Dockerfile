# MolMIM MCP Server Dockerfile
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy server code and entrypoint script
COPY . .

# Make entrypoint script executable
RUN chmod +x entrypoint.sh

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash molmim && \
    chown -R molmim:molmim /app
USER molmim

# Expose default HTTP Streamable port
EXPOSE 8001

# Set default environment variables
ENV MCP_TRANSPORT=http-streamable
ENV MCP_HOST=0.0.0.0
ENV MCP_PORT=8001
ENV PYTHONUNBUFFERED=1
ENV DOCKER_CONTAINER=true

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8001/health', timeout=5)" || exit 1

# Use entrypoint script
ENTRYPOINT ["./entrypoint.sh"]
