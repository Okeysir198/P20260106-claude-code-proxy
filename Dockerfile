# Use pinned Python version for reproducibility and security
FROM python:3.11-slim

WORKDIR /claude-code-proxy

# Create non-root user for security
RUN useradd -m -u 1000 appuser

# Copy package specifications
COPY pyproject.toml uv.lock ./

# Install uv and project dependencies
RUN pip install --upgrade uv && uv sync --locked

# Copy project code to current directory
COPY . .

# Change ownership to non-root user
RUN chown -R appuser:appuser /claude-code-proxy

# Switch to non-root user
USER appuser

# Start the proxy (no --reload in production)
EXPOSE 4000
CMD ["uv", "run", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "4000"]
