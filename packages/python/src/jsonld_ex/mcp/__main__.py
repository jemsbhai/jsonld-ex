"""Entry point for running the jsonld-ex MCP server.

Usage::

    python -m jsonld_ex.mcp              # stdio transport (default)
    python -m jsonld_ex.mcp --http       # streamable HTTP
    python -m jsonld_ex.mcp --sse        # SSE transport
"""

import sys

from jsonld_ex.mcp.server import mcp

if __name__ == "__main__":
    transport = "stdio"
    if "--http" in sys.argv:
        transport = "streamable-http"
    elif "--sse" in sys.argv:
        transport = "sse"
    mcp.run(transport=transport)
