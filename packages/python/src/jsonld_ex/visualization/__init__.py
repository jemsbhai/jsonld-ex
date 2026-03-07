"""
Visualization utilities for jsonld-ex graph structures.

Package-wide visualization module — not limited to SLNetwork.
Designed to support any current or future graph/network workflows
in jsonld-ex.

Modules:
    dot       — DOT/Graphviz string generation (zero dependencies)
    nx_export — NetworkX conversion (optional: ``pip install jsonld-ex[viz]``)
"""

from __future__ import annotations

from jsonld_ex.visualization.dot import to_dot

# to_networkx is lazily imported to avoid hard dependency on networkx.
# Access via: from jsonld_ex.visualization.nx_export import to_networkx
# or: from jsonld_ex.visualization import to_networkx (if networkx is installed)
try:
    from jsonld_ex.visualization.nx_export import to_networkx

    __all__ = [
        "to_dot",
        "to_networkx",
    ]
except ImportError:
    __all__ = [
        "to_dot",
    ]
