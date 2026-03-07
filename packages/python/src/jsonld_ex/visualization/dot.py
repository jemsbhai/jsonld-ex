"""
DOT/Graphviz string generation for jsonld-ex graph structures.

Zero external dependencies — pure string generation.

Produces DOT language output that can be rendered by Graphviz, or
pasted into online viewers (e.g. https://dreampuf.github.io/GraphvizOnline/).

Styling conventions:
    Nodes:
        - Content nodes: ellipse, filled, color intensity ∝ belief
        - Agent nodes: box shape, light blue fill
        - Label shows human-readable name + opinion (b/d/u)
    Edges:
        - Deduction: solid black, label shows conditional opinion
        - Trust: dashed blue, label shows trust opinion
        - Attestation: dotted green, label shows attestation opinion

References:
    DOT language: https://graphviz.org/doc/info/lang.html
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jsonld_ex.sl_network.network import SLNetwork

from jsonld_ex.confidence_algebra import Opinion


# ═══════════════════════════════════════════════════════════════════
# COLOR HELPERS
# ═══════════════════════════════════════════════════════════════════


def _belief_color(opinion: Opinion) -> str:
    """Map belief intensity to a fill color.

    High belief → saturated green, low belief → light gray,
    high uncertainty → white.  Uses HSV in Graphviz format.

    Returns:
        A Graphviz color string (hex RGB).
    """
    b = opinion.belief
    u = opinion.uncertainty

    # Saturation driven by (1 - uncertainty): vacuous → white
    saturation = 1.0 - u
    # Hue: green (120°) for high belief, red (0°) for high disbelief
    # Lightness: brighter for lower belief
    # Simple approach: interpolate green channel with belief
    r = int(255 * (1.0 - b * saturation * 0.6))
    g = int(255 * (1.0 - (1.0 - b) * saturation * 0.4))
    bl = int(255 * (1.0 - b * saturation * 0.3))

    # Clamp
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    bl = max(0, min(255, bl))

    return f"#{r:02x}{g:02x}{bl:02x}"


def _opinion_str(opinion: Opinion) -> str:
    """Format an opinion as a compact string for labels."""
    return f"({opinion.belief:.2f},{opinion.disbelief:.2f},{opinion.uncertainty:.2f})"


def _dot_id(node_id: str) -> str:
    """Escape a node ID for DOT output.

    Wraps in double quotes to handle special characters.
    """
    escaped = node_id.replace('"', '\\"')
    return f'"{escaped}"'


def _sanitize_graph_name(name: str | None) -> str:
    """Sanitize a network name for use as a DOT graph name."""
    if not name:
        return "SLNetwork"
    # Replace non-alphanumeric chars with underscores
    return "".join(c if c.isalnum() or c == "_" else "_" for c in name)


# ═══════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════


def to_dot(network: SLNetwork) -> str:
    """Generate a DOT (Graphviz) representation of an SLNetwork.

    Produces a complete ``digraph`` string that can be rendered by
    Graphviz tools (``dot``, ``neato``, etc.) or pasted into online
    viewers.

    Styling:
        - Content nodes: ellipse shape, fill color by belief intensity
        - Agent nodes: box shape, light blue fill
        - Deduction edges: solid black, conditional opinion label
        - Trust edges: dashed blue, trust opinion label
        - Attestation edges: dotted green, attestation opinion label

    Args:
        network: The SLNetwork to visualize.

    Returns:
        A DOT language string.
    """
    lines: list[str] = []
    graph_name = _sanitize_graph_name(network.name)

    lines.append(f"digraph {graph_name} {{")
    lines.append("    rankdir=TB;")
    lines.append('    node [style=filled, fontname="Helvetica", fontsize=10];')
    lines.append('    edge [fontname="Helvetica", fontsize=9];')
    lines.append("")

    # ── Nodes ──
    for node_id in sorted(network._nodes.keys()):
        node = network._nodes[node_id]
        display_name = node.label or node.node_id
        op_str = _opinion_str(node.opinion)

        if node.node_type == "agent":
            # Agent: box shape, light blue
            label = f"{display_name}\\n(agent)"
            lines.append(
                f"    {_dot_id(node_id)} ["
                f'label="{label}", '
                f'shape=box, '
                f'fillcolor="#d0e8ff", '
                f'color="#4a90d9"'
                f"];"
            )
        else:
            # Content: ellipse, color by belief
            fill = _belief_color(node.opinion)
            label = f"{display_name}\\nb/d/u={op_str}"
            lines.append(
                f"    {_dot_id(node_id)} ["
                f'label="{label}", '
                f'shape=ellipse, '
                f'fillcolor="{fill}", '
                f'color="#333333"'
                f"];"
            )

    lines.append("")

    # ── Deduction edges ──
    for (src, tgt), edge in sorted(network._edges.items()):
        cond_str = _opinion_str(edge.conditional)
        lines.append(
            f"    {_dot_id(src)} -> {_dot_id(tgt)} ["
            f'label="ded {cond_str}", '
            f'style=solid, '
            f'color="#333333"'
            f"];"
        )

    # ── Trust edges ──
    for (src, tgt), te in sorted(network._trust_edges.items()):
        trust_str = _opinion_str(te.trust_opinion)
        lines.append(
            f"    {_dot_id(src)} -> {_dot_id(tgt)} ["
            f'label="trust {trust_str}", '
            f'style=dashed, '
            f'color="#4a90d9"'
            f"];"
        )

    # ── Attestation edges ──
    for (aid, cid), ae in sorted(network._attestation_edges.items()):
        att_str = _opinion_str(ae.opinion)
        lines.append(
            f"    {_dot_id(aid)} -> {_dot_id(cid)} ["
            f'label="attest {att_str}", '
            f'style=dotted, '
            f'color="#2ecc71"'
            f"];"
        )

    lines.append("}")

    return "\n".join(lines)
