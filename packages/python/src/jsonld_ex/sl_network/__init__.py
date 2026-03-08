"""
SLNetwork — Subjective Logic Network inference engine.

A directed acyclic graph (DAG) inference engine for Subjective Logic,
extending jsonld-ex from flat operator calls into graph-structured
reasoning with Bayesian network interop, transitive trust propagation,
and temporal decay over graph edges.

Built on top of the existing jsonld-ex confidence algebra primitives
(Opinion, cumulative_fuse, deduce, trust_discount, decay_opinion).
SLNetwork composes these operators — it never reimplements them.

References:
    Jøsang, A. (2016). Subjective Logic: A Formalism for Reasoning
    Under Uncertainty. Springer.
"""

from __future__ import annotations

# Tier 1: Content Network types (Step 1)
from jsonld_ex.sl_network.types import (
    NodeType,
    EdgeType,
    SLNode,
    SLEdge,
    MultiParentEdge,
    InferenceStep,
    InferenceResult,
    # Tier 2: Trust Network types
    TrustEdge,
    AttestationEdge,
    TrustPropagationResult,
)

# Tier 1: Core graph container (Step 2)
from jsonld_ex.sl_network.network import (
    SLNetwork,
    CycleError,
    NodeNotFoundError,
)

# Tier 1: Counterfactual strategies (Step 3)
from jsonld_ex.sl_network.counterfactuals import (
    CounterfactualFn,
    vacuous_counterfactual,
    adversarial_counterfactual,
    prior_counterfactual,
    get_counterfactual_fn,
    COUNTERFACTUAL_STRATEGIES,
)

# Tier 1: Inference algorithms (Step 4)
from jsonld_ex.sl_network.inference import (
    infer_node,
    infer_all,
)

# Tier 2: Trust propagation and combined inference
from jsonld_ex.sl_network.trust import (
    propagate_trust,
    infer_with_trust,
)

# Tier 3: Temporal decay and point-in-time inference
from jsonld_ex.sl_network.temporal import (
    decay_network_nodes,
    decay_network_edges,
    network_at_time,
    TemporalDiffResult,
)

# Tier 3: JSON-LD bridge
from jsonld_ex.sl_network.jsonld_bridge import (
    network_from_jsonld_graph,
    network_to_jsonld_graph,
)

# Tier 1: Bayesian Network interop (optional: pip install jsonld-ex[bn])
# pgmpy is optional — bn_interop.py has no module-level pgmpy imports,
# so this import is safe even without pgmpy installed.  The functions
# themselves call _require_pgmpy() and raise ImportError with a clear
# message if pgmpy is missing at call time.
from jsonld_ex.sl_network.bn_interop import (
    from_bayesian_network,
    to_bayesian_network,
)

__all__ = [
    # Types
    "NodeType",
    "EdgeType",
    "SLNode",
    "SLEdge",
    "MultiParentEdge",
    "InferenceStep",
    "InferenceResult",
    # Tier 2 types
    "TrustEdge",
    "AttestationEdge",
    "TrustPropagationResult",
    # Network
    "SLNetwork",
    "CycleError",
    "NodeNotFoundError",
    # Counterfactuals
    "CounterfactualFn",
    "vacuous_counterfactual",
    "adversarial_counterfactual",
    "prior_counterfactual",
    "get_counterfactual_fn",
    "COUNTERFACTUAL_STRATEGIES",
    # Inference
    "infer_node",
    "infer_all",
    # Trust propagation and combined inference
    "propagate_trust",
    "infer_with_trust",
    # Tier 3: Temporal decay and point-in-time inference
    "decay_network_nodes",
    "decay_network_edges",
    "network_at_time",
    "TemporalDiffResult",
    # Tier 3: JSON-LD bridge
    "network_from_jsonld_graph",
    "network_to_jsonld_graph",
    # Tier 1: Bayesian Network interop
    "from_bayesian_network",
    "to_bayesian_network",
]
