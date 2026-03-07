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

__all__ = [
    # Types
    "NodeType",
    "EdgeType",
    "SLNode",
    "SLEdge",
    "MultiParentEdge",
    "InferenceStep",
    "InferenceResult",
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
]
