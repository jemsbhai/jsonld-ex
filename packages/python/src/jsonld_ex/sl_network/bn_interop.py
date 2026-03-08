"""
Bayesian Network ↔ SLNetwork interoperability.

Provides bidirectional conversion between pgmpy BayesianNetworks and
SLNetwork, enabling:

1. **BN → SLNetwork:** Import existing Bayesian networks into the
   Subjective Logic framework, gaining uncertainty quantification
   that scalar probabilities cannot express.

2. **SLNetwork → BN:** Export SLNetworks back to pgmpy for comparison
   or integration with existing probabilistic inference pipelines.
   This conversion is *lossy* — the uncertainty dimension is collapsed
   into projected probabilities.

**pgmpy is an optional dependency** — all imports are guarded.  If pgmpy
is not installed, calling these functions raises ``ImportError`` with a
clear installation message.

Conversion Logic (BN → SL):
    - Root nodes: ``Opinion.from_evidence(pos=P(X=1)*N, neg=P(X=0)*N)``
      where N is the sample count (configurable per-node or globally).
    - Non-root nodes: vacuous opinion (placeholder for inference).
    - Single-parent edges: ``SLEdge`` with conditional from P(Y=1|X=1)
      and counterfactual from P(Y=1|X=0).
    - Multi-parent edges: ``MultiParentEdge`` with full conditional
      opinion table from the CPT.

Conversion Logic (SL → BN):
    - Project all opinions to probabilities via P(ω) = b + a·u.
    - Build CPTs from projected conditional/counterfactual probabilities.
    - Information loss: uncertainty dimension is collapsed (documented).

References:
    Jøsang, A. (2016). Subjective Logic, §3.2, §12.6.
    SLNetworks_plan.md §1.6.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from jsonld_ex.confidence_algebra import Opinion
from jsonld_ex.sl_network.network import SLNetwork
from jsonld_ex.sl_network.types import MultiParentEdge, SLEdge, SLNode


def _require_pgmpy() -> None:
    """Raise ImportError with installation instructions if pgmpy is missing."""
    try:
        import pgmpy  # noqa: F401
    except ImportError:
        raise ImportError(
            "pgmpy is required for Bayesian network interoperability. "
            "Install it with: pip install pgmpy"
        ) from None
    except TypeError:
        # pgmpy uses PEP 604 union types (`int | float`) internally,
        # which fail on Python < 3.10.
        raise ImportError(
            "pgmpy requires Python >= 3.10 (uses PEP 604 type syntax). "
            "Please upgrade Python or use a pgmpy version compatible "
            "with your Python version."
        ) from None


def _get_pgmpy_bn_class() -> type:
    """Return the pgmpy BayesianNetwork class, handling API changes.

    pgmpy >= 0.0.2 renamed BayesianNetwork to DiscreteBayesianNetwork.
    """
    try:
        from pgmpy.models import DiscreteBayesianNetwork
        return DiscreteBayesianNetwork
    except (ImportError, TypeError):
        from pgmpy.models import BayesianNetwork
        return BayesianNetwork


def from_bayesian_network(
    bn: Any,
    sample_counts: dict[str, int] | None = None,
    default_sample_count: int = 100,
    base_rate: float = 0.5,
) -> SLNetwork:
    """Convert a pgmpy BayesianNetwork to an SLNetwork.

    Args:
        bn: A ``pgmpy.models.BayesianNetwork`` instance.
        sample_counts: Optional per-node evidence counts.  Keys are node
            names, values are integer sample counts.  Nodes not listed
            use ``default_sample_count``.
        default_sample_count: Default evidence count for nodes not in
            ``sample_counts``.  Higher values produce more dogmatic
            (lower uncertainty) opinions.  Default 100.
        base_rate: Prior probability for all generated opinions.
            Default 0.5.

    Returns:
        An ``SLNetwork`` with content nodes and deduction edges
        corresponding to the BN structure.

    Raises:
        ImportError: If pgmpy is not installed.
        ValueError: If the BN is invalid or contains non-binary variables.

    Conversion details:
        - **Root nodes** get opinions via ``Opinion.from_evidence()``
          using the marginal probability from the CPT and the sample
          count.  This preserves the relationship between evidence
          quantity and epistemic uncertainty.
        - **Non-root nodes** get vacuous opinions ``(b=0, d=0, u=1)``
          as placeholders.  Their actual opinions are computed by
          SLNetwork inference from the graph structure, not baked in
          from BN marginals.
        - **Single-parent edges** become ``SLEdge`` with:
          - ``conditional`` from ``P(Y=1|X=1)``
          - ``counterfactual`` from ``P(Y=1|X=0)``
        - **Multi-parent edges** become ``MultiParentEdge`` with the
          full conditional opinion table from the CPT.

    Example::

        >>> from pgmpy.models import BayesianNetwork
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> bn = BayesianNetwork([("A", "B")])
        >>> # ... add CPDs ...
        >>> net = from_bayesian_network(bn, default_sample_count=1000)
    """
    _require_pgmpy()

    if default_sample_count < 0:
        raise ValueError(
            f"default_sample_count must be non-negative, "
            f"got {default_sample_count}"
        )

    if sample_counts is None:
        sample_counts = {}

    for node_name, count in sample_counts.items():
        if count < 0:
            raise ValueError(
                f"sample_counts[{node_name!r}] must be non-negative, "
                f"got {count}"
            )

    # Identify root nodes (no parents in the BN)
    root_nodes = set()
    for node in bn.nodes():
        parents = list(bn.get_parents(node))
        if not parents:
            root_nodes.add(node)

    # --- Step 1: Create SLNodes ---
    net = SLNetwork(name="from_bn")

    for node_name in bn.nodes():
        n = sample_counts.get(node_name, default_sample_count)

        if node_name in root_nodes:
            # Root node: opinion from marginal probability
            cpd = bn.get_cpds(node_name)
            # pgmpy CPD values: rows are states, columns are parent configs
            # For a root node (no parents), it's a single column
            # State 0 = False, State 1 = True
            values = cpd.get_values()
            p_true = float(values[1, 0])  # P(X=1)
            p_false = float(values[0, 0])  # P(X=0)

            opinion = Opinion.from_evidence(
                positive=p_true * n,
                negative=p_false * n,
                base_rate=base_rate,
            )
        else:
            # Non-root node: vacuous placeholder
            opinion = Opinion(
                belief=0.0,
                disbelief=0.0,
                uncertainty=1.0,
                base_rate=base_rate,
            )

        sl_node = SLNode(
            node_id=node_name,
            opinion=opinion,
            node_type="content",
            label=node_name,
        )
        net.add_node(sl_node)

    # --- Step 2: Create edges from CPTs ---
    for node_name in bn.nodes():
        parents = list(bn.get_parents(node_name))
        if not parents:
            continue  # Root nodes have no incoming edges

        cpd = bn.get_cpds(node_name)
        values = cpd.get_values()
        # values shape: (variable_card, product of parent cards)
        # For binary: (2, 2^num_parents)

        if len(parents) == 1:
            # Single parent → SLEdge
            parent = parents[0]
            n = sample_counts.get(node_name, default_sample_count)

            # pgmpy orders parent states as 0, 1
            # Column 0: parent=0 (False), Column 1: parent=1 (True)
            p_true_given_parent_true = float(values[1, 1])   # P(Y=1|X=1)
            p_false_given_parent_true = float(values[0, 1])  # P(Y=0|X=1)
            p_true_given_parent_false = float(values[1, 0])  # P(Y=1|X=0)
            p_false_given_parent_false = float(values[0, 0]) # P(Y=0|X=0)

            conditional = Opinion.from_evidence(
                positive=p_true_given_parent_true * n,
                negative=p_false_given_parent_true * n,
                base_rate=base_rate,
            )
            counterfactual = Opinion.from_evidence(
                positive=p_true_given_parent_false * n,
                negative=p_false_given_parent_false * n,
                base_rate=base_rate,
            )

            edge = SLEdge(
                source_id=parent,
                target_id=node_name,
                conditional=conditional,
                counterfactual=counterfactual,
                edge_type="deduction",
            )
            net.add_edge(edge)

        else:
            # Multiple parents → MultiParentEdge
            n = sample_counts.get(node_name, default_sample_count)

            # Build the conditional opinion table
            # pgmpy column ordering: rightmost parent varies fastest
            # We need to map column index to parent state tuple
            parent_ids = tuple(parents)
            num_parents = len(parents)
            conditionals: dict[tuple[bool, ...], Opinion] = {}

            # Get parent cardinalities in CPD column order
            # pgmpy orders columns with first evidence variable varying slowest
            parent_cards = [
                cpd.get_cardinality([p])[p] for p in parents
            ]

            for col_idx in range(values.shape[1]):
                # Decode column index to parent state tuple
                # pgmpy: first parent varies slowest (row-major)
                state_tuple = []
                remainder = col_idx
                for i in range(num_parents - 1, -1, -1):
                    card = parent_cards[i]
                    state_tuple.insert(0, remainder % card)
                    remainder //= card

                # Convert 0/1 states to False/True
                bool_tuple = tuple(bool(s) for s in state_tuple)

                p_true = float(values[1, col_idx])
                p_false = float(values[0, col_idx])

                conditionals[bool_tuple] = Opinion.from_evidence(
                    positive=p_true * n,
                    negative=p_false * n,
                    base_rate=base_rate,
                )

            multi_edge = MultiParentEdge(
                target_id=node_name,
                parent_ids=parent_ids,
                conditionals=conditionals,
                edge_type="deduction",
            )
            net.add_edge(multi_edge)

    return net


def to_bayesian_network(
    network: SLNetwork,
) -> Any:
    """Convert an SLNetwork to a pgmpy BayesianNetwork.

    This conversion is **lossy**: the uncertainty dimension of each
    opinion is collapsed into a projected probability via
    ``P(ω) = b + a·u``.  The resulting BN cannot distinguish between
    high-confidence and low-confidence opinions that happen to have
    the same projected probability.

    Args:
        network: An ``SLNetwork`` containing only content nodes and
            deduction edges (no trust/agent nodes).

    Returns:
        A ``pgmpy.models.DiscreteBayesianNetwork`` (or
        ``BayesianNetwork`` on older pgmpy) with CPDs derived from
        projected probabilities.

    Raises:
        ImportError: If pgmpy is not installed.

    Information loss:
        - Uncertainty ``u`` is collapsed into the projected probability.
        - Two opinions with identical ``P(ω)`` but different ``(b, d, u)``
          become indistinguishable in the output BN.
        - Round-trip ``BN → SL → BN`` preserves probabilities only in
          the dogmatic limit (large sample counts, ``u ≈ 0``).

    Example::

        >>> net = SLNetwork(name="example")
        >>> # ... build network ...
        >>> bn = to_bayesian_network(net)
        >>> bn.check_model()  # True
    """
    _require_pgmpy()

    from pgmpy.factors.discrete import TabularCPD

    BNClass = _get_pgmpy_bn_class()

    # --- Step 1: Build graph structure ---
    edges: list[tuple[str, str]] = []

    # Collect edges from SLEdges
    for (src, tgt) in network._edges:
        edges.append((src, tgt))

    # Collect edges from MultiParentEdges
    for tgt, mpe in network._multi_parent_edges.items():
        for pid in mpe.parent_ids:
            edges.append((pid, tgt))

    bn = BNClass(edges) if edges else BNClass()

    # Add isolated nodes (roots with no children, if any)
    for node_id in network._nodes:
        if node_id not in bn.nodes():
            bn.add_node(node_id)

    # --- Step 2: Build CPDs ---
    topo_order = network.topological_sort()

    for node_id in topo_order:
        node = network.get_node(node_id)
        parents = network.get_parents(node_id)

        if not parents:
            # Root node: marginal CPD from projected probability
            p_true = node.opinion.projected_probability()
            p_false = 1.0 - p_true
            cpd = TabularCPD(
                variable=node_id,
                variable_card=2,
                values=[[p_false], [p_true]],
            )
            bn.add_cpds(cpd)

        elif node_id in network._multi_parent_edges:
            # Multi-parent node: build CPT from MultiParentEdge
            mpe = network._multi_parent_edges[node_id]
            parent_ids = list(mpe.parent_ids)
            num_parents = len(parent_ids)
            num_cols = 2 ** num_parents

            # Build values matrix: shape (2, 2^num_parents)
            # Column ordering: first parent varies slowest (pgmpy convention)
            p_false_row = []
            p_true_row = []

            for col_idx in range(num_cols):
                # Decode column index to parent states
                # pgmpy: first evidence parent varies slowest
                state_tuple = []
                remainder = col_idx
                for i in range(num_parents - 1, -1, -1):
                    state_tuple.insert(0, bool(remainder % 2))
                    remainder //= 2

                bool_key = tuple(state_tuple)
                p_true = mpe.conditionals[bool_key].projected_probability()
                p_false = 1.0 - p_true
                p_false_row.append(p_false)
                p_true_row.append(p_true)

            cpd = TabularCPD(
                variable=node_id,
                variable_card=2,
                values=[p_false_row, p_true_row],
                evidence=parent_ids,
                evidence_card=[2] * num_parents,
            )
            bn.add_cpds(cpd)

        else:
            # Single-parent node: build CPT from SLEdge
            assert len(parents) == 1
            parent = parents[0]
            edge = network._edges[(parent, node_id)]

            p_true_given_true = edge.conditional.projected_probability()
            if edge.counterfactual is not None:
                p_true_given_false = edge.counterfactual.projected_probability()
            else:
                # If no counterfactual, use base rate as fallback
                p_true_given_false = node.opinion.base_rate

            cpd = TabularCPD(
                variable=node_id,
                variable_card=2,
                values=[
                    [1.0 - p_true_given_false, 1.0 - p_true_given_true],
                    [p_true_given_false, p_true_given_true],
                ],
                evidence=[parent],
                evidence_card=[2],
            )
            bn.add_cpds(cpd)

    return bn
