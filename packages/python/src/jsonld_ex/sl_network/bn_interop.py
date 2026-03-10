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
from jsonld_ex.multinomial_algebra import MultinomialOpinion
from jsonld_ex.sl_network.network import SLNetwork
from jsonld_ex.sl_network.types import (
    MultiParentEdge,
    MultiParentMultinomialEdge,
    MultinomialEdge,
    SLEdge,
    SLNode,
)


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


def _get_variable_card(bn: Any, node_name: str) -> int:
    """Return the cardinality of a variable in the BN."""
    cpd = bn.get_cpds(node_name)
    return int(cpd.get_values().shape[0])


def _multinomial_opinion_from_cpd_column(
    values: Any,
    col_idx: int,
    child_card: int,
    n: int,
) -> MultinomialOpinion:
    """Build a MultinomialOpinion from one column of a CPD matrix.

    Args:
        values: The CPD values matrix (shape: child_card x num_parent_configs).
        col_idx: Which column (parent config) to read.
        child_card: Number of child states.
        n: Sample count for evidence weighting.

    Returns:
        MultinomialOpinion via from_evidence().
    """
    evidence: dict[str, int | float] = {}
    for state_idx in range(child_card):
        p = float(values[state_idx, col_idx])
        evidence[str(state_idx)] = p * n
    return MultinomialOpinion.from_evidence(evidence)


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
        ValueError: If the BN is invalid.

    Conversion details:
        - **Binary root nodes** (card=2) get ``Opinion.from_evidence()``.
        - **K-ary root nodes** (card>2) get ``MultinomialOpinion.from_evidence()``
          stored in ``SLNode.multinomial_opinion``.
        - **Non-root nodes** get vacuous opinions as placeholders.
        - **Binary single-parent edges** become ``SLEdge``.
        - **K-ary single-parent edges** (parent or child card>2) become
          ``MultinomialEdge``.
        - **Binary multi-parent edges** become ``MultiParentEdge``.
        - **K-ary multi-parent edges** (any card>2) become
          ``MultiParentMultinomialEdge``.

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
        cpd = bn.get_cpds(node_name)
        card = int(cpd.get_values().shape[0])

        multinomial_opinion = None

        if card > 2:
            # K-ary variable: MultinomialOpinion
            if node_name in root_nodes:
                values = cpd.get_values()
                evidence: dict[str, int | float] = {
                    str(i): float(values[i, 0]) * n
                    for i in range(card)
                }
                multinomial_opinion = MultinomialOpinion.from_evidence(
                    evidence
                )
            else:
                # Vacuous multinomial: uniform base rates, u=1
                multinomial_opinion = MultinomialOpinion(
                    beliefs={str(i): 0.0 for i in range(card)},
                    uncertainty=1.0,
                    base_rates={str(i): 1.0 / card for i in range(card)},
                )
            # Binomial opinion is vacuous placeholder for backward compat
            opinion = Opinion(
                belief=0.0,
                disbelief=0.0,
                uncertainty=1.0,
                base_rate=base_rate,
            )
        elif node_name in root_nodes:
            # Binary root node: opinion from marginal probability
            values = cpd.get_values()
            p_true = float(values[1, 0])   # P(X=1)
            p_false = float(values[0, 0])  # P(X=0)
            opinion = Opinion.from_evidence(
                positive=p_true * n,
                negative=p_false * n,
                base_rate=base_rate,
            )
        else:
            # Binary non-root node: vacuous placeholder
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
            multinomial_opinion=multinomial_opinion,
        )
        net.add_node(sl_node)

    # --- Step 2: Create edges from CPTs ---
    for node_name in bn.nodes():
        parents = list(bn.get_parents(node_name))
        if not parents:
            continue  # Root nodes have no incoming edges

        cpd = bn.get_cpds(node_name)
        values = cpd.get_values()
        child_card = int(values.shape[0])
        n = sample_counts.get(node_name, default_sample_count)

        # Determine parent cardinalities
        parent_cards = [
            cpd.get_cardinality([p])[p] for p in parents
        ]

        # Check if any variable is k-ary (card > 2)
        any_kary = child_card > 2 or any(c > 2 for c in parent_cards)

        if len(parents) == 1:
            parent = parents[0]
            parent_card = parent_cards[0]

            if any_kary:
                # K-ary single parent → MultinomialEdge
                # One MultinomialOpinion per parent state
                cond_map: dict[str, MultinomialOpinion] = {}
                for col_idx in range(parent_card):
                    cond_map[str(col_idx)] = \
                        _multinomial_opinion_from_cpd_column(
                            values, col_idx, child_card, n,
                        )

                edge = MultinomialEdge(
                    source_id=parent,
                    target_id=node_name,
                    conditionals=cond_map,
                    edge_type="deduction",
                )
                net.add_edge(edge)
            else:
                # Binary single parent → SLEdge
                p_true_given_parent_true = float(values[1, 1])
                p_false_given_parent_true = float(values[0, 1])
                p_true_given_parent_false = float(values[1, 0])
                p_false_given_parent_false = float(values[0, 0])

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
            # Multiple parents
            parent_ids = tuple(parents)
            num_parents = len(parents)

            if any_kary:
                # K-ary multi-parent → MultiParentMultinomialEdge
                mn_conditionals: dict[
                    tuple[str, ...], MultinomialOpinion
                ] = {}

                for col_idx in range(values.shape[1]):
                    # Decode column index to parent state tuple (str names)
                    state_list: list[str] = []
                    remainder = col_idx
                    for i in range(num_parents - 1, -1, -1):
                        pc = parent_cards[i]
                        state_list.insert(0, str(remainder % pc))
                        remainder //= pc

                    key = tuple(state_list)
                    mn_conditionals[key] = \
                        _multinomial_opinion_from_cpd_column(
                            values, col_idx, child_card, n,
                        )

                multi_edge = MultiParentMultinomialEdge(
                    target_id=node_name,
                    parent_ids=parent_ids,
                    conditionals=mn_conditionals,
                    edge_type="deduction",
                )
                net.add_edge(multi_edge)
            else:
                # Binary multi-parent → MultiParentEdge
                bin_conditionals: dict[tuple[bool, ...], Opinion] = {}

                for col_idx in range(values.shape[1]):
                    state_tuple: list[int] = []
                    remainder = col_idx
                    for i in range(num_parents - 1, -1, -1):
                        pc = parent_cards[i]
                        state_tuple.insert(0, remainder % pc)
                        remainder //= pc

                    bool_tuple = tuple(bool(s) for s in state_tuple)
                    p_true = float(values[1, col_idx])
                    p_false = float(values[0, col_idx])

                    bin_conditionals[bool_tuple] = Opinion.from_evidence(
                        positive=p_true * n,
                        negative=p_false * n,
                        base_rate=base_rate,
                    )

                multi_edge = MultiParentEdge(
                    target_id=node_name,
                    parent_ids=parent_ids,
                    conditionals=bin_conditionals,
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

    # Collect edges from MultinomialEdges
    for (src, tgt) in network._multinomial_edges:
        edges.append((src, tgt))

    # Collect edges from MultiParentEdges (binary)
    for tgt, mpe in network._multi_parent_edges.items():
        for pid in mpe.parent_ids:
            edges.append((pid, tgt))

    # Collect edges from MultiParentMultinomialEdges
    for tgt, mpe in network._multi_parent_multinomial_edges.items():
        for pid in mpe.parent_ids:
            edges.append((pid, tgt))

    bn = BNClass(edges) if edges else BNClass()

    # Add isolated nodes (roots with no children, if any)
    for node_id in network._nodes:
        if node_id not in bn.nodes():
            bn.add_node(node_id)

    # --- Helper: determine node cardinality ---
    def _node_card(nid: str) -> int:
        nd = network.get_node(nid)
        if nd.is_multinomial:
            return nd.multinomial_opinion.cardinality
        return 2  # binary

    # --- Step 2: Build CPDs ---
    topo_order = network.topological_sort()

    for node_id in topo_order:
        node = network.get_node(node_id)
        parents = network.get_parents(node_id)
        child_card = _node_card(node_id)

        if not parents:
            # Root node: marginal CPD
            if node.is_multinomial:
                pp = node.multinomial_opinion.projected_probability()
                # Rows ordered by sorted state name ("0", "1", "2", ...)
                values = [
                    [pp[s]] for s in sorted(pp.keys())
                ]
            else:
                p_true = node.opinion.projected_probability()
                values = [[1.0 - p_true], [p_true]]

            cpd = TabularCPD(
                variable=node_id,
                variable_card=child_card,
                values=values,
            )
            bn.add_cpds(cpd)

        elif node_id in network._multi_parent_multinomial_edges:
            # K-ary multi-parent: MultiParentMultinomialEdge
            mpe = network._multi_parent_multinomial_edges[node_id]
            parent_ids = list(mpe.parent_ids)
            num_parents = len(parent_ids)
            parent_cards = [_node_card(pid) for pid in parent_ids]
            num_cols = 1
            for pc in parent_cards:
                num_cols *= pc

            # Build values matrix: shape (child_card, num_cols)
            # pgmpy column ordering: first parent varies slowest
            rows: list[list[float]] = [[] for _ in range(child_card)]

            for col_idx in range(num_cols):
                # Decode column index to parent state name tuple
                state_list: list[str] = []
                remainder = col_idx
                for i in range(num_parents - 1, -1, -1):
                    pc = parent_cards[i]
                    state_list.insert(0, str(remainder % pc))
                    remainder //= pc

                key = tuple(state_list)
                pp = mpe.conditionals[key].projected_probability()
                for row_idx, state in enumerate(sorted(pp.keys())):
                    rows[row_idx].append(pp[state])

            cpd = TabularCPD(
                variable=node_id,
                variable_card=child_card,
                values=rows,
                evidence=parent_ids,
                evidence_card=parent_cards,
            )
            bn.add_cpds(cpd)

        elif node_id in network._multi_parent_edges:
            # Binary multi-parent: MultiParentEdge
            mpe = network._multi_parent_edges[node_id]
            parent_ids = list(mpe.parent_ids)
            num_parents = len(parent_ids)
            num_cols = 2 ** num_parents

            p_false_row: list[float] = []
            p_true_row: list[float] = []

            for col_idx in range(num_cols):
                state_tuple: list[bool] = []
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

        elif len(parents) == 1 and network.has_multinomial_edge(
            parents[0], node_id
        ):
            # K-ary single-parent: MultinomialEdge
            parent = parents[0]
            mn_edge = network.get_multinomial_edge(parent, node_id)
            parent_card = _node_card(parent)

            rows = [[] for _ in range(child_card)]
            for col_idx in range(parent_card):
                state_name = str(col_idx)
                pp = mn_edge.conditionals[state_name].projected_probability()
                for row_idx, state in enumerate(sorted(pp.keys())):
                    rows[row_idx].append(pp[state])

            cpd = TabularCPD(
                variable=node_id,
                variable_card=child_card,
                values=rows,
                evidence=[parent],
                evidence_card=[parent_card],
            )
            bn.add_cpds(cpd)

        else:
            # Binary single-parent: SLEdge
            assert len(parents) == 1
            parent = parents[0]
            edge = network._edges[(parent, node_id)]

            p_true_given_true = edge.conditional.projected_probability()
            if edge.counterfactual is not None:
                p_true_given_false = (
                    edge.counterfactual.projected_probability()
                )
            else:
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
