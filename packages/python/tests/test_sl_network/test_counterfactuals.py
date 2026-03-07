"""
Tests for counterfactual strategies (Tier 1, Step 3).

Covers:
    - All three built-in strategies (vacuous, adversarial, prior)
    - b+d+u=1 invariant for every strategy output
    - Mathematical properties specific to each strategy
    - Edge cases: dogmatic, vacuous, and extreme-base-rate inputs
    - Custom callable support
    - Strategy registry and get_counterfactual_fn resolver
    - Integration with the existing deduce() operator

TDD: These tests define the contract that counterfactuals.py must satisfy.
"""

from __future__ import annotations

import pytest

from jsonld_ex.confidence_algebra import Opinion, deduce
from jsonld_ex.sl_network.counterfactuals import (
    COUNTERFACTUAL_STRATEGIES,
    CounterfactualFn,
    adversarial_counterfactual,
    get_counterfactual_fn,
    prior_counterfactual,
    vacuous_counterfactual,
)


# ═══════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def high_belief() -> Opinion:
    """Strong positive conditional: ω_{y|x}."""
    return Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1)


@pytest.fixture
def moderate() -> Opinion:
    return Opinion(belief=0.5, disbelief=0.2, uncertainty=0.3)


@pytest.fixture
def vacuous_op() -> Opinion:
    return Opinion(belief=0.0, disbelief=0.0, uncertainty=1.0)


@pytest.fixture
def dogmatic_true() -> Opinion:
    return Opinion(belief=1.0, disbelief=0.0, uncertainty=0.0)


@pytest.fixture
def dogmatic_false() -> Opinion:
    return Opinion(belief=0.0, disbelief=1.0, uncertainty=0.0)


@pytest.fixture
def custom_base_rate() -> Opinion:
    """Opinion with non-default base rate."""
    return Opinion(belief=0.6, disbelief=0.2, uncertainty=0.2, base_rate=0.7)


# ═══════════════════════════════════════════════════════════════════
# VACUOUS COUNTERFACTUAL
# ═══════════════════════════════════════════════════════════════════


class TestVacuousCounterfactual:
    """Test the vacuous (default) counterfactual strategy."""

    def test_returns_vacuous_opinion(self, high_belief: Opinion) -> None:
        cf = vacuous_counterfactual(high_belief)
        assert cf.belief == 0.0
        assert cf.disbelief == 0.0
        assert cf.uncertainty == 1.0

    def test_preserves_base_rate(self, high_belief: Opinion) -> None:
        cf = vacuous_counterfactual(high_belief)
        assert cf.base_rate == high_belief.base_rate

    def test_preserves_custom_base_rate(self, custom_base_rate: Opinion) -> None:
        cf = vacuous_counterfactual(custom_base_rate)
        assert cf.base_rate == 0.7

    def test_additivity_invariant(self, high_belief: Opinion) -> None:
        cf = vacuous_counterfactual(high_belief)
        assert abs(cf.belief + cf.disbelief + cf.uncertainty - 1.0) < 1e-12

    def test_from_dogmatic_true(self, dogmatic_true: Opinion) -> None:
        """Dogmatic true conditional → vacuous counterfactual."""
        cf = vacuous_counterfactual(dogmatic_true)
        assert cf.uncertainty == 1.0

    def test_from_dogmatic_false(self, dogmatic_false: Opinion) -> None:
        cf = vacuous_counterfactual(dogmatic_false)
        assert cf.uncertainty == 1.0

    def test_from_vacuous(self, vacuous_op: Opinion) -> None:
        """Vacuous input → vacuous output (idempotent-like)."""
        cf = vacuous_counterfactual(vacuous_op)
        assert cf.uncertainty == 1.0

    def test_projected_probability_equals_base_rate(
        self, high_belief: Opinion
    ) -> None:
        """P(ω) = b + a·u = 0 + a·1 = a for vacuous opinion."""
        cf = vacuous_counterfactual(high_belief)
        assert abs(cf.projected_probability() - cf.base_rate) < 1e-12


# ═══════════════════════════════════════════════════════════════════
# ADVERSARIAL COUNTERFACTUAL
# ═══════════════════════════════════════════════════════════════════


class TestAdversarialCounterfactual:
    """Test the adversarial counterfactual strategy."""

    def test_swaps_belief_disbelief(self, high_belief: Opinion) -> None:
        cf = adversarial_counterfactual(high_belief)
        assert cf.belief == high_belief.disbelief
        assert cf.disbelief == high_belief.belief

    def test_preserves_uncertainty(self, high_belief: Opinion) -> None:
        cf = adversarial_counterfactual(high_belief)
        assert cf.uncertainty == high_belief.uncertainty

    def test_inverts_base_rate(self, high_belief: Opinion) -> None:
        cf = adversarial_counterfactual(high_belief)
        assert abs(cf.base_rate - (1.0 - high_belief.base_rate)) < 1e-12

    def test_inverts_custom_base_rate(self, custom_base_rate: Opinion) -> None:
        cf = adversarial_counterfactual(custom_base_rate)
        assert abs(cf.base_rate - 0.3) < 1e-12  # 1.0 - 0.7

    def test_additivity_invariant(self, high_belief: Opinion) -> None:
        cf = adversarial_counterfactual(high_belief)
        assert abs(cf.belief + cf.disbelief + cf.uncertainty - 1.0) < 1e-12

    def test_double_adversarial_is_identity(self, high_belief: Opinion) -> None:
        """Applying adversarial twice returns the original opinion."""
        cf1 = adversarial_counterfactual(high_belief)
        cf2 = adversarial_counterfactual(cf1)
        assert abs(cf2.belief - high_belief.belief) < 1e-12
        assert abs(cf2.disbelief - high_belief.disbelief) < 1e-12
        assert abs(cf2.uncertainty - high_belief.uncertainty) < 1e-12
        assert abs(cf2.base_rate - high_belief.base_rate) < 1e-12

    def test_from_dogmatic_true(self, dogmatic_true: Opinion) -> None:
        """Dogmatic true → dogmatic false."""
        cf = adversarial_counterfactual(dogmatic_true)
        assert cf.belief == 0.0
        assert cf.disbelief == 1.0
        assert cf.uncertainty == 0.0

    def test_from_dogmatic_false(self, dogmatic_false: Opinion) -> None:
        """Dogmatic false → dogmatic true."""
        cf = adversarial_counterfactual(dogmatic_false)
        assert cf.belief == 1.0
        assert cf.disbelief == 0.0
        assert cf.uncertainty == 0.0

    def test_from_vacuous(self, vacuous_op: Opinion) -> None:
        """Vacuous → vacuous (swapping b=0, d=0 gives same)."""
        cf = adversarial_counterfactual(vacuous_op)
        assert cf.belief == 0.0
        assert cf.disbelief == 0.0
        assert cf.uncertainty == 1.0

    def test_symmetric_opinion(self) -> None:
        """For symmetric opinion (b=d), adversarial is self-inverse on b,d."""
        sym = Opinion(belief=0.4, disbelief=0.4, uncertainty=0.2)
        cf = adversarial_counterfactual(sym)
        assert abs(cf.belief - 0.4) < 1e-12
        assert abs(cf.disbelief - 0.4) < 1e-12

    def test_projected_probability_inverts(self, high_belief: Opinion) -> None:
        """P(adversarial) + P(original) should relate via base rate inversion.

        P(ω) = b + a·u.  For adversarial: P(cf) = d + (1-a)·u.
        P(ω) + P(cf) = b + a·u + d + (1-a)·u = b + d + u = 1.
        """
        cf = adversarial_counterfactual(high_belief)
        p_orig = high_belief.projected_probability()
        p_cf = cf.projected_probability()
        assert abs(p_orig + p_cf - 1.0) < 1e-12


# ═══════════════════════════════════════════════════════════════════
# PRIOR COUNTERFACTUAL
# ═══════════════════════════════════════════════════════════════════


class TestPriorCounterfactual:
    """Test the prior-based counterfactual strategy."""

    def test_identical_to_vacuous_in_binomial(self, high_belief: Opinion) -> None:
        """In the binomial case, prior == vacuous."""
        cf_vacuous = vacuous_counterfactual(high_belief)
        cf_prior = prior_counterfactual(high_belief)
        assert cf_prior.belief == cf_vacuous.belief
        assert cf_prior.disbelief == cf_vacuous.disbelief
        assert cf_prior.uncertainty == cf_vacuous.uncertainty
        assert cf_prior.base_rate == cf_vacuous.base_rate

    def test_returns_vacuous_opinion(self, moderate: Opinion) -> None:
        cf = prior_counterfactual(moderate)
        assert cf.belief == 0.0
        assert cf.disbelief == 0.0
        assert cf.uncertainty == 1.0

    def test_preserves_base_rate(self, custom_base_rate: Opinion) -> None:
        cf = prior_counterfactual(custom_base_rate)
        assert cf.base_rate == 0.7

    def test_additivity_invariant(self, moderate: Opinion) -> None:
        cf = prior_counterfactual(moderate)
        assert abs(cf.belief + cf.disbelief + cf.uncertainty - 1.0) < 1e-12


# ═══════════════════════════════════════════════════════════════════
# STRATEGY REGISTRY
# ═══════════════════════════════════════════════════════════════════


class TestStrategyRegistry:
    """Test COUNTERFACTUAL_STRATEGIES dict."""

    def test_contains_all_three(self) -> None:
        assert "vacuous" in COUNTERFACTUAL_STRATEGIES
        assert "adversarial" in COUNTERFACTUAL_STRATEGIES
        assert "prior" in COUNTERFACTUAL_STRATEGIES

    def test_exactly_three_entries(self) -> None:
        assert len(COUNTERFACTUAL_STRATEGIES) == 3

    def test_values_are_callable(self) -> None:
        for name, fn in COUNTERFACTUAL_STRATEGIES.items():
            assert callable(fn), f"{name} is not callable"

    def test_each_produces_valid_opinion(self, high_belief: Opinion) -> None:
        for name, fn in COUNTERFACTUAL_STRATEGIES.items():
            cf = fn(high_belief)
            assert isinstance(cf, Opinion), f"{name} didn't return Opinion"
            total = cf.belief + cf.disbelief + cf.uncertainty
            assert abs(total - 1.0) < 1e-12, (
                f"{name}: b+d+u={total}"
            )


# ═══════════════════════════════════════════════════════════════════
# get_counterfactual_fn RESOLVER
# ═══════════════════════════════════════════════════════════════════


class TestGetCounterfactualFn:
    """Test the strategy resolver."""

    def test_resolve_vacuous(self) -> None:
        fn = get_counterfactual_fn("vacuous")
        assert fn is vacuous_counterfactual

    def test_resolve_adversarial(self) -> None:
        fn = get_counterfactual_fn("adversarial")
        assert fn is adversarial_counterfactual

    def test_resolve_prior(self) -> None:
        fn = get_counterfactual_fn("prior")
        assert fn is prior_counterfactual

    def test_unknown_name_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown.*'causal'"):
            get_counterfactual_fn("causal")

    def test_error_lists_available(self) -> None:
        with pytest.raises(ValueError, match="adversarial"):
            get_counterfactual_fn("bad_name")

    def test_passthrough_callable(self) -> None:
        """Custom callable is returned directly."""
        custom = lambda op: Opinion(0.5, 0.3, 0.2)  # noqa: E731
        fn = get_counterfactual_fn(custom)
        assert fn is custom

    def test_passthrough_named_function(self) -> None:
        def my_strategy(conditional: Opinion) -> Opinion:
            return Opinion(0.0, 0.0, 1.0, base_rate=conditional.base_rate)

        fn = get_counterfactual_fn(my_strategy)
        assert fn is my_strategy

    def test_non_string_non_callable_raises(self) -> None:
        with pytest.raises(TypeError, match="string or callable"):
            get_counterfactual_fn(42)  # type: ignore[arg-type]

    def test_none_raises(self) -> None:
        with pytest.raises(TypeError, match="string or callable"):
            get_counterfactual_fn(None)  # type: ignore[arg-type]


# ═══════════════════════════════════════════════════════════════════
# CUSTOM COUNTERFACTUAL FUNCTIONS
# ═══════════════════════════════════════════════════════════════════


class TestCustomCounterfactual:
    """Test user-defined counterfactual functions."""

    def test_custom_fn_accepted_by_resolver(self) -> None:
        def half_belief(conditional: Opinion) -> Opinion:
            return Opinion(
                belief=conditional.belief / 2,
                disbelief=conditional.disbelief / 2,
                uncertainty=1.0 - conditional.belief / 2 - conditional.disbelief / 2,
                base_rate=conditional.base_rate,
            )

        fn = get_counterfactual_fn(half_belief)
        result = fn(Opinion(0.8, 0.1, 0.1))
        assert abs(result.belief - 0.4) < 1e-12
        assert abs(result.disbelief - 0.05) < 1e-12
        assert abs(result.belief + result.disbelief + result.uncertainty - 1.0) < 1e-12

    def test_custom_lambda_works(self) -> None:
        mild_counter: CounterfactualFn = lambda op: Opinion(  # noqa: E731
            belief=op.belief * 0.1,
            disbelief=op.disbelief * 0.1,
            uncertainty=1.0 - op.belief * 0.1 - op.disbelief * 0.1,
            base_rate=op.base_rate,
        )
        fn = get_counterfactual_fn(mild_counter)
        result = fn(Opinion(0.8, 0.1, 0.1))
        assert abs(result.belief - 0.08) < 1e-12
        total = result.belief + result.disbelief + result.uncertainty
        assert abs(total - 1.0) < 1e-12


# ═══════════════════════════════════════════════════════════════════
# MATHEMATICAL PROPERTIES
# ═══════════════════════════════════════════════════════════════════


class TestMathematicalProperties:
    """Verify key mathematical properties of counterfactual strategies."""

    @pytest.mark.parametrize("strategy_name", ["vacuous", "adversarial", "prior"])
    def test_additivity_invariant_parametrized(self, strategy_name: str) -> None:
        """b+d+u=1 must hold for all strategy outputs across varied inputs."""
        fn = COUNTERFACTUAL_STRATEGIES[strategy_name]
        test_opinions = [
            Opinion(0.8, 0.1, 0.1),
            Opinion(0.0, 0.0, 1.0),
            Opinion(1.0, 0.0, 0.0),
            Opinion(0.0, 1.0, 0.0),
            Opinion(0.33, 0.33, 0.34),
            Opinion(0.5, 0.2, 0.3, base_rate=0.9),
            Opinion(0.1, 0.8, 0.1, base_rate=0.1),
        ]
        for op in test_opinions:
            cf = fn(op)
            total = cf.belief + cf.disbelief + cf.uncertainty
            assert abs(total - 1.0) < 1e-12, (
                f"{strategy_name} on {op}: b+d+u={total}"
            )

    def test_adversarial_is_involution(self) -> None:
        """adversarial(adversarial(ω)) = ω for any opinion."""
        opinions = [
            Opinion(0.8, 0.1, 0.1),
            Opinion(0.0, 0.5, 0.5),
            Opinion(0.3, 0.3, 0.4, base_rate=0.7),
        ]
        for op in opinions:
            double = adversarial_counterfactual(adversarial_counterfactual(op))
            assert abs(double.belief - op.belief) < 1e-12
            assert abs(double.disbelief - op.disbelief) < 1e-12
            assert abs(double.uncertainty - op.uncertainty) < 1e-12
            assert abs(double.base_rate - op.base_rate) < 1e-12

    def test_adversarial_p_complement(self) -> None:
        """P(adversarial(ω)) = 1 - P(ω) for any opinion.

        Proof: P(ω) = b + a·u.
               P(adv(ω)) = d + (1-a)·u.
               Sum = b + d + u = 1.
        """
        opinions = [
            Opinion(0.8, 0.1, 0.1),
            Opinion(0.0, 0.5, 0.5, base_rate=0.3),
            Opinion(0.4, 0.4, 0.2, base_rate=0.8),
        ]
        for op in opinions:
            cf = adversarial_counterfactual(op)
            p_sum = op.projected_probability() + cf.projected_probability()
            assert abs(p_sum - 1.0) < 1e-12, f"P(ω)+P(adv(ω))={p_sum}"

    def test_vacuous_projected_prob_is_base_rate(self) -> None:
        """P(vacuous(ω)) = base_rate for any input."""
        for a in [0.1, 0.3, 0.5, 0.7, 0.9]:
            op = Opinion(0.6, 0.2, 0.2, base_rate=a)
            cf = vacuous_counterfactual(op)
            assert abs(cf.projected_probability() - a) < 1e-12


# ═══════════════════════════════════════════════════════════════════
# INTEGRATION WITH deduce()
# ═══════════════════════════════════════════════════════════════════


class TestDeduceIntegration:
    """Verify counterfactual strategies integrate with the existing deduce().

    This ensures the counterfactual output is a valid Opinion that
    deduce() accepts, and that different strategies produce different
    (and correct) deduced opinions.
    """

    def test_deduce_with_vacuous_counterfactual(self) -> None:
        """deduce() accepts the output of vacuous_counterfactual."""
        antecedent = Opinion(0.7, 0.1, 0.2)
        conditional = Opinion(0.9, 0.05, 0.05)
        cf = vacuous_counterfactual(conditional)
        result = deduce(antecedent, conditional, cf)
        total = result.belief + result.disbelief + result.uncertainty
        assert abs(total - 1.0) < 1e-12

    def test_deduce_with_adversarial_counterfactual(self) -> None:
        """deduce() accepts the output of adversarial_counterfactual."""
        antecedent = Opinion(0.7, 0.1, 0.2)
        conditional = Opinion(0.9, 0.05, 0.05)
        cf = adversarial_counterfactual(conditional)
        result = deduce(antecedent, conditional, cf)
        total = result.belief + result.disbelief + result.uncertainty
        assert abs(total - 1.0) < 1e-12

    def test_deduce_with_prior_counterfactual(self) -> None:
        """deduce() accepts the output of prior_counterfactual."""
        antecedent = Opinion(0.7, 0.1, 0.2)
        conditional = Opinion(0.9, 0.05, 0.05)
        cf = prior_counterfactual(conditional)
        result = deduce(antecedent, conditional, cf)
        total = result.belief + result.disbelief + result.uncertainty
        assert abs(total - 1.0) < 1e-12

    def test_strategies_produce_different_deductions(self) -> None:
        """Different counterfactual strategies yield different deduced opinions.

        This demonstrates that the choice of strategy is meaningful
        and not degenerate.
        """
        antecedent = Opinion(0.5, 0.3, 0.2)  # Uncertain antecedent
        conditional = Opinion(0.8, 0.1, 0.1)

        cf_vac = vacuous_counterfactual(conditional)
        cf_adv = adversarial_counterfactual(conditional)

        result_vac = deduce(antecedent, conditional, cf_vac)
        result_adv = deduce(antecedent, conditional, cf_adv)

        # They should differ because d_x > 0 and u_x > 0, so the
        # counterfactual contributes to the deduction
        assert abs(result_vac.belief - result_adv.belief) > 0.01, (
            "Vacuous and adversarial should produce different deductions "
            "when antecedent has non-zero disbelief and uncertainty"
        )

    def test_dogmatic_antecedent_ignores_counterfactual(self) -> None:
        """When antecedent is dogmatic true (d=0, u=0), counterfactual
        has no effect on the deduced opinion.

        From the deduction formula:
            c_y = b_x · c_{y|x} + d_x · c_{y|¬x} + u_x · (...)
        When b_x=1, d_x=0, u_x=0:
            c_y = 1 · c_{y|x} + 0 + 0 = c_{y|x}
        """
        antecedent = Opinion(1.0, 0.0, 0.0)  # Dogmatic true
        conditional = Opinion(0.8, 0.1, 0.1)

        cf_vac = vacuous_counterfactual(conditional)
        cf_adv = adversarial_counterfactual(conditional)

        result_vac = deduce(antecedent, conditional, cf_vac)
        result_adv = deduce(antecedent, conditional, cf_adv)

        # Both should equal the conditional
        assert abs(result_vac.belief - conditional.belief) < 1e-12
        assert abs(result_adv.belief - conditional.belief) < 1e-12
        assert abs(result_vac.disbelief - conditional.disbelief) < 1e-12
        assert abs(result_adv.disbelief - conditional.disbelief) < 1e-12


# ═══════════════════════════════════════════════════════════════════
# MODULE IMPORTS
# ═══════════════════════════════════════════════════════════════════


class TestModuleImports:
    """Verify counterfactual module is importable from package."""

    def test_import_from_sl_network(self) -> None:
        from jsonld_ex.sl_network import (
            COUNTERFACTUAL_STRATEGIES,
            CounterfactualFn,
            adversarial_counterfactual,
            get_counterfactual_fn,
            prior_counterfactual,
            vacuous_counterfactual,
        )
        assert vacuous_counterfactual is not None
        assert adversarial_counterfactual is not None
        assert prior_counterfactual is not None
        assert get_counterfactual_fn is not None
