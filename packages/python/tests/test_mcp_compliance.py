"""Tests for the jsonld-ex MCP server — Compliance Algebra tools.

Tests the 10 compliance algebra MCP tools, 2 resources, and 2 prompts
directly (no transport needed). Verifies correct delegation to
jsonld_ex.compliance_algebra functions and proper serialization.
"""

import json
import math
import pytest

# Guard: skip entire module if `mcp` is not installed.
mcp = pytest.importorskip("mcp", reason="MCP SDK not installed (pip install mcp)")

from jsonld_ex.mcp.server import mcp as mcp_server  # noqa: E402

TOL = 1e-9


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _get_tool(name: str):
    """Retrieve an MCP tool function by name."""
    tool_manager = mcp_server._tool_manager
    tool = tool_manager._tools.get(name)
    if tool is None:
        available = sorted(tool_manager._tools.keys())
        raise KeyError(f"Tool {name!r} not found. Available: {available}")
    return tool.fn


def _get_resource(uri: str):
    """Retrieve an MCP resource function by URI."""
    resource_manager = mcp_server._resource_manager
    # FastMCP may key resources by URI string or AnyUrl
    for key, res in resource_manager._resources.items():
        key_str = str(key) if not isinstance(key, str) else key
        res_uri = str(getattr(res, 'uri', '')) if hasattr(res, 'uri') else ''
        if uri in (key_str, res_uri):
            return res.fn
    available = [str(k) for k in resource_manager._resources.keys()]
    raise KeyError(f"Resource {uri!r} not found. Available: {available}")


def _get_prompt(name: str):
    """Retrieve an MCP prompt function by name."""
    prompt_manager = mcp_server._prompt_manager
    prompt = prompt_manager._prompts.get(name)
    if prompt is None:
        available = sorted(prompt_manager._prompts.keys())
        raise KeyError(f"Prompt {name!r} not found. Available: {available}")
    return prompt.fn


def _make_opinion_json(**kwargs) -> str:
    """Create a JSON string for an opinion dict."""
    d = {"belief": 0.5, "disbelief": 0.2, "uncertainty": 0.3, "base_rate": 0.5}
    d.update(kwargs)
    return json.dumps(d)


def _assert_valid_compliance(result: dict, msg: str = ""):
    """Assert result is a valid compliance opinion dict."""
    assert "lawfulness" in result, f"Missing lawfulness {msg}"
    assert "violation" in result, f"Missing violation {msg}"
    assert "uncertainty" in result, f"Missing uncertainty {msg}"
    assert "base_rate" in result, f"Missing base_rate {msg}"
    assert "projected_probability" in result, f"Missing projected_probability {msg}"
    l, v, u = result["lawfulness"], result["violation"], result["uncertainty"]
    assert l >= -TOL, f"lawfulness < 0: {l} {msg}"
    assert v >= -TOL, f"violation < 0: {v} {msg}"
    assert u >= -TOL, f"uncertainty < 0: {u} {msg}"
    assert abs(l + v + u - 1.0) < TOL, f"l+v+u={l+v+u} != 1 {msg}"


# ═══════════════════════════════════════════════════════════════════
# Registration Tests
# ═══════════════════════════════════════════════════════════════════


class TestComplianceToolRegistration:
    """Verify all compliance tools, resources, and prompts are registered."""

    COMPLIANCE_TOOLS = [
        "create_compliance_opinion",
        "jurisdictional_meet",
        "compliance_propagation",
        "consent_validity",
        "withdrawal_override",
        "expiry_trigger",
        "review_due_trigger",
        "regulatory_change_trigger",
        "erasure_scope_opinion",
        "residual_contamination",
    ]

    def test_all_compliance_tools_registered(self):
        """Every compliance tool must be present."""
        tool_manager = mcp_server._tool_manager
        registered = set(tool_manager._tools.keys())
        for name in self.COMPLIANCE_TOOLS:
            assert name in registered, f"Tool '{name}' not registered"

    def test_total_tool_count(self):
        """41 existing + 10 compliance = 51 tools."""
        tool_manager = mcp_server._tool_manager
        assert len(tool_manager._tools) == 53

    def test_all_compliance_tools_have_descriptions(self):
        """Every compliance tool must have a substantive description."""
        tool_manager = mcp_server._tool_manager
        for name in self.COMPLIANCE_TOOLS:
            tool = tool_manager._tools[name]
            assert tool.description, f"Tool '{name}' has no description"
            assert len(tool.description) > 20, (
                f"Tool '{name}' description too short"
            )


# ═══════════════════════════════════════════════════════════════════
# Tool: create_compliance_opinion
# ═══════════════════════════════════════════════════════════════════


class TestCreateComplianceOpinion:
    """Tests for the create_compliance_opinion tool."""

    def test_basic_creation(self):
        fn = _get_tool("create_compliance_opinion")
        result = fn(lawfulness=0.8, violation=0.05, uncertainty=0.15, base_rate=0.3)
        _assert_valid_compliance(result)
        assert abs(result["lawfulness"] - 0.8) < TOL
        assert abs(result["violation"] - 0.05) < TOL
        assert abs(result["uncertainty"] - 0.15) < TOL
        assert abs(result["base_rate"] - 0.3) < TOL

    def test_projected_probability(self):
        fn = _get_tool("create_compliance_opinion")
        result = fn(lawfulness=0.8, violation=0.05, uncertainty=0.15, base_rate=0.5)
        expected_pp = 0.8 + 0.5 * 0.15  # l + a*u
        assert abs(result["projected_probability"] - expected_pp) < TOL

    def test_default_base_rate(self):
        fn = _get_tool("create_compliance_opinion")
        result = fn(lawfulness=0.7, violation=0.1, uncertainty=0.2)
        assert abs(result["base_rate"] - 0.5) < TOL

    def test_includes_belief_disbelief_aliases(self):
        """Result should include both compliance AND SL terminology."""
        fn = _get_tool("create_compliance_opinion")
        result = fn(lawfulness=0.8, violation=0.05, uncertainty=0.15)
        # Compliance aliases
        assert "lawfulness" in result
        assert "violation" in result
        # Standard SL fields too
        assert "belief" in result
        assert "disbelief" in result
        assert abs(result["belief"] - result["lawfulness"]) < TOL
        assert abs(result["disbelief"] - result["violation"]) < TOL

    def test_invalid_constraint(self):
        fn = _get_tool("create_compliance_opinion")
        with pytest.raises(ValueError):
            fn(lawfulness=0.8, violation=0.5, uncertainty=0.15)  # sum > 1


# ═══════════════════════════════════════════════════════════════════
# Tool: jurisdictional_meet
# ═══════════════════════════════════════════════════════════════════


class TestJurisdictionalMeetTool:
    """Tests for the jurisdictional_meet tool."""

    def test_two_jurisdictions(self):
        fn = _get_tool("jurisdictional_meet")
        opinions = [
            {"belief": 0.8, "disbelief": 0.05, "uncertainty": 0.15, "base_rate": 0.5},
            {"belief": 0.7, "disbelief": 0.10, "uncertainty": 0.20, "base_rate": 0.5},
        ]
        result = fn(opinions_json=json.dumps(opinions))
        _assert_valid_compliance(result)
        # l⊓ = l₁·l₂ = 0.56
        assert abs(result["lawfulness"] - 0.56) < TOL
        # v⊓ = v₁+v₂-v₁·v₂ = 0.145
        assert abs(result["violation"] - 0.145) < TOL

    def test_single_opinion_passthrough(self):
        fn = _get_tool("jurisdictional_meet")
        opinions = [
            {"belief": 0.8, "disbelief": 0.05, "uncertainty": 0.15, "base_rate": 0.3},
        ]
        result = fn(opinions_json=json.dumps(opinions))
        _assert_valid_compliance(result)
        assert abs(result["lawfulness"] - 0.8) < TOL

    def test_three_way(self):
        fn = _get_tool("jurisdictional_meet")
        opinions = [
            {"belief": 0.8, "disbelief": 0.05, "uncertainty": 0.15, "base_rate": 0.5},
            {"belief": 0.7, "disbelief": 0.10, "uncertainty": 0.20, "base_rate": 0.5},
            {"belief": 0.9, "disbelief": 0.02, "uncertainty": 0.08, "base_rate": 0.5},
        ]
        result = fn(opinions_json=json.dumps(opinions))
        _assert_valid_compliance(result)
        # Monotonic restriction: l⊓ ≤ min(all l_i)
        assert result["lawfulness"] <= 0.7 + TOL

    def test_annihilator(self):
        fn = _get_tool("jurisdictional_meet")
        opinions = [
            {"belief": 0.8, "disbelief": 0.05, "uncertainty": 0.15, "base_rate": 0.5},
            {"belief": 0.0, "disbelief": 1.0, "uncertainty": 0.0, "base_rate": 0.5},
        ]
        result = fn(opinions_json=json.dumps(opinions))
        assert abs(result["lawfulness"]) < TOL
        assert abs(result["violation"] - 1.0) < TOL

    def test_empty_raises(self):
        fn = _get_tool("jurisdictional_meet")
        with pytest.raises(ValueError):
            fn(opinions_json="[]")

    def test_invalid_json_raises(self):
        fn = _get_tool("jurisdictional_meet")
        with pytest.raises(ValueError):
            fn(opinions_json="not json")


# ═══════════════════════════════════════════════════════════════════
# Tool: compliance_propagation
# ═══════════════════════════════════════════════════════════════════


class TestCompliancePropagationTool:
    """Tests for the compliance_propagation tool."""

    def test_basic_propagation(self):
        fn = _get_tool("compliance_propagation")
        source = {"belief": 0.9, "disbelief": 0.03, "uncertainty": 0.07, "base_rate": 0.5}
        trust = {"belief": 0.95, "disbelief": 0.01, "uncertainty": 0.04, "base_rate": 0.5}
        purpose = {"belief": 0.90, "disbelief": 0.02, "uncertainty": 0.08, "base_rate": 0.5}
        result = fn(
            source_json=json.dumps(source),
            derivation_trust_json=json.dumps(trust),
            purpose_compat_json=json.dumps(purpose),
        )
        _assert_valid_compliance(result)
        # l_D = t·p·l_S = 0.95 * 0.90 * 0.9 = 0.7695
        assert abs(result["lawfulness"] - 0.95 * 0.90 * 0.9) < TOL

    def test_degradation_monotonicity(self):
        fn = _get_tool("compliance_propagation")
        source = {"belief": 0.9, "disbelief": 0.03, "uncertainty": 0.07, "base_rate": 0.5}
        trust = {"belief": 0.8, "disbelief": 0.05, "uncertainty": 0.15, "base_rate": 0.5}
        purpose = {"belief": 0.7, "disbelief": 0.10, "uncertainty": 0.20, "base_rate": 0.5}
        result = fn(
            source_json=json.dumps(source),
            derivation_trust_json=json.dumps(trust),
            purpose_compat_json=json.dumps(purpose),
        )
        assert result["lawfulness"] <= source["belief"] + TOL


# ═══════════════════════════════════════════════════════════════════
# Tool: consent_validity
# ═══════════════════════════════════════════════════════════════════


class TestConsentValidityTool:
    """Tests for the consent_validity tool."""

    def test_six_conditions_all_strong(self):
        fn = _get_tool("consent_validity")
        conditions = [
            {"belief": 0.9, "disbelief": 0.02, "uncertainty": 0.08, "base_rate": 0.5}
        ] * 6
        result = fn(conditions_json=json.dumps(conditions))
        _assert_valid_compliance(result)
        # l = 0.9^6 ≈ 0.531441
        assert abs(result["lawfulness"] - 0.9 ** 6) < TOL

    def test_one_weak_condition_dominates(self):
        fn = _get_tool("consent_validity")
        strong = {"belief": 0.9, "disbelief": 0.02, "uncertainty": 0.08, "base_rate": 0.5}
        weak = {"belief": 0.2, "disbelief": 0.5, "uncertainty": 0.3, "base_rate": 0.5}
        conditions = [strong, strong, strong, strong, strong, weak]
        result = fn(conditions_json=json.dumps(conditions))
        _assert_valid_compliance(result)
        # l = 0.9^5 * 0.2 ≈ 0.118
        assert result["lawfulness"] < 0.15

    def test_wrong_count_raises(self):
        fn = _get_tool("consent_validity")
        conditions = [
            {"belief": 0.9, "disbelief": 0.02, "uncertainty": 0.08, "base_rate": 0.5}
        ] * 4
        with pytest.raises(ValueError):
            fn(conditions_json=json.dumps(conditions))

    def test_invalid_json_raises(self):
        fn = _get_tool("consent_validity")
        with pytest.raises(ValueError):
            fn(conditions_json="not json")


# ═══════════════════════════════════════════════════════════════════
# Tool: withdrawal_override
# ═══════════════════════════════════════════════════════════════════


class TestWithdrawalOverrideTool:
    """Tests for the withdrawal_override tool."""

    def test_pre_withdrawal(self):
        fn = _get_tool("withdrawal_override")
        consent = {"belief": 0.85, "disbelief": 0.05, "uncertainty": 0.10, "base_rate": 0.5}
        withdrawal = {"belief": 0.1, "disbelief": 0.7, "uncertainty": 0.2, "base_rate": 0.5}
        result = fn(
            consent_json=json.dumps(consent),
            withdrawal_json=json.dumps(withdrawal),
            assessment_time=50.0,
            withdrawal_time=100.0,
        )
        _assert_valid_compliance(result)
        assert abs(result["lawfulness"] - 0.85) < TOL

    def test_post_withdrawal(self):
        fn = _get_tool("withdrawal_override")
        consent = {"belief": 0.85, "disbelief": 0.05, "uncertainty": 0.10, "base_rate": 0.5}
        withdrawal = {"belief": 0.1, "disbelief": 0.7, "uncertainty": 0.2, "base_rate": 0.5}
        result = fn(
            consent_json=json.dumps(consent),
            withdrawal_json=json.dumps(withdrawal),
            assessment_time=150.0,
            withdrawal_time=100.0,
        )
        _assert_valid_compliance(result)
        assert abs(result["lawfulness"] - 0.1) < TOL

    def test_at_boundary(self):
        fn = _get_tool("withdrawal_override")
        consent = {"belief": 0.85, "disbelief": 0.05, "uncertainty": 0.10, "base_rate": 0.5}
        withdrawal = {"belief": 0.1, "disbelief": 0.7, "uncertainty": 0.2, "base_rate": 0.5}
        result = fn(
            consent_json=json.dumps(consent),
            withdrawal_json=json.dumps(withdrawal),
            assessment_time=100.0,
            withdrawal_time=100.0,
        )
        # t >= t_w → withdrawal
        assert abs(result["lawfulness"] - 0.1) < TOL


# ═══════════════════════════════════════════════════════════════════
# Tool: expiry_trigger
# ═══════════════════════════════════════════════════════════════════


class TestExpiryTriggerTool:
    """Tests for the expiry_trigger tool."""

    def test_pre_trigger_unchanged(self):
        fn = _get_tool("expiry_trigger")
        opinion = {"belief": 0.8, "disbelief": 0.05, "uncertainty": 0.15, "base_rate": 0.5}
        result = fn(
            opinion_json=json.dumps(opinion),
            assessment_time=50.0,
            trigger_time=100.0,
        )
        _assert_valid_compliance(result)
        assert abs(result["lawfulness"] - 0.8) < TOL

    def test_hard_expiry(self):
        fn = _get_tool("expiry_trigger")
        opinion = {"belief": 0.8, "disbelief": 0.05, "uncertainty": 0.15, "base_rate": 0.5}
        result = fn(
            opinion_json=json.dumps(opinion),
            assessment_time=150.0,
            trigger_time=100.0,
            residual_factor=0.0,
        )
        _assert_valid_compliance(result)
        assert abs(result["lawfulness"]) < TOL
        assert abs(result["violation"] - 0.85) < TOL  # v + l
        assert abs(result["uncertainty"] - 0.15) < TOL  # unchanged

    def test_partial_expiry(self):
        fn = _get_tool("expiry_trigger")
        opinion = {"belief": 0.8, "disbelief": 0.05, "uncertainty": 0.15, "base_rate": 0.5}
        result = fn(
            opinion_json=json.dumps(opinion),
            assessment_time=150.0,
            trigger_time=100.0,
            residual_factor=0.5,
        )
        _assert_valid_compliance(result)
        assert abs(result["lawfulness"] - 0.4) < TOL  # γ·l
        assert abs(result["violation"] - 0.45) < TOL  # v + (1-γ)·l


# ═══════════════════════════════════════════════════════════════════
# Tool: review_due_trigger
# ═══════════════════════════════════════════════════════════════════


class TestReviewDueTriggerTool:
    """Tests for the review_due_trigger tool."""

    def test_pre_trigger_unchanged(self):
        fn = _get_tool("review_due_trigger")
        opinion = {"belief": 0.8, "disbelief": 0.05, "uncertainty": 0.15, "base_rate": 0.5}
        result = fn(
            opinion_json=json.dumps(opinion),
            assessment_time=50.0,
            trigger_time=100.0,
            accelerated_half_life=30.0,
        )
        _assert_valid_compliance(result)
        assert abs(result["lawfulness"] - 0.8) < TOL

    def test_post_trigger_decays(self):
        fn = _get_tool("review_due_trigger")
        opinion = {"belief": 0.8, "disbelief": 0.05, "uncertainty": 0.15, "base_rate": 0.5}
        result = fn(
            opinion_json=json.dumps(opinion),
            assessment_time=200.0,
            trigger_time=100.0,
            accelerated_half_life=30.0,
        )
        _assert_valid_compliance(result)
        # Should decay significantly — 100 elapsed / 30 half_life ≈ 3.3 half-lives
        assert result["lawfulness"] < 0.15
        assert result["uncertainty"] > 0.5


# ═══════════════════════════════════════════════════════════════════
# Tool: regulatory_change_trigger
# ═══════════════════════════════════════════════════════════════════


class TestRegulatoryChangeTriggerTool:
    """Tests for the regulatory_change_trigger tool."""

    def test_pre_trigger(self):
        fn = _get_tool("regulatory_change_trigger")
        opinion = {"belief": 0.8, "disbelief": 0.05, "uncertainty": 0.15, "base_rate": 0.5}
        new_op = {"belief": 0.3, "disbelief": 0.2, "uncertainty": 0.5, "base_rate": 0.5}
        result = fn(
            opinion_json=json.dumps(opinion),
            assessment_time=50.0,
            trigger_time=100.0,
            new_opinion_json=json.dumps(new_op),
        )
        _assert_valid_compliance(result)
        assert abs(result["lawfulness"] - 0.8) < TOL

    def test_post_trigger_replaces(self):
        fn = _get_tool("regulatory_change_trigger")
        opinion = {"belief": 0.8, "disbelief": 0.05, "uncertainty": 0.15, "base_rate": 0.5}
        new_op = {"belief": 0.3, "disbelief": 0.2, "uncertainty": 0.5, "base_rate": 0.5}
        result = fn(
            opinion_json=json.dumps(opinion),
            assessment_time=150.0,
            trigger_time=100.0,
            new_opinion_json=json.dumps(new_op),
        )
        _assert_valid_compliance(result)
        assert abs(result["lawfulness"] - 0.3) < TOL
        assert abs(result["violation"] - 0.2) < TOL


# ═══════════════════════════════════════════════════════════════════
# Tool: erasure_scope_opinion
# ═══════════════════════════════════════════════════════════════════


class TestErasureScopeOpinionTool:
    """Tests for the erasure_scope_opinion tool."""

    def test_two_nodes(self):
        fn = _get_tool("erasure_scope_opinion")
        opinions = [
            {"belief": 0.95, "disbelief": 0.02, "uncertainty": 0.03, "base_rate": 0.5},
            {"belief": 0.90, "disbelief": 0.05, "uncertainty": 0.05, "base_rate": 0.5},
        ]
        result = fn(opinions_json=json.dumps(opinions))
        _assert_valid_compliance(result)
        # e_R = 0.95 * 0.90 = 0.855
        assert abs(result["lawfulness"] - 0.855) < TOL

    def test_exponential_degradation(self):
        fn = _get_tool("erasure_scope_opinion")
        opinions = [
            {"belief": 0.99, "disbelief": 0.005, "uncertainty": 0.005, "base_rate": 0.5}
        ] * 10
        result = fn(opinions_json=json.dumps(opinions))
        _assert_valid_compliance(result)
        # e_R = 0.99^10 ≈ 0.9044
        assert abs(result["lawfulness"] - 0.99 ** 10) < 1e-4

    def test_empty_raises(self):
        fn = _get_tool("erasure_scope_opinion")
        with pytest.raises(ValueError):
            fn(opinions_json="[]")


# ═══════════════════════════════════════════════════════════════════
# Tool: residual_contamination
# ═══════════════════════════════════════════════════════════════════


class TestResidualContaminationTool:
    """Tests for the residual_contamination tool."""

    def test_basic_contamination(self):
        fn = _get_tool("residual_contamination")
        opinions = [
            {"belief": 0.9, "disbelief": 0.05, "uncertainty": 0.05, "base_rate": 0.5},
            {"belief": 0.8, "disbelief": 0.10, "uncertainty": 0.10, "base_rate": 0.5},
        ]
        result = fn(opinions_json=json.dumps(opinions))
        _assert_valid_compliance(result)
        # r = 1 - (1-0.05)(1-0.10) = 1 - 0.855 = 0.145
        assert abs(result["violation"] - 0.145) < TOL
        # r̄ = 0.9 * 0.8 = 0.72
        assert abs(result["lawfulness"] - 0.72) < TOL

    def test_constraint(self):
        fn = _get_tool("residual_contamination")
        opinions = [
            {"belief": 0.7, "disbelief": 0.15, "uncertainty": 0.15, "base_rate": 0.5},
            {"belief": 0.6, "disbelief": 0.20, "uncertainty": 0.20, "base_rate": 0.5},
            {"belief": 0.8, "disbelief": 0.10, "uncertainty": 0.10, "base_rate": 0.5},
        ]
        result = fn(opinions_json=json.dumps(opinions))
        _assert_valid_compliance(result)

    def test_empty_raises(self):
        fn = _get_tool("residual_contamination")
        with pytest.raises(ValueError):
            fn(opinions_json="[]")


# ═══════════════════════════════════════════════════════════════════
# Resources
# ═══════════════════════════════════════════════════════════════════


class TestComplianceResources:
    """Tests for compliance algebra resources."""

    def test_compliance_opinion_schema(self):
        fn = _get_resource("jsonld-ex://schema/compliance-opinion")
        raw = fn()
        schema = json.loads(raw)
        assert schema["title"] == "ComplianceOpinion"
        assert "lawfulness" in schema["properties"]
        assert "violation" in schema["properties"]
        assert "uncertainty" in schema["properties"]
        assert "base_rate" in schema["properties"]
        # Must include constraint documentation
        assert "description" in schema

    def test_compliance_context(self):
        fn = _get_resource("jsonld-ex://context/compliance")
        raw = fn()
        ctx = json.loads(raw)
        assert "@context" in ctx
        context = ctx["@context"]
        assert "@lawfulness" in context or "lawfulness" in context


# ═══════════════════════════════════════════════════════════════════
# Prompts
# ═══════════════════════════════════════════════════════════════════


class TestCompliancePrompts:
    """Tests for compliance algebra prompts."""

    def test_gdpr_compliance_assessment_prompt(self):
        fn = _get_prompt("gdpr_compliance_assessment")
        result = fn(jurisdiction_list="EU-GDPR, US-CCPA, UK-DPA")
        assert isinstance(result, str)
        assert "jurisdictional_meet" in result
        assert "EU-GDPR" in result
        assert len(result) > 100

    def test_consent_lifecycle_prompt(self):
        fn = _get_prompt("consent_lifecycle")
        result = fn(purpose="marketing analytics")
        assert isinstance(result, str)
        assert "consent_validity" in result
        assert "withdrawal_override" in result
        assert "marketing analytics" in result
        assert len(result) > 100
