"""Tests for Boole-Frechet LP bounds."""

import pytest

from ml.bounds import compute_bounds
from ml.formula import (
    OperatorNode,
    VariableNode,
    eval_formula_on_assignment,
)


def _var(name: str) -> VariableNode:
    return VariableNode(var=name)


def _and(*args) -> OperatorNode:
    return OperatorNode.model_construct(node_type="and", arguments=list(args))


def _or(*args) -> OperatorNode:
    return OperatorNode.model_construct(node_type="or", arguments=list(args))


def _not(child) -> OperatorNode:
    return OperatorNode.model_construct(node_type="not", arguments=[child])


class TestEvalFormula:

    def test_variable_true(self):
        assert eval_formula_on_assignment(_var("X"), {"X": True}) is True

    def test_variable_false(self):
        assert eval_formula_on_assignment(_var("X"), {"X": False}) is False

    def test_not(self):
        assert eval_formula_on_assignment(_not(_var("X")), {"X": True}) is False

    def test_and(self):
        f = _and(_var("A"), _var("B"))
        assert eval_formula_on_assignment(f, {"A": True, "B": True}) is True
        assert eval_formula_on_assignment(f, {"A": True, "B": False}) is False

    def test_or(self):
        f = _or(_var("A"), _var("B"))
        assert eval_formula_on_assignment(f, {"A": False, "B": False}) is False
        assert eval_formula_on_assignment(f, {"A": True, "B": False}) is True


class TestComputeBounds:
    """compute_bounds produces correct Frechet bounds."""

    def test_and_bounds(self):
        # A AND B: lower = max(0, P(A)+P(B)-1), upper = min(P(A), P(B))
        f = _and(_var("A"), _var("B"))
        lo, hi = compute_bounds(f, {"A": 0.6, "B": 0.7})
        assert lo == pytest.approx(0.3, abs=1e-8)
        assert hi == pytest.approx(0.6, abs=1e-8)

    def test_or_bounds(self):
        # A OR B: lower = max(P(A), P(B)), upper = min(1, P(A)+P(B))
        f = _or(_var("A"), _var("B"))
        lo, hi = compute_bounds(f, {"A": 0.6, "B": 0.7})
        assert lo == pytest.approx(0.7, abs=1e-8)
        assert hi == pytest.approx(1.0, abs=1e-8)

    def test_not_exact(self):
        f = _not(_var("A"))
        lo, hi = compute_bounds(f, {"A": 0.6})
        assert lo == pytest.approx(0.4, abs=1e-8)
        assert hi == pytest.approx(0.4, abs=1e-8)

    def test_single_variable(self):
        f = _var("A")
        lo, hi = compute_bounds(f, {"A": 0.5})
        assert lo == pytest.approx(0.5, abs=1e-8)
        assert hi == pytest.approx(0.5, abs=1e-8)

    def test_tautology_and_contradiction(self):
        f = _or(_var("A"), _not(_var("A")))
        lo, hi = compute_bounds(f, {"A": 0.3})
        assert lo == pytest.approx(1.0, abs=1e-8)
        assert hi == pytest.approx(1.0, abs=1e-8)

    def test_three_variable_and(self):
        # A AND B AND C: lower = max(0, P(A)+P(B)+P(C)-2)
        f = _and(_var("A"), _var("B"), _var("C"))
        lo, hi = compute_bounds(f, {"A": 0.9, "B": 0.8, "C": 0.7})
        assert lo == pytest.approx(0.4, abs=1e-8)
        assert hi == pytest.approx(0.7, abs=1e-8)

    def test_bounds_are_within_01(self):
        f = _and(_var("A"), _var("B"))
        lo, hi = compute_bounds(f, {"A": 0.1, "B": 0.2})
        assert 0.0 <= lo <= hi <= 1.0
