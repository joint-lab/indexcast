"""Tests for Boole-Frechet LP bounds and CNF conversion."""

import itertools

import pytest

from ml.bounds import compute_bounds
from ml.formula import (
    OperatorNode,
    VariableNode,
    eval_formula_on_assignment,
    to_cnf,
    to_nnf,
)


def _var(name: str) -> VariableNode:
    return VariableNode(var=name)


def _and(*args) -> OperatorNode:
    return OperatorNode.model_construct(node_type="and", arguments=list(args))


def _or(*args) -> OperatorNode:
    return OperatorNode.model_construct(node_type="or", arguments=list(args))


def _not(child) -> OperatorNode:
    return OperatorNode.model_construct(node_type="not", arguments=[child])


def _truth_table(formula, variables: list[str]) -> set[tuple[bool, ...]]:
    """Return the set of truth assignments that make the formula true."""
    true_rows = set()
    for bits in itertools.product([False, True], repeat=len(variables)):
        assignment = dict(zip(variables, bits, strict=False))
        if eval_formula_on_assignment(formula, assignment):
            true_rows.add(bits)
    return true_rows


class TestCnfConversion:
    """CNF conversion preserves truth table equivalence."""

    def test_variable(self):
        f = _var("A")
        cnf = to_cnf(f)
        assert _truth_table(f, ["A"]) == _truth_table(cnf, ["A"])

    def test_not_variable(self):
        f = _not(_var("A"))
        cnf = to_cnf(f)
        assert _truth_table(f, ["A"]) == _truth_table(cnf, ["A"])

    def test_double_negation(self):
        f = _not(_not(_var("A")))
        cnf = to_cnf(f)
        assert _truth_table(f, ["A"]) == _truth_table(cnf, ["A"])

    def test_and(self):
        f = _and(_var("A"), _var("B"))
        cnf = to_cnf(f)
        assert _truth_table(f, ["A", "B"]) == _truth_table(cnf, ["A", "B"])

    def test_or(self):
        f = _or(_var("A"), _var("B"))
        cnf = to_cnf(f)
        assert _truth_table(f, ["A", "B"]) == _truth_table(cnf, ["A", "B"])

    def test_de_morgan_not_and(self):
        f = _not(_and(_var("A"), _var("B")))
        cnf = to_cnf(f)
        vars_ = ["A", "B"]
        assert _truth_table(f, vars_) == _truth_table(cnf, vars_)

    def test_de_morgan_not_or(self):
        f = _not(_or(_var("A"), _var("B")))
        cnf = to_cnf(f)
        vars_ = ["A", "B"]
        assert _truth_table(f, vars_) == _truth_table(cnf, vars_)

    def test_nested_formula(self):
        # (A AND B) OR (NOT C)
        f = _or(_and(_var("A"), _var("B")), _not(_var("C")))
        cnf = to_cnf(f)
        vars_ = ["A", "B", "C"]
        assert _truth_table(f, vars_) == _truth_table(cnf, vars_)

    def test_complex_three_variable(self):
        # NOT(A OR (B AND C))
        f = _not(_or(_var("A"), _and(_var("B"), _var("C"))))
        cnf = to_cnf(f)
        vars_ = ["A", "B", "C"]
        assert _truth_table(f, vars_) == _truth_table(cnf, vars_)


class TestNnf:
    """NNF pushes negations down to variables."""

    def _only_variable_negations(self, formula) -> bool:
        if isinstance(formula, VariableNode):
            return True
        if formula.node_type == "not":
            return isinstance(formula.arguments[0], VariableNode)
        return all(self._only_variable_negations(arg) for arg in formula.arguments)

    def test_nnf_de_morgan(self):
        f = _not(_and(_var("A"), _var("B")))
        nnf = to_nnf(f)
        assert self._only_variable_negations(nnf)

    def test_nnf_double_negation(self):
        f = _not(_not(_var("A")))
        nnf = to_nnf(f)
        assert isinstance(nnf, VariableNode)


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
