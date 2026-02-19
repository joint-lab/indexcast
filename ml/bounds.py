"""
Boole-Frechet bounds via LP for arbitrary boolean rules.

Computes dependency-free bounds on the probability that a boolean rule
evaluates to true, given only the marginal probabilities of each variable.

Pipeline:
    1. Convert Formula AST to Conjunctive Normal Form (CNF)
    2. Solve a linear program to find min/max P(rule = true)

Authors:
- JGY <jyoung22@uvm.edu>
"""

import itertools

import numpy as np
from scipy.optimize import linprog

from ml.rules import (
    Formula,
    OperatorNode,
    VariableNode,
    extract_literals_from_formula,
)

# ---------------------------------------------------------------------------
# Part 1: CNF Conversion
# ---------------------------------------------------------------------------


def _make_not(child: Formula) -> OperatorNode:
    """Create a NOT node, bypassing arity validation."""
    return OperatorNode.model_construct(node_type="not", arguments=[child])


def _make_op(op: str, args: list[Formula]) -> Formula:
    """
    Create an AND/OR node, bypassing arity validation for CNF intermediates.

    Returns the single child directly if *args* has length 1.
    """
    if len(args) == 1:
        return args[0]
    return OperatorNode.model_construct(node_type=op, arguments=args)


def to_nnf(formula: Formula) -> Formula:
    """
    Convert a formula to Negation Normal Form.

    Push NOT inward using De Morgan's laws so that negations appear only
    directly on variables. Double negations are eliminated.
    """
    if isinstance(formula, VariableNode):
        return formula

    op = formula.node_type

    if op == "not":
        child = formula.arguments[0]

        if isinstance(child, VariableNode):
            return formula

        child_op = child.node_type

        if child_op == "not":
            # Double negation: NOT(NOT(A)) -> A
            return to_nnf(child.arguments[0])
        if child_op == "and":
            # De Morgan: NOT(A AND B) -> NOT(A) OR NOT(B)
            new_args = [to_nnf(_make_not(arg)) for arg in child.arguments]
            return _make_op("or", new_args)
        if child_op == "or":
            # De Morgan: NOT(A OR B) -> NOT(A) AND NOT(B)
            new_args = [to_nnf(_make_not(arg)) for arg in child.arguments]
            return _make_op("and", new_args)

    if op in ("and", "or"):
        new_args = [to_nnf(arg) for arg in formula.arguments]
        return _make_op(op, new_args)

    return formula


def _collect_children(node: Formula, op: str) -> list[Formula]:
    """Recursively flatten nested nodes of the same operator type."""
    if isinstance(node, OperatorNode) and node.node_type == op:
        result: list[Formula] = []
        for child in node.arguments:
            result.extend(_collect_children(child, op))
        return result
    return [node]


def distribute_or_over_and(formula: Formula) -> Formula:
    """
    Convert an NNF formula to CNF by distributing OR over AND.

    Recursively applies the rule:
        A OR (B AND C) -> (A OR B) AND (A OR C)
    """
    if isinstance(formula, VariableNode):
        return formula

    op = formula.node_type

    if op == "not":
        # In NNF, NOT only appears on variables — pass through.
        return formula

    if op == "and":
        converted = [distribute_or_over_and(arg) for arg in formula.arguments]
        flat = []
        for c in converted:
            flat.extend(_collect_children(c, "and"))
        return _make_op("and", flat)

    if op == "or":
        converted = [distribute_or_over_and(arg) for arg in formula.arguments]

        # Flatten nested ORs, then gather conjunct lists for distribution.
        flat_or: list[Formula] = []
        for c in converted:
            flat_or.extend(_collect_children(c, "or"))

        # Each element is either an AND (list of conjuncts) or a single term.
        conjunct_lists: list[list[Formula]] = []
        for term in flat_or:
            if isinstance(term, OperatorNode) and term.node_type == "and":
                conjunct_lists.append(list(term.arguments))
            else:
                conjunct_lists.append([term])

        # Cartesian product: each combo yields one OR-clause.
        clauses: list[Formula] = []
        for combo in itertools.product(*conjunct_lists):
            # Flatten any residual nested ORs inside the combo.
            clause_lits: list[Formula] = []
            for lit in combo:
                clause_lits.extend(_collect_children(lit, "or"))
            clauses.append(_make_op("or", clause_lits))

        return _make_op("and", clauses)

    return formula


def to_cnf(formula: Formula) -> Formula:
    """
    Convert a formula to Conjunctive Normal Form.

    Applies NNF conversion followed by OR-over-AND distribution.
    """
    return distribute_or_over_and(to_nnf(formula))


# ---------------------------------------------------------------------------
# Part 2: LP Bounds Solver
# ---------------------------------------------------------------------------


def eval_formula_on_assignment(formula: Formula, assignment: dict[str, bool]) -> bool:
    """
    Evaluate a boolean formula against a truth assignment.

    Args:
        formula: A Formula AST node.
        assignment: Mapping from variable name to boolean value.

    Returns:
        The truth value of the formula under the given assignment.

    """
    if isinstance(formula, VariableNode):
        return assignment[formula.var]

    op = formula.node_type

    if op == "not":
        return not eval_formula_on_assignment(formula.arguments[0], assignment)
    if op == "and":
        return all(eval_formula_on_assignment(arg, assignment) for arg in formula.arguments)
    if op == "or":
        return any(eval_formula_on_assignment(arg, assignment) for arg in formula.arguments)

    msg = f"Unknown node_type: {op}"
    raise ValueError(msg)


def compute_bounds(formula: Formula, marginals: dict[str, float]) -> tuple[float, float]:
    """
    Compute tight Boole-Frechet bounds on P(formula = true).

    Solves two linear programs (min and max) over the space of joint
    distributions consistent with the given marginal probabilities.

    Args:
        formula: A boolean rule as a Formula AST.
        marginals: Mapping from variable name to marginal probability.

    Returns:
        (lower_bound, upper_bound) — the tightest possible bounds on the
        probability of the formula being true.

    """
    # Deduplicated, order-preserving variable list.
    variables = list(dict.fromkeys(extract_literals_from_formula(formula)))
    n = len(variables)
    num_worlds = 1 << n  # 2^n

    # Enumerate all truth assignments.
    assignments = list(itertools.product([False, True], repeat=n))

    # Objective: c[w] = 1 if the formula is true for world w, else 0.
    c = np.zeros(num_worlds)
    for w_idx, bits in enumerate(assignments):
        asgn = dict(zip(variables, bits, strict=True))
        if eval_formula_on_assignment(formula, asgn):
            c[w_idx] = 1.0

    # Equality constraints: A_eq @ p = b_eq
    #   Row 0       : sum(p) = 1           (valid distribution)
    #   Row i+1     : sum(p where Xi=1) = marginal_i
    a_eq = np.zeros((1 + n, num_worlds))
    b_eq = np.zeros(1 + n)

    a_eq[0, :] = 1.0
    b_eq[0] = 1.0

    for i, var in enumerate(variables):
        for w_idx, bits in enumerate(assignments):
            if bits[i]:
                a_eq[i + 1, w_idx] = 1.0
        b_eq[i + 1] = marginals[var]

    bounds = [(0, None)] * num_worlds

    # Lower bound: minimise c·p
    res_min = linprog(c, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    lower = float(res_min.fun)

    # Upper bound: maximise c·p  ⇔  minimise (−c)·p
    res_max = linprog(-c, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    upper = float(-res_max.fun)

    return (lower, upper)
