"""
Boolean formula AST, manipulation, and evaluation.

Authors:
- JGY <jyoung22@uvm.edu>
- Erik Arnold <ewarnold@uvm.edu>
"""

import itertools
from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

# =============================================================================
# Constants
# =============================================================================

MAX_RULE_DEPTH = 3
MAX_LITERALS_PER_RULE = 3
MIN_CHILDREN_PER_NODE = 2
MAX_CHILDREN_PER_NODE = 3
ALLOWED_OPS = ["and", "or", "not"]


# =============================================================================
# AST Nodes
# =============================================================================


class VariableNode(BaseModel):
    """Variable node."""

    node_type: Literal["variable"] = "variable"
    var: str = Field(description="Market ID")

    @model_validator(mode="after")
    def validate_node_type(self):
        """Make sure variable is valid."""
        if self.node_type != "variable":
            raise ValueError(f"Invalid node_type for Var: {self.node_type}")
        return self

    model_config = dict[str, bool | str](populate_by_name=True, extra="forbid")


class OperatorNode(BaseModel):
    """Logical operator node."""

    node_type: Literal["and", "or", "not"]
    arguments: list["Formula"] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_arity(self):
        """Strict arity validation."""
        if self.node_type == "not":
            if len(self.arguments) != 1:
                raise ValueError("NOT requires exactly 1 argument")
        else:
            if not (MIN_CHILDREN_PER_NODE <= len(self.arguments) <= MAX_CHILDREN_PER_NODE):
                raise ValueError(
                    f"{self.node_type.upper()} must have "
                    f"{MIN_CHILDREN_PER_NODE}-{MAX_CHILDREN_PER_NODE} children"
                )
        return self

    @model_validator(mode="after")
    def validate_node_type(self):
        """Make sure valid node type."""
        if self.node_type not in ["and", "or", "not"]:
            raise ValueError(f"Invalid node_type for Op: {self.node_type}")
        return self

    model_config = dict[str, str](extra="forbid")


Formula = Annotated[
    VariableNode | OperatorNode,
    Field(discriminator="node_type"),
]

# Rebuild Op model to resolve ForwardRef
OperatorNode.model_rebuild()


# =============================================================================
# Validation
# =============================================================================


def validate_rule_depth(rule: Formula, max_depth=MAX_RULE_DEPTH):
    """Ensure max depth is not exceeded."""

    def depth(node: Formula, d=0):
        if node.node_type == "variable":
            return d
        elif node.node_type == "not":
            return depth(node.arguments[0], d + 1)
        elif node.node_type in ["and", "or"]:
            return max(depth(child, d + 1) for child in node.arguments)
        else:
            raise ValueError(f"Unknown node_type: {node.node_type}")

    rule_depth = depth(rule)
    if rule_depth > max_depth:
        raise ValueError(f"Rule depth {rule_depth} exceeds maximum allowed {max_depth}")


def validate_literal_count(rule: Formula, max_literals=MAX_LITERALS_PER_RULE):
    """Validate the literal count."""
    count = len(extract_literals_from_formula(rule))
    if count > max_literals:
        raise ValueError(f"Rule uses {count} literals but max allowed is {max_literals}")


def validate_market_ids(rule: Formula, allowed_ids: set[str]):
    """Ensure all variables use allowed market IDs."""
    vars_ = extract_literals_from_formula(rule)
    for v in vars_:
        if v not in allowed_ids:
            raise ValueError(f"Invalid market ID in rule: {v}")


# =============================================================================
# Formula Utilities
# =============================================================================


def stringify_formula(f: Formula) -> str:
    """Human-readable display of a formula."""
    if f.node_type == "variable":
        return f"[VAR: {f.var}]"

    if f.node_type == "not":
        return f"NOT ({stringify_formula(f.arguments[0])})"

    # AND / OR
    children = [stringify_formula(a) for a in f.arguments]
    op = f.node_type.upper()
    return f" {op} ".join(f"({c})" for c in children)


def extract_literals_from_formula(formula: Formula) -> list[str]:
    """
    Extract all variable names from a formula, stripping surrounding quotes.

    Args:
        formula: A Formula instance containing the rule tree

    Returns:
        List of cleaned variable names found in the formula

    """
    literals = []

    def traverse(node: Formula):
        if node.node_type == "variable":
            # Strip leading/trailing quotes (single or double)
            cleaned_name = node.var.strip('"').strip("'")
            literals.append(cleaned_name)
        elif hasattr(node, "arguments"):
            for child in node.arguments:
                traverse(child)

    traverse(formula)
    return literals


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


# =============================================================================
# CNF Conversion
# =============================================================================


def _make_not(child: Formula) -> OperatorNode:
    """Create a NOT node, bypassing arity validation."""
    return OperatorNode.model_construct(node_type="not", arguments=[child])


def _make_op(op: str, args: list[Formula]) -> Formula:
    """Create an operator node, bypassing arity validation."""
    if len(args) == 1:
        return args[0]
    return OperatorNode.model_construct(node_type=op, arguments=args)


def to_nnf(formula: Formula) -> Formula:
    """Convert a formula to Negation Normal Form."""
    if isinstance(formula, VariableNode):
        return formula

    op = formula.node_type

    if op == "not":
        child = formula.arguments[0]

        if isinstance(child, VariableNode):
            return formula

        child_op = child.node_type

        if child_op == "not":
            return to_nnf(child.arguments[0])
        if child_op == "and":
            new_args = [to_nnf(_make_not(arg)) for arg in child.arguments]
            return _make_op("or", new_args)
        if child_op == "or":
            new_args = [to_nnf(_make_not(arg)) for arg in child.arguments]
            return _make_op("and", new_args)

    if op in ("and", "or"):
        new_args = [to_nnf(arg) for arg in formula.arguments]
        return _make_op(op, new_args)

    return formula


def _collect_children(node: Formula, op: str) -> list[Formula]:
    """Flatten nested nodes of the same operator type."""
    if isinstance(node, OperatorNode) and node.node_type == op:
        result: list[Formula] = []
        for child in node.arguments:
            result.extend(_collect_children(child, op))
        return result
    return [node]


def distribute_or_over_and(formula: Formula) -> Formula:
    """Convert an NNF formula to CNF by distributing OR over AND."""
    if isinstance(formula, VariableNode):
        return formula

    op = formula.node_type

    if op == "not":
        return formula

    if op == "and":
        converted = [distribute_or_over_and(arg) for arg in formula.arguments]
        flat = []
        for c in converted:
            flat.extend(_collect_children(c, "and"))
        return _make_op("and", flat)

    if op == "or":
        converted = [distribute_or_over_and(arg) for arg in formula.arguments]

        flat_or: list[Formula] = []
        for c in converted:
            flat_or.extend(_collect_children(c, "or"))

        conjunct_lists: list[list[Formula]] = []
        for term in flat_or:
            if isinstance(term, OperatorNode) and term.node_type == "and":
                conjunct_lists.append(list(term.arguments))
            else:
                conjunct_lists.append([term])

        clauses: list[Formula] = []
        for combo in itertools.product(*conjunct_lists):
            clause_lits: list[Formula] = []
            for lit in combo:
                clause_lits.extend(_collect_children(lit, "or"))
            clauses.append(_make_op("or", clause_lits))

        return _make_op("and", clauses)

    return formula


def to_cnf(formula: Formula) -> Formula:
    """Convert a formula to Conjunctive Normal Form."""
    return distribute_or_over_and(to_nnf(formula))
