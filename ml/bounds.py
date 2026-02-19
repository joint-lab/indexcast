"""
Boole-Frechet bounds via linear programming.

Authors:
- JGY <jyoung22@uvm.edu>
"""

import itertools

import numpy as np
from scipy.optimize import linprog

from ml.formula import (
    Formula,
    eval_formula_on_assignment,
    extract_literals_from_formula,
)


def compute_bounds(formula: Formula, marginals: dict[str, float]) -> tuple[float, float]:
    """
    Compute tight Boole-Frechet bounds on P(formula = true).

    Solves two linear programs (min and max) over the space of joint
    distributions consistent with the given marginal probabilities.

    Args:
        formula: A boolean rule as a Formula AST.
        marginals: Mapping from variable name to marginal probability.

    Returns:
        (lower_bound, upper_bound) tuple.

    """
    variables = list(dict.fromkeys(extract_literals_from_formula(formula)))
    n = len(variables)
    num_worlds = 1 << n

    assignments = list(itertools.product([False, True], repeat=n))

    # Objective: c[w] = 1 if the formula is true for world w
    c = np.zeros(num_worlds)
    for w_idx, bits in enumerate(assignments):
        asgn = dict(zip(variables, bits, strict=True))
        if eval_formula_on_assignment(formula, asgn):
            c[w_idx] = 1.0

    # Equality constraints: sum(p) = 1, and marginal for each variable
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

    res_min = linprog(c, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    lower = float(res_min.fun)

    res_max = linprog(-c, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    upper = float(-res_max.fun)

    return (lower, upper)
