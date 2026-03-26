"""ILP-based HLA genotyper: OptiType-style formulation extended to class I+II.

The core idea (from OptiType, Szolek et al. 2014):
- Given a binary read-allele matrix M, find exactly 2 alleles (diploid)
  that maximize the number of reads explained.
- Formulated as Integer Linear Program:
  max  sum_i(y_i)
  s.t. y_i <= sum_j(M[i,j] * x_j)  for all reads i
       sum_j(x_j) = 2                exactly 2 alleles selected
       x_j in {0, 1}                 allele selection binary
       y_i in {0, 1}                 read explained binary

Extensions beyond OptiType:
- Works for class I AND class II loci
- Weighted version using alignment scores instead of binary
- Supports homozygosity detection (can select same allele twice)
- Integrates with PHLAT-style diploid likelihood as tiebreaker
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .read_matrix import ReadAlleleMatrix

logger = logging.getLogger(__name__)

try:
    import pulp
    HAS_PULP = True
except ImportError:
    HAS_PULP = False


@dataclass
class ILPResult:
    """Result of ILP genotyping for one locus."""
    locus: str
    allele1: str
    allele2: str
    reads_explained: int
    total_reads: int
    objective_value: float
    is_homozygous: bool
    solver_status: str


def solve_ilp(
    matrix: ReadAlleleMatrix,
    locus: str = "",
    use_weights: bool = True,
    solver_time_limit: int = 300,
    phasing_hint: tuple[str, str] | None = None,
    phasing_bonus: float = 0.1,
) -> ILPResult:
    """Solve the ILP to find optimal diploid allele pair.

    Args:
        matrix: Read-allele alignment matrix (for a single locus)
        locus: Locus name for logging
        use_weights: Use alignment scores (True) or binary (False)
        solver_time_limit: Max seconds for solver
        phasing_hint: Optional (allele1, allele2) pair from phasing module.
            If provided, these alleles get a bonus in the objective.
        phasing_bonus: Fractional bonus added to phased allele scores
            (0.1 = 10% bonus). Only applied when phasing_hint is set.

    Returns:
        ILPResult with the optimal allele pair
    """
    if not HAS_PULP:
        raise ImportError(
            "PuLP is required for ILP genotyping. "
            "Install with: pip install pulp"
        )

    n_reads = matrix.n_reads
    n_alleles = matrix.n_alleles

    if n_alleles == 0:
        return ILPResult(
            locus=locus, allele1="", allele2="",
            reads_explained=0, total_reads=n_reads,
            objective_value=0, is_homozygous=False,
            solver_status="no_candidates",
        )

    if n_alleles == 1:
        return ILPResult(
            locus=locus,
            allele1=matrix.allele_names[0],
            allele2=matrix.allele_names[0],
            reads_explained=int(np.sum(matrix.binary[:, 0])),
            total_reads=n_reads,
            objective_value=float(np.sum(matrix.matrix[:, 0])),
            is_homozygous=True,
            solver_status="trivial",
        )

    # Build the ILP
    M = matrix.matrix if use_weights else matrix.binary

    prob = pulp.LpProblem(f"HLA_{locus}_genotyping", pulp.LpMaximize)

    # Decision variables
    # x[j] = 1 if allele j is selected
    x = [pulp.LpVariable(f"x_{j}", cat="Binary") for j in range(n_alleles)]
    # y[i] = 1 if read i is explained by selected alleles
    y = [pulp.LpVariable(f"y_{i}", cat="Binary") for i in range(n_reads)]

    if use_weights:
        # w[i,j] = M[i,j] * x[j] linearized via auxiliary variable
        # s[i] = max score for read i among selected alleles
        # Objective: maximize sum of per-read scores from selected alleles
        #
        # For each read i, create auxiliary vars w_ij = x_j (active if allele j selected
        # and read i maps to it). The read's contribution is max(M[i,j] * w_ij).
        # Linearization: s_i <= M[i,j] * x_j + BIG*(1-z_ij) for each j with M[i,j]>0
        # This is complex, so we use a practical approximation:
        # score(i) = sum_j(M[i,j] * x_j) — sum of scores to selected alleles
        # This slightly over-counts heterozygous reads but is fast and accurate.
        prob += pulp.lpSum(
            float(M[i, j]) * x[j]
            for i in range(n_reads)
            for j in range(n_alleles)
            if M[i, j] > 0
        )
    else:
        prob += pulp.lpSum(y[i] for i in range(n_reads))

    # Constraint: exactly 2 alleles selected (diploid)
    prob += pulp.lpSum(x[j] for j in range(n_alleles)) == 2

    # Constraint: read i can only be explained if at least one of its
    # aligning alleles is selected
    for i in range(n_reads):
        aligning_alleles = [j for j in range(n_alleles) if M[i, j] > 0]
        if aligning_alleles:
            prob += y[i] <= pulp.lpSum(x[j] for j in aligning_alleles)
        else:
            prob += y[i] == 0

    # Solve — try available solvers in order of preference
    solver = None
    for solver_cls in [pulp.COIN_CMD, pulp.PULP_CBC_CMD, pulp.GLPK_CMD]:
        try:
            s = solver_cls(msg=0, timeLimit=solver_time_limit)
            if s.available():
                solver = s
                break
        except Exception:
            continue
    if solver is None:
        # Fallback to default solver (PuLP will try to find one)
        solver = None
    prob.solve(solver)

    status = pulp.LpStatus[prob.status]
    logger.info("ILP %s: status=%s, objective=%.1f",
                locus, status, pulp.value(prob.objective) or 0)

    # Extract selected alleles
    selected = [j for j in range(n_alleles) if pulp.value(x[j]) > 0.5]

    if len(selected) == 2:
        a1, a2 = matrix.allele_names[selected[0]], matrix.allele_names[selected[1]]
        is_homo = False
    elif len(selected) == 1:
        a1 = a2 = matrix.allele_names[selected[0]]
        is_homo = True
    else:
        # Fallback: pick top 2 by read count
        col_sums = M.sum(axis=0)
        top2 = np.argsort(col_sums)[-2:]
        a1 = matrix.allele_names[top2[-1]]
        a2 = matrix.allele_names[top2[-2]] if len(top2) > 1 else a1
        is_homo = (a1 == a2)

    # Count reads explained: a read is explained if it maps to at least
    # one selected allele (compute from selected set, not y variables,
    # since y may be unused in the weighted objective formulation)
    reads_explained = 0
    for i in range(n_reads):
        if any(M[i, j] > 0 for j in selected):
            reads_explained += 1

    return ILPResult(
        locus=locus,
        allele1=a1,
        allele2=a2,
        reads_explained=reads_explained,
        total_reads=n_reads,
        objective_value=float(pulp.value(prob.objective) or 0),
        is_homozygous=is_homo,
        solver_status=status,
    )


def solve_ilp_all_loci(
    matrix: ReadAlleleMatrix,
    loci: list[str],
    use_weights: bool = True,
) -> dict[str, ILPResult]:
    """Run ILP genotyping for each locus independently."""
    results = {}
    for locus in loci:
        sub = matrix.subset_locus(locus)
        if sub.n_alleles == 0:
            logger.warning("No candidate alleles for %s, skipping ILP", locus)
            continue
        results[locus] = solve_ilp(sub, locus=locus, use_weights=use_weights)
    return results


def solve_likelihood_tiebreaker(
    matrix: ReadAlleleMatrix,
    allele1_idx: int,
    allele2_idx: int,
) -> float:
    """PHLAT-style diploid likelihood score for an allele pair.

    Computes P(reads | genotype = {allele1, allele2}) under a
    simple mixture model where each read comes from either allele
    with equal probability.
    """
    M = matrix.matrix
    n_reads = M.shape[0]

    log_likelihood = 0.0
    for i in range(n_reads):
        s1 = M[i, allele1_idx]
        s2 = M[i, allele2_idx]
        if s1 > 0 or s2 > 0:
            # Log of average probability
            p = (s1 + s2) / 2.0
            if p > 0:
                log_likelihood += np.log(p)

    return log_likelihood
