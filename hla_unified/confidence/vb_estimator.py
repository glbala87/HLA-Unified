"""Variational Bayes confidence estimation (HLA-VBSeq-style).

After the ILP selects the optimal allele pair, this module provides
calibrated posterior probabilities using variational Bayes EM.

Model (Dirichlet-Multinomial mixture):
- Allele mixing proportions pi ~ Dirichlet(alpha)
- Each read r assigned to allele z_r ~ Categorical(pi)
- P(read r | z_r = k) proportional to alignment score M[r, k]
- Variational family: q(pi) = Dir(alpha_q), q(z_r) = Cat(resp_r)

ELBO = E_q[log p(X|Z)] + E_q[log p(Z|pi)] + E_q[log p(pi)]
       - E_q[log q(Z)] - E_q[log q(pi)]

This gives confidence scores more calibrated than simple log-likelihood
ratios (HLA-LA's Q1) because it models uncertainty in read-to-allele
assignment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.special import digamma, gammaln

from ..genotyper.read_matrix import ReadAlleleMatrix

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceResult:
    """Calibrated confidence for a diploid genotype call."""
    locus: str
    allele1: str
    allele2: str
    posterior_prob: float  # P(this pair is correct | data)
    allele1_dosage: float  # estimated fraction of reads from allele1
    allele2_dosage: float
    confidence_class: str  # "HIGH", "MEDIUM", "LOW"
    convergence_iterations: int
    elbo: float  # final ELBO value
    alternative_pairs: list[tuple[str, str, float]]  # (a1, a2, posterior)


class VariationalBayesEstimator:
    """VBSeq-style variational Bayes EM for HLA genotype confidence.

    Given the read-allele matrix and the ILP's top candidates,
    estimates posterior probabilities over possible diploid genotypes.
    """

    def __init__(
        self,
        max_iter: int = 200,
        tol: float = 1e-6,
        prior_alpha: float = 1.0,
        n_top_pairs: int = 20,
        allele_frequencies: dict[str, float] | None = None,
    ) -> None:
        self.max_iter = max_iter
        self.tol = tol
        self.prior_alpha = prior_alpha
        self.n_top_pairs = n_top_pairs
        # Population frequency priors: allele_name -> frequency (0-1)
        # When provided, the Dirichlet prior alpha is set proportional to
        # population frequency rather than uniform. This improves calls for
        # rare vs common alleles and reduces false positives on rare types.
        self.allele_frequencies = allele_frequencies

    def estimate(
        self,
        matrix: ReadAlleleMatrix,
        locus: str,
        candidate_alleles: list[str] | None = None,
    ) -> ConfidenceResult:
        """Estimate posterior confidence for a locus."""
        sub = matrix.subset_locus(locus) if candidate_alleles is None else matrix
        if sub.n_alleles < 2:
            return self._trivial_result(locus, sub)

        # Step 1: Run VB-EM to get allele mixture weights
        weights, responsibilities, n_iter, final_elbo = self._vb_em(sub)

        # Step 2: Enumerate top allele pairs by mixture weight
        pairs = self._enumerate_pairs(sub, weights, responsibilities)

        # Step 3: Compute pair-level posteriors via marginal likelihood
        pair_posteriors = self._compute_pair_posteriors(sub, pairs)

        if not pair_posteriors:
            return self._trivial_result(locus, sub)

        # Best pair
        best = pair_posteriors[0]
        a1_idx, a2_idx, posterior = best

        a1 = sub.allele_names[a1_idx]
        a2 = sub.allele_names[a2_idx]

        # Dosage from responsibilities
        dosage1 = float(responsibilities[:, a1_idx].mean()) if a1_idx < responsibilities.shape[1] else 0.5
        dosage2 = float(responsibilities[:, a2_idx].mean()) if a2_idx < responsibilities.shape[1] else 0.5
        total = dosage1 + dosage2
        if total > 0:
            dosage1 /= total
            dosage2 /= total

        # Confidence class
        if posterior > 0.99:
            conf_class = "HIGH"
        elif posterior > 0.90:
            conf_class = "MEDIUM"
        else:
            conf_class = "LOW"

        alternatives = [
            (sub.allele_names[i], sub.allele_names[j], p)
            for i, j, p in pair_posteriors[1:6]
        ]

        return ConfidenceResult(
            locus=locus,
            allele1=a1,
            allele2=a2,
            posterior_prob=posterior,
            allele1_dosage=dosage1,
            allele2_dosage=dosage2,
            confidence_class=conf_class,
            convergence_iterations=n_iter,
            elbo=final_elbo,
            alternative_pairs=alternatives,
        )

    def _vb_em(
        self, matrix: ReadAlleleMatrix
    ) -> tuple[np.ndarray, np.ndarray, int, float]:
        """Run Variational Bayes EM with correct ELBO.

        Returns (allele_weights, responsibilities, n_iterations, final_elbo).
        """
        M = matrix.matrix.copy()
        n_reads, n_alleles = M.shape

        # Convert alignment scores to log-likelihoods
        # Normalize per-row so max score -> log(1) = 0
        log_lik = np.full_like(M, -np.inf)
        for i in range(n_reads):
            row_max = M[i].max()
            if row_max > 0:
                nonzero = M[i] > 0
                log_lik[i, nonzero] = np.log(M[i, nonzero] / row_max)

        # Initialize Dirichlet prior — frequency-informed if available
        # When population allele frequencies are provided, the prior alpha
        # for each allele is proportional to its frequency. This gives
        # common alleles a slight head start while still allowing rare
        # alleles to be called when the data supports them.
        if self.allele_frequencies and matrix.allele_names:
            freq_prior = np.array([
                self.allele_frequencies.get(name, 1.0 / n_alleles)
                for name in matrix.allele_names
            ])
            freq_prior = freq_prior / freq_prior.sum()  # normalize
            alpha_0_values = self.prior_alpha + freq_prior * n_alleles
        else:
            alpha_0_values = np.full(n_alleles, self.prior_alpha)

        # Initialize Dirichlet posterior parameters
        alpha_q = alpha_0_values + n_reads / n_alleles
        alpha_0 = alpha_0_values.copy()

        # Initialize responsibilities via E-step
        resp = self._e_step(log_lik, alpha_q, n_reads, n_alleles)

        prev_elbo = -np.inf
        n_iter = 0
        final_elbo = -np.inf

        for iteration in range(self.max_iter):
            # M-step: update Dirichlet posterior parameters
            alpha_q = alpha_0 + resp.sum(axis=0)

            # E-step: update responsibilities
            resp = self._e_step(log_lik, alpha_q, n_reads, n_alleles)

            # Compute ELBO
            elbo = self._compute_elbo(log_lik, resp, alpha_q, alpha_0)

            if abs(elbo - prev_elbo) < self.tol:
                n_iter = iteration + 1
                final_elbo = elbo
                break
            prev_elbo = elbo
            final_elbo = elbo
            n_iter = iteration + 1

        # Final weights = expected pi under Dirichlet posterior
        weights = alpha_q / alpha_q.sum()

        logger.debug("VB-EM converged in %d iterations, ELBO=%.2f", n_iter, final_elbo)
        return weights, resp, n_iter, final_elbo

    def _e_step(
        self,
        log_lik: np.ndarray,
        alpha_q: np.ndarray,
        n_reads: int,
        n_alleles: int,
    ) -> np.ndarray:
        """E-step: update responsibilities q(z_r = k).

        log q(z_r = k) = E_q[log pi_k] + log p(x_r | z_r = k) + const
        """
        # E_q[log pi_k] = digamma(alpha_q_k) - digamma(sum(alpha_q))
        e_log_pi = digamma(alpha_q) - digamma(alpha_q.sum())

        resp = np.zeros((n_reads, n_alleles))
        for i in range(n_reads):
            log_resp = e_log_pi + log_lik[i]
            # Normalize in log-space for numerical stability
            finite_mask = np.isfinite(log_resp)
            if not finite_mask.any():
                continue
            max_lr = log_resp[finite_mask].max()
            log_resp_shifted = np.where(finite_mask, log_resp - max_lr, -np.inf)
            resp[i] = np.where(finite_mask, np.exp(log_resp_shifted), 0.0)
            total = resp[i].sum()
            if total > 0:
                resp[i] /= total

        return resp

    def _compute_elbo(
        self,
        log_lik: np.ndarray,
        resp: np.ndarray,
        alpha_q: np.ndarray,
        alpha_0: np.ndarray,
    ) -> float:
        """Compute the full Evidence Lower Bound (ELBO).

        ELBO = E_q[log p(X|Z)] + E_q[log p(Z|pi)] + E_q[log p(pi)]
               - E_q[log q(Z)] - E_q[log q(pi)]
        """
        n_reads, n_alleles = resp.shape
        e_log_pi = digamma(alpha_q) - digamma(alpha_q.sum())

        # Term 1: E_q[log p(X|Z)] = sum_r sum_k resp[r,k] * log p(x_r|z_r=k)
        # Only sum over finite log_lik entries
        safe_log_lik = np.where(np.isfinite(log_lik), log_lik, 0.0)
        term1 = float(np.sum(resp * safe_log_lik))

        # Term 2: E_q[log p(Z|pi)] = sum_r sum_k resp[r,k] * E_q[log pi_k]
        term2 = float(np.sum(resp * e_log_pi[np.newaxis, :]))

        # Term 3: E_q[log p(pi)] = log B(alpha_0) + sum_k (alpha_0_k - 1) * E_q[log pi_k]
        term3 = self._log_dirichlet_normalization(alpha_0)
        term3 += float(np.sum((alpha_0 - 1.0) * e_log_pi))

        # Term 4: -E_q[log q(Z)] = -sum_r sum_k resp[r,k] * log resp[r,k]  (entropy)
        safe_resp = np.where(resp > 1e-300, resp, 1e-300)
        term4 = -float(np.sum(resp * np.log(safe_resp)))

        # Term 5: -E_q[log q(pi)] = -[log B(alpha_q) + sum_k (alpha_q_k - 1) * E_q[log pi_k]]
        term5 = -self._log_dirichlet_normalization(alpha_q)
        term5 -= float(np.sum((alpha_q - 1.0) * e_log_pi))

        elbo = term1 + term2 + term3 + term4 + term5
        return elbo

    @staticmethod
    def _log_dirichlet_normalization(alpha: np.ndarray) -> float:
        """Compute log B(alpha) = sum(gammaln(alpha_k)) - gammaln(sum(alpha_k))."""
        return float(np.sum(gammaln(alpha)) - gammaln(np.sum(alpha)))

    def _enumerate_pairs(
        self,
        matrix: ReadAlleleMatrix,
        weights: np.ndarray,
        responsibilities: np.ndarray,
    ) -> list[tuple[int, int]]:
        """Enumerate candidate diploid pairs by weight."""
        n = matrix.n_alleles
        pairs = []
        for i in range(n):
            for j in range(i, n):
                score = weights[i] + weights[j]
                pairs.append((i, j, score))

        pairs.sort(key=lambda x: -x[2])
        return [(i, j) for i, j, _ in pairs[:self.n_top_pairs * 3]]

    def _compute_pair_posteriors(
        self,
        matrix: ReadAlleleMatrix,
        pairs: list[tuple[int, int]],
    ) -> list[tuple[int, int, float]]:
        """Compute marginal log-likelihood for each diploid pair -> posteriors.

        For each pair (a1, a2), compute:
            log P(X | genotype={a1,a2}) = sum_r log[ (P(x_r|a1) + P(x_r|a2)) / 2 ]
        Then convert to posteriors via softmax with uniform prior.
        """
        M = matrix.matrix
        n_reads = M.shape[0]

        # Normalize scores to pseudo-probabilities per read
        row_max = M.max(axis=1, keepdims=True)
        row_max[row_max == 0] = 1
        M_prob = M / row_max  # values in [0, 1]

        log_likes = []
        for a1, a2 in pairs:
            ll = 0.0
            for i in range(n_reads):
                p1, p2 = M_prob[i, a1], M_prob[i, a2]
                if p1 > 0 or p2 > 0:
                    # Diploid mixture: equal probability from either allele
                    p_mix = (p1 + p2) / 2.0
                    ll += np.log(max(p_mix, 1e-300))
                else:
                    # Read doesn't map to either allele — small penalty
                    ll += np.log(1e-10)
            log_likes.append(ll)

        if not log_likes:
            return []

        # Convert to posteriors via softmax (assumes uniform prior over pairs)
        log_likes_arr = np.array(log_likes)
        log_likes_arr -= log_likes_arr.max()
        probs = np.exp(log_likes_arr)
        total = probs.sum()
        if total > 0:
            probs /= total

        results = [
            (pairs[i][0], pairs[i][1], float(probs[i]))
            for i in range(len(pairs))
        ]
        results.sort(key=lambda x: -x[2])
        return results[:self.n_top_pairs]

    def posterior_entropy(self, pair_posteriors: list[tuple[int, int, float]]) -> float:
        """Compute Shannon entropy of the posterior distribution over pairs.

        High entropy = high ambiguity (many competing pairs).
        Low entropy = single dominant pair (clear call).
        Returns entropy in nats.
        """
        probs = np.array([p for _, _, p in pair_posteriors if p > 0])
        if len(probs) == 0:
            return 0.0
        probs = probs / probs.sum()  # normalize
        return -float(np.sum(probs * np.log(np.where(probs > 1e-300, probs, 1e-300))))

    def _trivial_result(
        self, locus: str, matrix: ReadAlleleMatrix
    ) -> ConfidenceResult:
        """Result for degenerate cases (0 or 1 allele)."""
        if matrix.allele_names:
            a = matrix.allele_names[0]
            # Single allele: compute coverage-based confidence
            reads_mapping = int(np.sum(matrix.matrix[:, 0] > 0)) if matrix.n_reads > 0 else 0
            conf = "HIGH" if reads_mapping > 10 else ("MEDIUM" if reads_mapping > 3 else "LOW")
            post = min(1.0, reads_mapping / 20.0) if reads_mapping > 0 else 0.0
        else:
            a, conf, post = "", "LOW", 0.0

        return ConfidenceResult(
            locus=locus, allele1=a, allele2=a,
            posterior_prob=post,
            allele1_dosage=0.5, allele2_dosage=0.5,
            confidence_class=conf,
            convergence_iterations=0,
            elbo=0.0,
            alternative_pairs=[],
        )
