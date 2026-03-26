"""HLA-Unified: Multi-strategy HLA typing combining the best algorithms.

Integrates approaches from:
- xHLA (fast pre-filtering)
- HLA-HD (iterative refinement)
- OptiType (ILP genotyping)
- HLA-VBSeq (Bayesian confidence)
- HLAminer (assembly fallback)
- HLAforest (k-mer validation)
- PHLAT (diploid likelihood)
- HLAscan (full-gene coverage)
"""

__version__ = "2.0.0"
