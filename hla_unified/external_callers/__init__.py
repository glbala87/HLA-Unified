"""External HLA caller wrappers.

Provides a unified interface for running established, validated HLA
callers (OptiType, HLA*LA, T1K, arcasHLA) and converting their output
into the HLA-Unified format. This allows HLA-Unified to be used as an
orchestration layer over proven callers, leveraging its infrastructure
(validation, reporting, ambiguity classification, novel allele detection,
clinical reports) without requiring its own algorithms to be production-grade.

Usage:
    from hla_unified.external_callers import OptiTypeWrapper, HLALAWrapper

    wrapper = OptiTypeWrapper(work_dir="/tmp/optitype")
    if wrapper.is_available():
        calls = wrapper.run(bam_path="sample.bam")
        # calls: dict[locus, (allele1, allele2)]
"""

from .base import ExternalCaller, ExternalCallResult
from .optitype import OptiTypeWrapper
from .hlala import HLALAWrapper
from .t1k import T1KWrapper
from .arcas import ArcasHLAWrapper
from .orchestrator import (
    MultiCallerOrchestrator,
    OrchestrationResult,
    ConsensusCall,
)

__all__ = [
    "ExternalCaller",
    "ExternalCallResult",
    "OptiTypeWrapper",
    "HLALAWrapper",
    "T1KWrapper",
    "ArcasHLAWrapper",
    "MultiCallerOrchestrator",
    "OrchestrationResult",
    "ConsensusCall",
]
