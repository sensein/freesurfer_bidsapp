"""
BIDS integration module for FreeSurfer

This package provides utilities for BIDS compatibility and provenance tracking
for FreeSurfer outputs in BIDS-compliant derivatives.
"""

from .provenance import BIDSProvenance, create_bids_provenance

__all__ = ['BIDSProvenance', 'create_bids_provenance']