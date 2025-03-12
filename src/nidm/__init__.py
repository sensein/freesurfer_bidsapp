"""
Neuroimaging Data Model (NIDM) conversion module for FreeSurfer

This package provides functions and classes to convert FreeSurfer outputs
to NIDM format for standardized representation and interoperability.
"""

from .fs2nidm import FreeSurferToNIDM, convert_subject, create_group_nidm

__all__ = ['FreeSurferToNIDM', 'convert_subject', 'create_group_nidm']