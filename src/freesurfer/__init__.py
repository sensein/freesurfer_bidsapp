"""
FreeSurfer BIDS integration package

This package provides tools for running FreeSurfer's recon-all command
on BIDS datasets and organizing the outputs in a BIDS-compliant format.
"""

from .wrapper import FreeSurferWrapper

__all__ = ['FreeSurferWrapper', 'create_parser', 'main']