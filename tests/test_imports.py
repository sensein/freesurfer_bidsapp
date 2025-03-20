def test_standard_library_imports():
    """Test that all standard library imports work"""
    try:
        import datetime
        import json
        import logging
        import os
        import shutil
        import subprocess
        import sys
        from pathlib import Path
        assert True  # If we get here, imports worked
    except ImportError as e:
        assert False, f"Standard library import failed: {e}"

def test_third_party_imports():
    """Test that all third-party package imports work"""
    try:
        import click
        import numpy as np
        import rdflib
        from rdflib import Graph, Literal, Namespace, URIRef
        from rdflib.namespace import RDF, RDFS, XSD
        from bids.layout import BIDSLayout
        assert True  # If we get here, imports worked
    except ImportError as e:
        assert False, f"Third-party import failed: {e}"

def test_local_imports():
    """Test that all local module imports work"""
    try:
        from src.bids.provenance import BIDSProvenance, create_bids_provenance
        from src.freesurfer.wrapper import FreeSurferWrapper
        from src.nidm.fs2nidm import FreeSurferToNIDM, convert_subject, create_group_nidm
        from src.utils import get_freesurfer_version, setup_logging
        assert True  # If we get here, imports worked
    except ImportError as e:
        assert False, f"Local import failed: {e}" 