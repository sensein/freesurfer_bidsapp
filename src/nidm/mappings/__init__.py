"""
FreeSurfer to NIDM terminology mappings

This package provides mappings between FreeSurfer terms and standard neuroimaging ontologies.
"""

import json
import os
from pathlib import Path


def get_mapping_path(filename):
    """
    Get the absolute path to a mapping file in this directory.

    Parameters
    ----------
    filename : str
        Name of the mapping file

    Returns
    -------
    str
        Absolute path to the mapping file
    """
    return os.path.join(os.path.dirname(__file__), filename)


def load_json_mapping(filename="fsmap.json"):
    """
    Load a JSON mapping file.

    Parameters
    ----------
    filename : str, optional
        Name of the JSON mapping file (default: fsmap.json)

    Returns
    -------
    dict
        Mapping dictionary
    """
    mapping_path = get_mapping_path(filename)

    try:
        with open(mapping_path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading mapping file {mapping_path}: {e}")
        return {}
