#!/usr/bin/env python3
"""
Utility functions for BIDS-FreeSurfer Application.

This module provides common utility functions used across the BIDS-FreeSurfer
application, including environment setup, version checks, and logging configuration.
"""

import logging
import os
import re
import subprocess
import sys
from pathlib import Path
import datetime


def get_freesurfer_version():
    """
    Get FreeSurfer version string from Dockerfile.

    Returns
    -------
    str
        FreeSurfer version string, or "unknown" if not available
    """
    try:
        # Read version from Dockerfile
        dockerfile_path = Path(__file__).parent.parent / "Dockerfile"
        if dockerfile_path.exists():
            with open(dockerfile_path, "r") as f:
                for line in f:
                    if line.startswith("FROM"):
                        # Extract version from image name (e.g., "FROM vnmd/freesurfer_8.0.0")
                        image = line.strip().split()[1]
                        version_match = re.search(r"(\d+\.\d+\.\d+)", image)
                        if version_match:
                            return version_match.group(1)

        return "unknown"
    except Exception as e:
        logging.warning(f"Failed to get FreeSurfer version from Dockerfile: {str(e)}")
        return "unknown"

def setup_logging(log_level=logging.INFO, log_file=None):
    """
    Configure logging for the application.

    Parameters
    ----------
    log_level : int, optional
        Logging level (e.g., logging.INFO, logging.DEBUG)
    log_file : str, optional
        Path to log file (if None, logs to console only)
    """
    # Define log format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Reset root logger
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Configure logging
    if log_file:
        # Create directory for log file if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        # Set up file handler
        handlers = [logging.FileHandler(log_file), logging.StreamHandler()]
    else:
        handlers = [logging.StreamHandler()]

    # Configure root logger
    logging.basicConfig(level=log_level, format=log_format, handlers=handlers)

    # Set log level for specific loggers
    # Silence overly verbose packages
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("nibabel").setLevel(logging.WARNING)


def check_dependencies():
    """
    Check if all required dependencies are installed.

    Returns
    -------
    bool
        True if all dependencies are met, False otherwise
    """
    # Check for FreeSurfer
    if "FREESURFER_HOME" not in os.environ:
        logging.error("FREESURFER_HOME environment variable not set")
        return False

    # Check for FreeSurfer license
    license_locations = [
        os.environ.get("FS_LICENSE"),
        "/license.txt",  # Docker mount location
        os.path.join(os.environ.get("FREESURFER_HOME", ""), "license.txt"),
        os.path.expanduser("~/.freesurfer.txt"),
    ]

    has_license = False
    for loc in license_locations:
        if loc and os.path.exists(loc):
            has_license = True
            break

    if not has_license:
        logging.warning("FreeSurfer license not found in standard locations")

    # Check for recon-all command
    try:
        subprocess.run(
            ["recon-all", "--help"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        logging.error("recon-all command not found in PATH")
        return False

    return True


def validate_bids_dataset(bids_dir, validate=True):
    """
    Validate BIDS dataset, returning a BIDSLayout object.

    Parameters
    ----------
    bids_dir : str
        Path to BIDS dataset
    validate : bool, optional
        Whether to validate the dataset against BIDS specification

    Returns
    -------
    BIDSLayout
        BIDS layout object

    Raises
    ------
    ValueError
        If dataset validation fails
    """
    try:
        from bids import BIDSLayout

        layout = BIDSLayout(bids_dir, validate=validate)
        subjects = layout.get_subjects()

        if not subjects:
            logging.warning(f"No subjects found in BIDS dataset at {bids_dir}")
        else:
            logging.info(f"Found {len(subjects)} subjects in BIDS dataset")

        return layout
    except Exception as e:
        logging.error(f"Failed to validate BIDS dataset: {str(e)}")
        raise ValueError(f"Invalid BIDS dataset: {str(e)}")


def get_version_info():
    """
    Get comprehensive version information for the application.

    Returns
    -------
    dict
        Dictionary containing version information for all components
    """
    version_info = {
        "bids_freesurfer": {
            "version": "unknown",
            "source": "package",
            "timestamp": datetime.datetime.now().isoformat()
        },
        "freesurfer": {
            "version": get_freesurfer_version(),
            "source": "base_image",
            "build_stamp": None,
            "image": None
        },
        "python": {
            "version": sys.version,
            "packages": {}
        }
    }

    # Get BIDS app version from setup.py
    try:
        setup_path = Path(__file__).parent.parent / "setup.py"
        if setup_path.exists():
            with open(setup_path, "r") as f:
                for line in f:
                    if line.startswith('    version="'):
                        version = line.strip().split('"')[1]
                        version_info["bids_freesurfer"]["version"] = version
                        version_info["bids_freesurfer"]["source"] = "setup.py"
                        break
    except Exception as e:
        logging.warning(f"Failed to read setup.py: {str(e)}")

    # Fallback to package version if setup.py not available
    if version_info["bids_freesurfer"]["version"] == "unknown":
        try:
            from importlib.metadata import version
            version_info["bids_freesurfer"]["version"] = version("bids-freesurfer")
            version_info["bids_freesurfer"]["source"] = "package"
        except ImportError:
            try:
                from pkg_resources import get_distribution
                version_info["bids_freesurfer"]["version"] = get_distribution("bids-freesurfer").version
            except Exception:
                pass

    # Get FreeSurfer image from Dockerfile
    try:
        dockerfile_path = Path(__file__).parent.parent / "Dockerfile"
        if dockerfile_path.exists():
            with open(dockerfile_path, "r") as f:
                for line in f:
                    if line.startswith("FROM"):
                        # Extract image name (e.g., "FROM vnmd/freesurfer_8.0.0")
                        image = line.strip().split()[1]
                        version_info["freesurfer"]["image"] = image
                        # Extract version from image name
                        if "_" in image:
                            version_info["freesurfer"]["version"] = image.split("_")[-1]
    except Exception as e:
        logging.warning(f"Failed to read Dockerfile: {str(e)}")

    # Get build stamp if available (as additional information)
    fs_home = os.environ.get("FREESURFER_HOME")
    if fs_home:
        build_stamp_path = Path(fs_home) / "build-stamp.txt"
        if build_stamp_path.exists():
            try:
                with open(build_stamp_path, "r") as f:
                    build_stamp = f.read().strip()
                    version_info["freesurfer"]["build_stamp"] = build_stamp
            except Exception as e:
                logging.warning(f"Failed to read build-stamp.txt: {str(e)}")

    # Get Python package versions
    try:
        import pkg_resources
        for package in ["numpy", "pandas", "nibabel", "bids", "rdflib"]:
            try:
                version_info["python"]["packages"][package] = pkg_resources.get_distribution(package).version
            except pkg_resources.DistributionNotFound:
                pass
    except Exception:
        pass

    return version_info
