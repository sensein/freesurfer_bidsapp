#!/usr/bin/env python3
"""
BIDS App for FreeSurfer processing with NIDM output support.

This BIDS App runs FreeSurfer's recon-all pipeline on T1w and optionally T2w images
from a BIDS dataset and outputs both standard FreeSurfer results and NIDM format results.
"""

import logging
import os
import subprocess
import sys
from pathlib import Path

import click

from bids import BIDSLayout
from src.freesurfer.wrapper import FreeSurferWrapper
from src.utils import get_version_info, setup_logging

try:
    from importlib.metadata import version

    __version__ = version("bids-freesurfer")
except ImportError:
    __version__ = "0.1.0"

logger = logging.getLogger("bids-freesurfer")


def _log_version_info(version_info):
    """Log version information."""
    logger.info(f"BIDS-FreeSurfer version: {version_info['bids_freesurfer']['version']}")
    logger.info(f"FreeSurfer version: {version_info['freesurfer']['version']}")
    if version_info["freesurfer"]["build_stamp"]:
        logger.info(f"FreeSurfer build stamp: {version_info['freesurfer']['build_stamp']}")
    logger.info(f"Python version: {version_info['python']['version']}")
    if version_info["python"]["packages"]:
        logger.info("Python package versions:")
        for package, version in version_info["python"]["packages"].items():
            logger.info(f"  {package}: {version}")


def initialize(bids_dir, freesurfer_license, output_dir, skip_bids_validation, verbose):
    """Initialize the BIDS-FreeSurfer app.
    Args:
        bids_dir (str): Path to BIDS root directory
        freesurfer_license (str): Path to FreeSurfer license file
        output_dir (str): Path to output directory
        skip_bids_validation (bool): Skip BIDS validation
        verbose (bool): Enable verbose output
    """
    # Convert paths to Path objects
    bids_dir = Path(bids_dir)

    # First create the main output directory
    app_output_dir = Path(output_dir) / "freesurfer_bidsapp"
    freesurfer_dir = app_output_dir / "freesurfer"
    nidm_dir = app_output_dir / "nidm"

    # Set logging level and print version info
    setup_logging(logging.DEBUG if verbose else logging.INFO)
    version_info = get_version_info()
    _log_version_info(version_info)

    if freesurfer_license:
        freesurfer_license = Path(freesurfer_license)

    # Set FreeSurfer environment variables
    os.environ["FS_ALLOW_DEEP"] = "1"  # Enable ML routines
    os.environ["SUBJECTS_DIR"] = str(freesurfer_dir)

    # Create FreeSurfer output directory
    os.makedirs(freesurfer_dir, exist_ok=True)

    # Load BIDS dataset
    try:
        layout = BIDSLayout(str(bids_dir), validate=not skip_bids_validation)
        logger.info("Found BIDS dataset")
    except Exception as e:
        logger.error(f"Error loading BIDS dataset: {str(e)}")
        sys.exit(1)

    # Let the FreeSurfer wrapper handle its directory
    try:
        freesurfer_wrapper = FreeSurferWrapper(
            bids_dir,
            app_output_dir,  # Pass the app_output_dir to FreeSurferWrapper
            freesurfer_license,
        )
    except Exception as e:
        logger.error(f"Error initializing FreeSurfer wrapper: {str(e)}")
        sys.exit(1)

    return layout, freesurfer_wrapper, freesurfer_dir, nidm_dir, version_info


def nidm_conversion(nidm_dir, freesurfer_dir, participant_label, freesurfer_wrapper, bids_session=None, verbose=False):
    """Convert FreeSurfer outputs to NIDM format.
    Args:
        nidm_dir (str): Path to NIDM output directory
        freesurfer_dir (str): Path to FreeSurfer output directory
        participant_label (str): Participant label (without "sub-" prefix)
        freesurfer_wrapper (FreeSurferWrapper): Instance of FreeSurferWrapper containing T1 info
        bids_session (str): Session label (without "ses-" prefix)
        verbose (bool): Enable verbose output
    """
    # Determine subject directory with session info
    if bids_session is None:
        fs_subject_id = participant_label
    else:
        fs_subject_id = f"{participant_label}_ses-{bids_session}"
    subject_dir = os.path.join(freesurfer_dir, fs_subject_id)

    # Get T1 and T2 image information
    t1_info = freesurfer_wrapper.get_subject_t1_info(participant_label, bids_session)
    t1_images = t1_info.get('T1w_images', [])
    t2_images = t1_info.get('T2w_images', [])
    if not t1_images:
        logger.warning(f"No T1 image information found for {fs_subject_id}")

    os.makedirs(nidm_dir, exist_ok=True)

    # Build the command
    fs_to_nidm_path = os.path.join(os.path.dirname(__file__), "segstats_jsonld", "segstats_jsonld", "fs_to_nidm.py")
    cmd = [
        "python3",
        fs_to_nidm_path,
        "-s",
        subject_dir,  # subject directory
        "-o",
        nidm_dir,  # output directory
        "-j",  # output as JSON-LD
    ]

    # Add T1 image information if available
    if t1_images:
        cmd.extend(["--t1"] + t1_images)

    # Add T2 image information if available
    if t2_images:
        cmd.extend(["--t2"] + t2_images)

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        logger.info("================================")
        logger.info(f"NIDM conversion complete for {fs_subject_id}")
        logger.info("================================")
    else:
        logger.error(f"NIDM conversion failed for {fs_subject_id}")
        logger.error(f"Error output: {result.stderr}")
        if verbose:
            logger.error(f"Command output: {result.stdout}")
        sys.exit(1)


def process_participant(
    bids_dir,
    output_dir,
    participant_label,
    freesurfer_license,
    skip_bids_validation,
    skip_nidm,
    verbose,
):
    """Process a single participant with FreeSurfer.

    Args:
        bids_dir (str): Path to BIDS root directory
        output_dir (str): Path to output directory
        participant_label (str): Participant label (without "sub-" prefix)
        freesurfer_license (str): Path to FreeSurfer license file
        skip_bids_validation (bool): Skip BIDS validation
        skip_nidm (bool): Skip NIDM export
        verbose (bool): Enable verbose output
    """
    layout, freesurfer_wrapper, freesurfer_dir, nidm_dir, version_info = initialize(
        bids_dir, freesurfer_license, output_dir, skip_bids_validation, verbose
    )

    # Validate that the subject exists (strip "sub-" for BIDS query)
    available_subjects = layout.get_subjects()
    bids_subject = participant_label[4:]  # Strip "sub-" for BIDS query
    if bids_subject not in available_subjects:
        logger.error(f"Subject {participant_label} not found in dataset")
        sys.exit(1)

    # Run participant analysis
    try:
        success = freesurfer_wrapper.process_subject(participant_label, layout)
        # Save processing summary
        summary = freesurfer_wrapper.get_processing_summary()
        summary["version_info"] = version_info
        freesurfer_wrapper.save_processing_summary(summary)

        logger.info("================================")
        logger.info("Processing complete!")
        logger.info("================================")

    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        sys.exit(1)

    if success and not skip_nidm:
        nidm_conversion(nidm_dir, freesurfer_dir, participant_label, freesurfer_wrapper, verbose=verbose)

    logger.info("BIDS-FreeSurfer processing complete.")
    return 0


def process_session(
    bids_dir,
    output_dir,
    participant_label,
    session_label,
    freesurfer_license,
    skip_bids_validation,
    skip_nidm,
    verbose,
):
    """Process a single session for a participant with FreeSurfer.

    Args:
        bids_dir (str): Path to BIDS root directory
        output_dir (str): Path to output directory
        participant_label (str): Participant label (without "sub-" prefix)
        session_label (str): Session label (without "ses-" prefix)
        freesurfer_license (str): Path to FreeSurfer license file
        skip_bids_validation (bool): Skip BIDS validation
        skip_nidm (bool): Skip NIDM export
        verbose (bool): Enable verbose output
    """
    layout, freesurfer_wrapper, freesurfer_dir, nidm_dir, version_info = initialize(
        bids_dir, freesurfer_license, output_dir, skip_bids_validation, verbose
    )

    # Validate that the subject exists (strip "sub-" for BIDS query)
    available_subjects = layout.get_subjects()
    bids_subject = participant_label[4:]  # Strip "sub-" for BIDS query
    if bids_subject not in available_subjects:
        logger.error(f"Subject {participant_label} not found in dataset")
        sys.exit(1)

    # Validate that the session exists
    available_sessions = layout.get_sessions(subject=bids_subject)
    bids_session = session_label[4:] if session_label.startswith("ses-") else session_label  # Strip "ses-" if present
    if bids_session not in available_sessions:
        logger.error(f"Session {session_label} not found for subject {participant_label}")
        sys.exit(1)

    # Run session-level analysis
    try:
        # Use the enhanced process_subject method with session_label
        success = freesurfer_wrapper.process_subject(participant_label, layout, session_label=bids_session)
        # Save processing summary
        summary = freesurfer_wrapper.get_processing_summary()
        summary["version_info"] = version_info
        freesurfer_wrapper.save_processing_summary(summary)
        logger.info("================================")
        logger.info("Processing complete!")
        logger.info("================================")

    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        sys.exit(1)

    if success and not skip_nidm:
        nidm_conversion(nidm_dir, freesurfer_dir, participant_label, freesurfer_wrapper, bids_session, verbose=verbose)

    logger.info("BIDS-FreeSurfer processing complete.")
    return 0


@click.command()
@click.version_option(version=__version__)
@click.argument("bids_dir", type=click.Path(exists=True, file_okay=False, resolve_path=True))
@click.argument("output_dir", type=click.Path(file_okay=False, resolve_path=True))
@click.argument(
    "analysis_level",
    type=click.Choice(["participant", "session"]),
)
@click.option(
    "--participant_label",
    "--participant-label",
    help='The label of the participant to analyze (including "sub-" prefix, e.g., "sub-001").',
)
@click.option(
    "--session_label",
    "--session-label",
    help='The label of the session to analyze (including "ses-" prefix, e.g., "ses-01"). Only used with "session" analysis level.',
)
@click.option(
    "--freesurfer_license",
    "--fs-license-file",
    type=click.Path(exists=True, resolve_path=True),
    help="Path to FreeSurfer license file.",
)
@click.option("--skip-bids-validation", is_flag=True, help="Skip BIDS validation.")
@click.option("--skip_nidm", "--skip-nidm", is_flag=True, help="Skip NIDM output generation.")
@click.option("--verbose", is_flag=True, help="Enable verbose output.")
def cli(
    bids_dir,
    output_dir,
    analysis_level,
    participant_label,
    session_label,
    freesurfer_license,
    skip_bids_validation,
    skip_nidm,
    verbose,
):
    """FreeSurfer BIDS App with NIDM Output.

    This BIDS App runs FreeSurfer's recon-all pipeline on T1w images from a BIDS dataset.
    It supports individual participant analysis and can generate NIDM outputs.

    BIDS_DIR is the path to the BIDS dataset directory.

    OUTPUT_DIR is the path where results will be stored.

    ANALYSIS_LEVEL determines the processing stage to be run:
    - 'participant': processes a single subject
    - 'session': processes a single session for a subject
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    if analysis_level == "participant":
        if not participant_label:
            logger.error("Participant label is required for participant-level analysis")
            sys.exit(1)
        return process_participant(
            bids_dir,
            output_dir,
            participant_label,
            freesurfer_license,
            skip_bids_validation,
            skip_nidm,
            verbose,
        )
    elif analysis_level == "session":
        if not participant_label or not session_label:
            logger.error("Both participant and session labels are required for session-level analysis")
            sys.exit(1)
        return process_session(
            bids_dir,
            output_dir,
            participant_label,
            session_label,
            freesurfer_license,
            skip_bids_validation,
            skip_nidm,
            verbose,
        )


def main():
    """Entry point for the CLI."""
    try:
        cli()
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
