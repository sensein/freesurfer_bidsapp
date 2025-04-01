#!/usr/bin/env python3
"""
BIDS App for FreeSurfer processing with NIDM output support.

This BIDS App runs FreeSurfer's recon-all pipeline on T1w and optionally T2w images
from a BIDS dataset and outputs both standard FreeSurfer results and NIDM format results.
"""

import datetime
import json
import logging
import os
import sys
from pathlib import Path

# Third-party libraries
import click
from bids.layout import BIDSLayout

# Local modules
from src.freesurfer.wrapper import FreeSurferWrapper
from src.nidm.fs2nidm import FreeSurferToNIDM
from src.utils import get_freesurfer_version, setup_logging, get_version_info

# Get version from package metadata
try:
    from importlib.metadata import version
    __version__ = version("bids-freesurfer")
except ImportError:
    __version__ = "0.1.0"

# Set up logging
logger = logging.getLogger("bids-freesurfer")


def process_participant(
    bids_dir,
    output_dir,
    participant_label,
    session_label,
    freesurfer_license,
    skip_bids_validator,
    fs_options,
    skip_nidm,
    freesurfer_dir,
    verbose,
):
    """Process a single participant."""
    # Set logging level based on verbosity
    setup_logging(logging.DEBUG if verbose else logging.INFO)

    # Get version information
    version_info = get_version_info()

    # Print version and check environment
    logger.info(f"BIDS-FreeSurfer version: {version_info['bids_freesurfer']['version']}")
    logger.info(f"FreeSurfer version: {version_info['freesurfer']['version']}")
    if version_info['freesurfer']['build_stamp']:
        logger.info(f"FreeSurfer build stamp: {version_info['freesurfer']['build_stamp']}")
    logger.info(f"Python version: {version_info['python']['version']}")
    if version_info['python']['packages']:
        logger.info("Python package versions:")
        for package, version in version_info['python']['packages'].items():
            logger.info(f"  {package}: {version}")

    # Convert all paths to Path objects
    bids_dir = Path(bids_dir)
    output_dir = Path(output_dir)
    if freesurfer_license:
        freesurfer_license = Path(freesurfer_license)

    # Create output directories
    nidm_dir = output_dir / "nidm"
    nidm_dir.mkdir(parents=True, exist_ok=True)

    # Set FreeSurfer subjects directory
    if freesurfer_dir:
        freesurfer_dir = Path(freesurfer_dir)
    else:
        freesurfer_dir = output_dir / "freesurfer"
    freesurfer_dir.mkdir(parents=True, exist_ok=True)

    # Load BIDS dataset
    try:
        # Convert Path to string for BIDSLayout
        layout = BIDSLayout(str(bids_dir), validate=not skip_bids_validator)
        logger.info(f"Found BIDS dataset with {len(layout.get_subjects())} subjects")
    except Exception as e:
        logger.error(f"Error loading BIDS dataset: {str(e)}")
        sys.exit(1)

    # Initialize FreeSurfer wrapper
    try:
        freesurfer_wrapper = FreeSurferWrapper(
            bids_dir,
            output_dir,
            freesurfer_license,
            fs_options
        )
    except Exception as e:
        logger.error(f"Error initializing FreeSurfer wrapper: {str(e)}")
        sys.exit(1)

    # Get subjects to analyze
    available_subjects = layout.get_subjects()
    if participant_label:
        subjects = list(participant_label)
        # Validate that all requested subjects exist
        invalid_subjects = [s for s in subjects if s not in available_subjects]
        if invalid_subjects:
            logger.error(f"Subject(s) {', '.join(invalid_subjects)} not found in dataset")
            sys.exit(1)
        logger.info(f"Processing specific subjects: {', '.join(subjects)}")
    else:
        subjects = available_subjects
        logger.info(f"Processing all {len(subjects)} subjects in dataset")

    # Get sessions to analyze
    if session_label:
        sessions = list(session_label)
        logger.info(f"Processing specific sessions: {', '.join(sessions)}")
    else:
        sessions = layout.get_sessions()
        if not sessions:
            logger.info("No sessions found in dataset, using session-less mode")
            sessions = [None]
        else:
            logger.info(f"Processing all {len(sessions)} sessions in dataset")

    # Run participant level analysis
    try:
        for subject in subjects:
            for session in sessions:
                # Process with FreeSurfer
                success = freesurfer_wrapper.process_subject(subject, session, layout)
                
                if success and not skip_nidm:
                    # Convert to NIDM format
                    try:
                        converter = FreeSurferToNIDM(
                            freesurfer_dir,
                            subject,
                            session,
                            nidm_dir
                        )
                        converter.convert()
                        logger.info(f"Successfully converted {subject} to NIDM format")
                    except Exception as e:
                        logger.error(f"Error converting {subject} to NIDM format: {str(e)}")

        # Save processing summary with version information
        summary = freesurfer_wrapper.get_processing_summary()
        summary["version_info"] = version_info
        freesurfer_wrapper.save_processing_summary(summary)

        # Print summary
        logger.info("================================")
        logger.info("Processing complete!")
        logger.info(f"Total subjects/sessions processed: {summary['total']}")
        logger.info(f"Successfully processed: {summary['success']}")
        logger.info(f"Failed to process: {summary['failure']}")
        logger.info(f"Skipped (already processed or missing data): {summary['skipped']}")
        logger.info("================================")

    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        sys.exit(1)

    logger.info("BIDS-FreeSurfer processing complete.")
    return 0


def process_group(
    bids_dir,
    output_dir,
    freesurfer_license,
    skip_bids_validator,
    verbose,
):
    """Process group level analysis."""
    # Set logging level based on verbosity
    setup_logging(logging.DEBUG if verbose else logging.INFO)

    # Print version and check environment
    logger.info(f"BIDS-FreeSurfer version: {__version__}")
    logger.info(f"FreeSurfer version: {get_freesurfer_version()}")

    # Convert paths to Path objects
    bids_dir = Path(bids_dir)
    output_dir = Path(output_dir)
    if freesurfer_license:
        freesurfer_license = Path(freesurfer_license)

    # Load BIDS dataset
    try:
        layout = BIDSLayout(str(bids_dir), validate=not skip_bids_validator)
        logger.info(f"Found BIDS dataset with {len(layout.get_subjects())} subjects")
    except Exception as e:
        logger.error(f"Error loading BIDS dataset: {str(e)}")
        sys.exit(1)

    # Check if we have any subjects processed
    freesurfer_dir = output_dir / "freesurfer"
    if not freesurfer_dir.exists():
        logger.error("No FreeSurfer output directory found. Please run participant level analysis first.")
        sys.exit(1)

    # Get list of processed subjects
    processed_subjects = [d.name for d in freesurfer_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    if not processed_subjects:
        logger.error("No processed subjects found. Please run participant level analysis first.")
        sys.exit(1)

    logger.info(f"Found {len(processed_subjects)} processed subjects")
    logger.info("Group level analysis not implemented yet")
    logger.info("In the future, this will include:")
    logger.info("- Group level statistics")
    logger.info("- Cross-subject comparisons")
    logger.info("- Population-level analyses")
    return 0


@click.group()
@click.version_option(version=__version__)
def cli():
    """FreeSurfer BIDS App with NIDM Output.

    This BIDS App runs FreeSurfer's recon-all pipeline on T1w and optionally T2w images
    from a BIDS dataset and outputs both standard FreeSurfer results and NIDM format results.
    """
    pass


@cli.command()
@click.argument(
    "bids_dir", type=click.Path(exists=True, file_okay=False, resolve_path=True)
)
@click.argument("output_dir", type=click.Path(file_okay=False, resolve_path=True))
@click.option(
    "--participant_label",
    multiple=True,
    help='The label(s) of the participant(s) to analyze (without "sub-").',
)
@click.option(
    "--session_label",
    multiple=True,
    help='The label(s) of the session(s) to analyze (without "ses-").',
)
@click.option(
    "--freesurfer_license",
    "--fs-license-file", 
    type=click.Path(exists=True, resolve_path=True),
    help="Path to FreeSurfer license file.",
)
@click.option("--skip_bids_validator", is_flag=True, help="Skip BIDS validation.")
@click.option(
    "--fs_options",
    help='Additional options for recon-all, e.g. "--fs_options=-parallel"',
)
@click.option("--skip_nidm", is_flag=True, help="Skip NIDM output generation.")
@click.option(
    "--freesurfer_dir",
    type=click.Path(file_okay=False, resolve_path=True),
    help="Path to FreeSurfer subjects directory (default: output_dir/freesurfer)",
)
@click.option("--verbose", is_flag=True, help="Enable verbose output.")
def participant(
    bids_dir,
    output_dir,
    participant_label,
    session_label,
    freesurfer_license,
    skip_bids_validator,
    fs_options,
    skip_nidm,
    freesurfer_dir,
    verbose,
):
    """Run participant level analysis."""
    return process_participant(
        bids_dir,
        output_dir,
        participant_label,
        session_label,
        freesurfer_license,
        skip_bids_validator,
        fs_options,
        skip_nidm,
        freesurfer_dir,
        verbose,
    )


@cli.command()
@click.argument(
    "bids_dir", type=click.Path(exists=True, file_okay=False, resolve_path=True)
)
@click.argument("output_dir", type=click.Path(file_okay=False, resolve_path=True))
@click.option(
    "--freesurfer_license",
    "--fs-license-file", 
    type=click.Path(exists=True, resolve_path=True),
    help="Path to FreeSurfer license file.",
)
@click.option("--skip_bids_validator", is_flag=True, help="Skip BIDS validation.")
@click.option("--verbose", is_flag=True, help="Enable verbose output.")
def group(
    bids_dir,
    output_dir,
    freesurfer_license,
    skip_bids_validator,
    verbose,
):
    """Run group level analysis."""
    return process_group(
        bids_dir,
        output_dir,
        freesurfer_license,
        skip_bids_validator,
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
