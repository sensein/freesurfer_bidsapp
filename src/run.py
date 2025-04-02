#!/usr/bin/env python3
"""
BIDS App for FreeSurfer processing with NIDM output support.

This BIDS App runs FreeSurfer's recon-all pipeline on T1w and optionally T2w images
from a BIDS dataset and outputs both standard FreeSurfer results and NIDM format results.
"""

import logging
import sys
from pathlib import Path
import os

import click
from bids import BIDSLayout

from src.freesurfer.wrapper import FreeSurferWrapper
from src.nidm.fs2nidm import FreeSurferToNIDM
from src.utils import get_freesurfer_version, setup_logging, get_version_info

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
    if version_info['freesurfer']['build_stamp']:
        logger.info(f"FreeSurfer build stamp: {version_info['freesurfer']['build_stamp']}")
    logger.info(f"Python version: {version_info['python']['version']}")
    if version_info['python']['packages']:
        logger.info("Python package versions:")
        for package, version in version_info['python']['packages'].items():
            logger.info(f"  {package}: {version}")


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
    freesurfer_dir = os.path.join(output_dir, "freesurfer")
    nidm_dir = os.path.join(output_dir, "nidm")

    # Set logging level and print version info
    setup_logging(logging.DEBUG if verbose else logging.INFO)
    version_info = get_version_info()
    _log_version_info(version_info)

    # Convert paths to Path objects
    bids_dir = Path(bids_dir)
    output_dir = Path(output_dir)
    if freesurfer_license:
        freesurfer_license = Path(freesurfer_license)

    # Set FreeSurfer environment variables
    os.environ['FS_ALLOW_DEEP'] = '1'  # Enable ML routines
    os.environ['SUBJECTS_DIR'] = freesurfer_dir  # Use the already defined freesurfer_dir
    
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
            output_dir,
            freesurfer_license,
        )
    except Exception as e:
        logger.error(f"Error initializing FreeSurfer wrapper: {str(e)}")
        sys.exit(1)

    # Validate that the subject exists (strip "sub-" for BIDS query)
    available_subjects = layout.get_subjects()
    bids_subject = participant_label[4:]  # Strip "sub-" for BIDS query
    if bids_subject not in available_subjects:
        logger.error(f"Subject {participant_label} not found in dataset")
        sys.exit(1)

    # Run participant analysis
    try:
        success = freesurfer_wrapper.process_subject(participant_label, layout)
        
        if success and not skip_nidm:
            os.makedirs(nidm_dir, exist_ok=True)

            try:
                converter = FreeSurferToNIDM(
                    freesurfer_dir=freesurfer_dir,
                    bids_dir=bids_dir,  # Pass BIDS directory
                    subject_id=participant_label,  # Add sub- prefix
                    output_dir=nidm_dir
                )
                converter.convert()
                logger.info(f"NIDM conversion complete for {participant_label}")
            except Exception as e:
                logger.error(f"NIDM conversion failed for {participant_label}: {str(e)}")
                if verbose:
                    logger.exception(e)

        # Save processing summary
        summary = freesurfer_wrapper.get_processing_summary()
        summary["version_info"] = version_info
        freesurfer_wrapper.save_processing_summary(summary)

        logger.info("================================")
        logger.info("Processing complete!")
        logger.info("Subject was successfully processed" if success else "Subject processing failed")
        logger.info("================================")

    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        sys.exit(1)

    logger.info("BIDS-FreeSurfer processing complete.")
    return 0


@click.command()
@click.version_option(version=__version__)
@click.argument(
    "bids_dir", type=click.Path(exists=True, file_okay=False, resolve_path=True)
)
@click.argument("output_dir", type=click.Path(file_okay=False, resolve_path=True))
@click.option(
    "--participant_label", "--participant-label",
    help='The label of the participant to analyze (including "sub-" prefix, e.g., "sub-001").',
)
@click.option(
    "--freesurfer_license",
    "--fs-license-file", 
    type=click.Path(exists=True, resolve_path=True),
    help="Path to FreeSurfer license file.",
)
@click.option(
    "--skip-bids-validation",
    is_flag=True, 
    help="Skip BIDS validation."
)
@click.option("--skip_nidm", "--skip-nidm", is_flag=True, help="Skip NIDM output generation.")
@click.option("--verbose", is_flag=True, help="Enable verbose output.")
def cli(
    bids_dir,
    output_dir,
    participant_label,
    freesurfer_license,
    skip_bids_validation,
    skip_nidm,
    verbose,
):
    """FreeSurfer BIDS App with NIDM Output."""
    return process_participant(
        bids_dir,
        output_dir,
        participant_label,
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
