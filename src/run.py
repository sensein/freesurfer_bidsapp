#!/usr/bin/env python3
"""
BIDS App for FreeSurfer processing with NIDM output support.

This BIDS App runs FreeSurfer's recon-all pipeline on T1w and optionally T2w images
from a BIDS dataset and outputs both standard FreeSurfer results and NIDM format results.
"""

import datetime
import json
import logging

# Standard library imports
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Third-party libraries
import click
import numpy as np
import rdflib
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, XSD

from bids.layout import BIDSLayout
from src.bids.provenance import BIDSProvenance, create_bids_provenance

# Local modules
from src.freesurfer.wrapper import FreeSurferWrapper
from src.nidm.fs2nidm import FreeSurferToNIDM, convert_subject, create_group_nidm
from src.utils import get_freesurfer_version, setup_logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("bids-freesurfer")


@click.command()
@click.argument(
    "bids_dir", type=click.Path(exists=True, file_okay=False, resolve_path=True)
)
@click.argument("output_dir", type=click.Path(file_okay=False, resolve_path=True))
@click.argument("analysis_level", type=click.Choice(["participant", "group"]))
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
@click.option("--verbose", is_flag=True, help="Enable verbose output.")
def main(
    bids_dir,
    output_dir,
    analysis_level,
    participant_label,
    session_label,
    freesurfer_license,
    skip_bids_validator,
    fs_options,
    skip_nidm,
    verbose,
):
    """FreeSurfer BIDS App with NIDM Output.

    This BIDS App runs FreeSurfer's recon-all pipeline on T1w and optionally T2w images
    from a BIDS dataset and outputs both standard FreeSurfer results and NIDM format results.
    """
    # Set logging level based on verbosity
    log_level = logging.DEBUG if verbose else logging.INFO
    setup_logging(log_level)

    # Print version and check environment
    version = "0.1.0"
    logger.info(f"BIDS-FreeSurfer version: {version}")
    logger.info(f"FreeSurfer version: {get_freesurfer_version()}")

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    freesurfer_dir = os.path.join(output_dir, "freesurfer")
    os.makedirs(freesurfer_dir, exist_ok=True)
    nidm_dir = os.path.join(output_dir, "nidm")
    os.makedirs(nidm_dir, exist_ok=True)

    # Create BIDS provenance
    bids_provenance = create_bids_provenance(bids_dir, output_dir, version)

    # Load BIDS dataset
    try:
        layout = BIDSLayout(bids_dir, validate=not skip_bids_validator)
        logger.info(f"Found BIDS dataset with {len(layout.get_subjects())} subjects")
    except Exception as e:
        logger.error(f"Error loading BIDS dataset: {str(e)}")
        sys.exit(1)

    # Initialize FreeSurfer wrapper
    try:
        freesurfer_wrapper = FreeSurferWrapper(
            bids_dir, freesurfer_dir, freesurfer_license, fs_options
        )
    except Exception as e:
        logger.error(f"Error initializing FreeSurfer wrapper: {str(e)}")
        sys.exit(1)

    # Get subjects to analyze
    subjects_to_analyze = get_subjects_to_analyze(layout, participant_label)
    sessions_to_analyze = get_sessions_to_analyze(layout, session_label)

    # Run analysis
    try:
        if analysis_level == "participant":
            run_participant_level(
                layout,
                subjects_to_analyze,
                sessions_to_analyze,
                freesurfer_wrapper,
                bids_provenance,
                freesurfer_dir,
                nidm_dir,
                output_dir,
                skip_nidm,
            )
        else:  # group level
            run_group_level(
                subjects_to_analyze,
                bids_provenance,
                freesurfer_dir,
                nidm_dir,
                skip_nidm,
            )
    except Exception as e:
        logger.error(f"Error during {analysis_level} level processing: {str(e)}")
        sys.exit(1)

    logger.info("BIDS-FreeSurfer processing complete.")
    return 0


def get_subjects_to_analyze(layout, participant_label):
    """
    Get list of subjects to analyze.

    Parameters:
    -----------
    layout : BIDSLayout
        The BIDS layout object
    participant_label : list
        List of participant labels to include

    Returns:
    --------
    list
        List of subject IDs with 'sub-' prefix
    """
    if participant_label:
        subjects = ["sub-" + label for label in participant_label]
        # Verify subjects exist in dataset
        available_subjects = ["sub-" + s for s in layout.get_subjects()]
        missing = [s for s in subjects if s not in available_subjects]
        if missing:
            logger.warning(
                f"Requested participants not found in dataset: {', '.join(missing)}"
            )
        return [s for s in subjects if s in available_subjects]
    else:
        return ["sub-" + s for s in layout.get_subjects()]


def get_sessions_to_analyze(layout, session_label):
    """
    Get list of sessions to analyze.

    Parameters:
    -----------
    layout : BIDSLayout
        The BIDS layout object
    session_label : list
        List of session labels to include

    Returns:
    --------
    list
        List of session IDs (may be empty if no sessions in dataset)
    """
    if session_label:
        sessions = list(session_label)
        available_sessions = layout.get_sessions()
        missing = [s for s in sessions if s not in available_sessions]
        if missing:
            logger.warning(
                f"Requested sessions not found in dataset: {', '.join(missing)}"
            )
        return [s for s in sessions if s in available_sessions]
    else:
        sessions = layout.get_sessions()
        if not sessions:
            return [None]  # No sessions in dataset
        return sessions


def run_participant_level(
    layout,
    subjects,
    sessions,
    freesurfer_wrapper,
    bids_provenance,
    freesurfer_dir,
    nidm_dir,
    output_dir,
    skip_nidm,
):
    """
    Run participant level analysis.

    Parameters:
    -----------
    layout : BIDSLayout
        The BIDS layout object
    subjects : list
        List of subject IDs to process
    sessions : list
        List of session IDs to process
    freesurfer_wrapper : FreeSurferWrapper
        FreeSurfer wrapper object
    bids_provenance : BIDSProvenance
        BIDS provenance tracker
    freesurfer_dir : str
        FreeSurfer output directory
    nidm_dir : str
        NIDM output directory
    output_dir : str
        Main output directory
    skip_nidm : bool
        Whether to skip NIDM output generation
    """
    logger.info(f"Running participant level analysis on {len(subjects)} subjects")

    # Track successes and failures
    results = {"success": [], "failure": []}

    for subject in subjects:
        subject_label = subject.replace("sub-", "")

        for session in sessions:
            session_label = session.replace("ses-", "") if session else None

            try:
                # Process this subject/session with FreeSurfer
                logger.info(
                    f"Processing subject {subject_label}{f' session {session_label}' if session_label else ''}"
                )
                success = freesurfer_wrapper.process_subject(
                    subject_label, session_label, layout
                )

                if success:
                    # Create BIDS provenance
                    t1w_images = freesurfer_wrapper._find_t1w_images(
                        layout, subject_label, session_label
                    )
                    t2w_images = freesurfer_wrapper._find_t2w_images(
                        layout, subject_label, session_label
                    )

                    command = f"recon-all -all -subjid {subject_label}{f'_{session_label}' if session_label else ''}"
                    if t2w_images:
                        command += " -T2 -T2pial"
                    if freesurfer_wrapper.fs_options:
                        command += f" {freesurfer_wrapper.fs_options}"

                    bids_provenance.create_subject_provenance(
                        subject_label, session_label, t1w_images, t2w_images, command
                    )

                    # Set up output directory path
                    if session_label:
                        subject_output_dir = os.path.join(
                            output_dir, f"sub-{subject_label}", f"ses-{session_label}"
                        )
                    else:
                        subject_output_dir = os.path.join(
                            output_dir, f"sub-{subject_label}"
                        )

                    # Copy outputs to BIDS-compliant structure
                    copy_freesurfer_outputs(
                        freesurfer_dir, subject_label, session_label, subject_output_dir
                    )

                    # Create NIDM output if requested
                    if not skip_nidm:
                        convert_subject(
                            freesurfer_dir, subject_label, session_label, nidm_dir
                        )

                    results["success"].append(
                        f"{subject_label}{f'_{session_label}' if session_label else ''}"
                    )
                else:
                    results["failure"].append(
                        f"{subject_label}{f'_{session_label}' if session_label else ''}"
                    )
                    logger.error(
                        f"Failed to process subject {subject_label}{f' session {session_label}' if session_label else ''}"
                    )
            except Exception as e:
                results["failure"].append(
                    f"{subject_label}{f'_{session_label}' if session_label else ''}"
                )
                logger.error(
                    f"Error processing subject {subject_label}{f' session {session_label}' if session_label else ''}: {str(e)}"
                )

    # Print summary
    logger.info(
        f"Processing complete. Successfully processed {len(results['success'])} subject/sessions."
    )
    if results["failure"]:
        logger.warning(
            f"Failed to process {len(results['failure'])} subject/sessions: {', '.join(results['failure'])}"
        )

    # Save processing summary
    summary_path = os.path.join(output_dir, "processing_summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {"timestamp": datetime.datetime.now().isoformat(), "results": results},
            f,
            indent=2,
        )
    logger.info(f"Processing summary saved to {summary_path}")


def run_group_level(subjects, bids_provenance, freesurfer_dir, nidm_dir, skip_nidm):
    """
    Run group level analysis.

    Parameters:
    -----------
    subjects : list
        List of subject IDs
    bids_provenance : BIDSProvenance
        BIDS provenance tracker
    freesurfer_dir : str
        FreeSurfer output directory
    nidm_dir : str
        NIDM output directory
    skip_nidm : bool
        Whether to skip NIDM output generation
    """
    logger.info("Running group level analysis")

    # Create group provenance
    bids_provenance.create_group_provenance(subjects)

    # FreeSurfer group analysis would go here
    # TODO: Implement FreeSurfer group-level analyses
    logger.info("No FreeSurfer group-level analyses implemented yet")

    # Create group NIDM
    if not skip_nidm:
        create_group_nidm(subjects, nidm_dir)


def copy_freesurfer_outputs(
    freesurfer_dir, subject_label, session_label, output_subject_dir
):
    """
    Copy relevant FreeSurfer outputs to BIDS-compliant structure.

    Parameters:
    -----------
    freesurfer_dir : str
        FreeSurfer output directory
    subject_label : str
        Subject label
    session_label : str or None
        Session label (or None if no session)
    output_subject_dir : str
        Output directory for this subject
    """
    # Determine source directory based on whether there's a session or not
    if session_label:
        subject_id = f"{subject_label}_{session_label}"
    else:
        subject_id = subject_label

    source_dir = os.path.join(freesurfer_dir, subject_id)

    if not os.path.exists(source_dir):
        logger.warning(f"FreeSurfer output directory {source_dir} not found.")
        return

    # Create output subdirectories
    for subdir in ["anat", "surf", "stats", "label", "mri", "scripts"]:
        os.makedirs(os.path.join(output_subject_dir, subdir), exist_ok=True)

    # File mapping
    file_mapping = {
        # Critical volumes in MRI directory
        "mri/T1.mgz": "anat/T1.mgz",
        "mri/aparc+aseg.mgz": "anat/aparc+aseg.mgz",
        "mri/aparc.a2009s+aseg.mgz": "anat/aparc.a2009s+aseg.mgz",
        "mri/aparc.DKTatlas+aseg.mgz": "anat/aparc.DKTatlas+aseg.mgz",
        "mri/brainmask.mgz": "anat/brainmask.mgz",
        "mri/brain.mgz": "anat/brain.mgz",
        "mri/aseg.mgz": "anat/aseg.mgz",
        "mri/wm.mgz": "anat/wm.mgz",
        "mri/wmparc.mgz": "anat/wmparc.mgz",
        "mri/ribbon.mgz": "anat/ribbon.mgz",
        "mri/entowm.mgz": "anat/entowm.mgz",  # Include specialized files if present
        # Important surfaces
        "surf/lh.white": "surf/lh.white",
        "surf/rh.white": "surf/rh.white",
        "surf/lh.pial": "surf/lh.pial",
        "surf/rh.pial": "surf/rh.pial",
        "surf/lh.inflated": "surf/lh.inflated",
        "surf/rh.inflated": "surf/rh.inflated",
        "surf/lh.sphere.reg": "surf/lh.sphere.reg",
        "surf/rh.sphere.reg": "surf/rh.sphere.reg",
        "surf/lh.thickness": "surf/lh.thickness",
        "surf/rh.thickness": "surf/rh.thickness",
        "surf/lh.area": "surf/lh.area",
        "surf/rh.area": "surf/rh.area",
        "surf/lh.curv": "surf/lh.curv",
        "surf/rh.curv": "surf/rh.curv",
        "surf/lh.sulc": "surf/lh.sulc",
        "surf/rh.sulc": "surf/rh.sulc",
        # Essential stats
        "stats/aseg.stats": "stats/aseg.stats",
        "stats/lh.aparc.stats": "stats/lh.aparc.stats",
        "stats/rh.aparc.stats": "stats/rh.aparc.stats",
        "stats/lh.aparc.a2009s.stats": "stats/lh.aparc.a2009s.stats",
        "stats/rh.aparc.a2009s.stats": "stats/rh.aparc.a2009s.stats",
        "stats/lh.aparc.DKTatlas.stats": "stats/lh.aparc.DKTatlas.stats",
        "stats/rh.aparc.DKTatlas.stats": "stats/rh.aparc.DKTatlas.stats",
        "stats/wmparc.stats": "stats/wmparc.stats",
        "stats/brainvol.stats": "stats/brainvol.stats",
        "stats/entowm.stats": "stats/entowm.stats",  # Include specialized files if present
        # Critical labels/annotations
        "label/lh.aparc.annot": "label/lh.aparc.annot",
        "label/rh.aparc.annot": "label/rh.aparc.annot",
        "label/lh.aparc.a2009s.annot": "label/lh.aparc.a2009s.annot",
        "label/rh.aparc.a2009s.annot": "label/rh.aparc.a2009s.annot",
        "label/lh.aparc.DKTatlas.annot": "label/lh.aparc.DKTatlas.annot",
        "label/rh.aparc.DKTatlas.annot": "label/rh.aparc.DKTatlas.annot",
        # Processing logs and scripts
        "scripts/recon-all.log": "scripts/recon-all.log",
        "scripts/recon-all.done": "scripts/recon-all.done",
        "scripts/recon-all.env": "scripts/recon-all.env",
        "scripts/build-stamp.txt": "scripts/build-stamp.txt",
    }

    # Copy files
    copied_files = []
    missing_files = []

    for src, dst in file_mapping.items():
        src_path = os.path.join(source_dir, src)
        dst_path = os.path.join(output_subject_dir, dst)

        if os.path.exists(src_path):
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            try:
                shutil.copy2(src_path, dst_path)
                copied_files.append(src)
                logger.debug(f"Copied {src_path} to {dst_path}")
            except Exception as e:
                logger.warning(f"Failed to copy {src_path}: {str(e)}")
        else:
            missing_files.append(src)
            logger.debug(f"Source file not found: {src_path}")

    logger.info(
        f"Copied {len(copied_files)} FreeSurfer output files for subject {subject_label}{f' session {session_label}' if session_label else ''}"
    )
    if len(missing_files) > 0:
        logger.debug(f"Missing {len(missing_files)} expected FreeSurfer output files")


if __name__ == "__main__":
    main()
