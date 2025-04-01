#!/usr/bin/env python3
"""
FreeSurfer Wrapper for BIDS App

This module provides a wrapper around FreeSurfer's recon-all command
to process BIDS datasets and generate FreeSurfer derivatives in a
BIDS-compliant structure.
"""

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
import time

# Standardize on click throughout the application
import click

from bids import BIDSLayout
from src.utils import get_freesurfer_version

# Configure logging
logger = logging.getLogger("bids-freesurfer.wrapper")


class FreeSurferWrapper:
    """Wrapper for FreeSurfer's recon-all command."""

    def __init__(self, bids_dir, output_dir, freesurfer_license=None, fs_options=None, 
                 parallel=False, num_threads=None, memory_limit=None):
        """
        Initialize FreeSurfer wrapper.

        Parameters
        ----------
        bids_dir : str
            Path to BIDS dataset directory
        output_dir : str
            Path to output derivatives directory
        freesurfer_license : str, optional
            Path to FreeSurfer license file
        fs_options : str, optional
            Additional options to pass to recon-all
        parallel : bool, optional
            Whether to enable parallel processing
        num_threads : int, optional
            Number of threads to use for parallel processing
        memory_limit : int, optional
            Memory limit in GB
        """
        self.bids_dir = Path(bids_dir)
        self.output_dir = Path(output_dir)
        self.freesurfer_dir = self.output_dir / "freesurfer"
        self.freesurfer_license = freesurfer_license
        self.fs_options = fs_options or ""
        self.parallel = parallel
        self.num_threads = num_threads
        self.memory_limit = memory_limit

        # Track processing results
        self.results = {"success": [], "failure": [], "skipped": []}
        self.temp_files = []

        # Ensure output directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.freesurfer_dir.mkdir(parents=True, exist_ok=True)

        # Setup FreeSurfer environment
        try:
            self._setup_freesurfer_env()
            # Log the FreeSurfer version
            logger.info(f"Using FreeSurfer version: {get_freesurfer_version()}")
        except Exception as e:
            logger.error(f"Failed to set up FreeSurfer environment: {str(e)}")
            raise

    def _setup_freesurfer_env(self):
        """
        Setup FreeSurfer environment and license.

        Raises
        ------
        EnvironmentError
            If FREESURFER_HOME is not set
        FileNotFoundError
            If FreeSurfer license is not found
        """
        # Check FREESURFER_HOME environment variable
        if "FREESURFER_HOME" not in os.environ:
            logger.error("FREESURFER_HOME environment variable not set")
            raise EnvironmentError("FREESURFER_HOME environment variable not set")

        # Set SUBJECTS_DIR to our output freesurfer directory
        os.environ["SUBJECTS_DIR"] = str(self.freesurfer_dir)

        # Check license
        if self.freesurfer_license:
            if os.path.exists(self.freesurfer_license):
                os.environ["FS_LICENSE"] = str(self.freesurfer_license)
                logger.info(
                    f"Using provided FreeSurfer license: {self.freesurfer_license}"
                )
            else:
                logger.error(
                    f"FreeSurfer license not found at specified path: {self.freesurfer_license}"
                )
                raise FileNotFoundError(
                    f"FreeSurfer license not found at: {self.freesurfer_license}"
                )
        else:
            # Try standard locations
            license_locations = [
                "/license.txt",  # Docker mount location
                os.path.join(os.environ.get("FREESURFER_HOME", ""), "license.txt"),
                os.path.expanduser("~/.freesurfer.txt"),
            ]

            for loc in license_locations:
                if os.path.exists(loc):
                    logger.info(f"Using FreeSurfer license from {loc}")
                    os.environ["FS_LICENSE"] = loc
                    break
            else:
                logger.error("FreeSurfer license not found in standard locations")
                raise FileNotFoundError(
                    "FreeSurfer license not found. Please specify with --freesurfer_license"
                )

    def _create_recon_all_command(self, subject_id, t1w_images, t2w_images=None):
        """
        Create FreeSurfer recon-all command.

        Parameters
        ----------
        subject_id : str
            Subject ID for FreeSurfer
        t1w_images : list
            List of T1w image paths
        t2w_images : list, optional
            List of T2w image paths

        Returns
        -------
        list
            Command list for subprocess
        """
        cmd = ["recon-all", "-subjid", subject_id, "-all"]

        # Add T1w images
        for t1w in t1w_images:
            cmd.extend(["-i", t1w])

        # Add T2w image if available
        if t2w_images and len(t2w_images) > 0:
            cmd.extend(["-T2", t2w_images[0]])
            cmd.extend(["-T2pial"])

        # Add parallel processing options
        if self.parallel:
            cmd.extend(["-parallel"])
            if self.num_threads:
                cmd.extend(["-openmp", str(self.num_threads)])

        # Add memory limit if specified
        if self.memory_limit:
            cmd.extend(["-max-threads", str(self.memory_limit)])

        # Add additional FreeSurfer options
        if self.fs_options:
            if isinstance(self.fs_options, str):
                fs_options_list = self.fs_options.split()
                cmd.extend(fs_options_list)
            elif isinstance(self.fs_options, list):
                cmd.extend(self.fs_options)

        return cmd

    def _monitor_progress(self, subject_id):
        """
        Monitor recon-all progress.

        Parameters
        ----------
        subject_id : str
            Subject ID being processed
        """
        log_file = self.freesurfer_dir / subject_id / "scripts" / "recon-all.log"
        if not log_file.exists():
            return

        try:
            with open(log_file) as f:
                # Seek to end of file
                f.seek(0, 2)
                while True:
                    line = f.readline()
                    if line:
                        if "ERROR" in line:
                            logger.error(f"FreeSurfer error: {line.strip()}")
                        elif "WARNING" in line:
                            logger.warning(f"FreeSurfer warning: {line.strip()}")
                        else:
                            logger.info(f"FreeSurfer progress: {line.strip()}")
                    time.sleep(1)
        except Exception as e:
            logger.warning(f"Error monitoring progress: {str(e)}")

    def cleanup(self):
        """Clean up temporary files and directories."""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                logger.warning(f"Error removing temporary file {temp_file}: {str(e)}")

    def _organize_bids_output(self, subject_id, subject_label, session_label=None):
        """
        Organize FreeSurfer outputs in BIDS-compliant format.

        Parameters
        ----------
        subject_id : str
            FreeSurfer subject ID
        subject_label : str
            BIDS subject label
        session_label : str, optional
            BIDS session label
        """
        # Define BIDS-compliant paths
        if session_label:
            bids_subject_dir = self.output_dir / f"sub-{subject_label}" / f"ses-{session_label}"
        else:
            bids_subject_dir = self.output_dir / f"sub-{subject_label}"

        # Create BIDS-compliant directories
        anat_dir = bids_subject_dir / "anat"
        anat_dir.mkdir(parents=True, exist_ok=True)

        # Copy FreeSurfer outputs to BIDS structure
        fs_subject_dir = self.freesurfer_dir / subject_id
        if not fs_subject_dir.exists():
            logger.error(f"FreeSurfer subject directory not found: {fs_subject_dir}")
            return

        # Copy T1w brain
        if (fs_subject_dir / "mri" / "brain.mgz").exists():
            shutil.copy2(
                fs_subject_dir / "mri" / "brain.mgz",
                anat_dir / f"sub-{subject_label}{'_ses-' + session_label if session_label else ''}_desc-brain_T1w.nii.gz"
            )

        # Copy aparc+aseg
        if (fs_subject_dir / "mri" / "aparc.DKTatlas+aseg.mgz").exists():
            shutil.copy2(
                fs_subject_dir / "mri" / "aparc.DKTatlas+aseg.mgz",
                anat_dir / f"sub-{subject_label}{'_ses-' + session_label if session_label else ''}_desc-aparcaseg_dseg.nii.gz"
            )

        # Copy wmparc
        if (fs_subject_dir / "mri" / "wmparc.mgz").exists():
            shutil.copy2(
                fs_subject_dir / "mri" / "wmparc.mgz",
                anat_dir / f"sub-{subject_label}{'_ses-' + session_label if session_label else ''}_desc-wmparc_dseg.nii.gz"
            )

        # Copy stats files
        stats_dir = bids_subject_dir / "stats"
        stats_dir.mkdir(parents=True, exist_ok=True)
        
        if (fs_subject_dir / "stats").exists():
            for stat_file in (fs_subject_dir / "stats").glob("*.stats"):
                shutil.copy2(
                    stat_file,
                    stats_dir / f"sub-{subject_label}{'_ses-' + session_label if session_label else ''}_{stat_file.name}"
                )

        # Create dataset_description.json if it doesn't exist
        dataset_desc = self.output_dir / "dataset_description.json"
        if not dataset_desc.exists():
            with open(dataset_desc, "w") as f:
                json.dump({
                    "Name": "FreeSurfer Derivatives",
                    "BIDSVersion": "1.4.0",
                    "DatasetType": "derivative",
                    "GeneratedBy": [{
                        "Name": "FreeSurfer",
                        "Version": get_freesurfer_version(),
                        "Description": "FreeSurfer cortical reconstruction and parcellation"
                    }]
                }, f, indent=2)

        # Create README
        readme = self.output_dir / "README"
        if not readme.exists():
            with open(readme, "w") as f:
                f.write("""FreeSurfer Derivatives
====================

This directory contains FreeSurfer derivatives organized according to the BIDS specification.
The following files are included:
- Brain-extracted T1w images
- Cortical parcellation (aparc+aseg)
- White matter parcellation (wmparc)
- Statistical measurements in the stats directory

For more information about FreeSurfer, visit: http://surfer.nmr.mgh.harvard.edu/
""")

    def process_subject(self, subject_label, session_label=None, layout=None):
        """
        Process a single subject with FreeSurfer.

        Parameters
        ----------
        subject_label : str
            Subject label without 'sub-' prefix
        session_label : str, optional
            Session label without 'ses-' prefix
        layout : BIDSLayout, optional
            BIDS layout object (if not provided, one will be created)

        Returns
        -------
        bool
            True if processing was successful, False otherwise
        """
        subject_key = f"{subject_label}{f'_{session_label}' if session_label else ''}"
        logger.info(
            f"Processing subject sub-{subject_label}{' session ses-' + session_label if session_label else ''}"
        )

        try:
            # Create or use existing BIDS layout
            if layout is None:
                try:
                    layout = BIDSLayout(self.bids_dir)
                except Exception as e:
                    logger.error(f"Failed to create BIDS layout: {str(e)}")
                    self.results["failure"].append(subject_key)
                    return False

            # Find T1w images
            t1w_images = self._find_t1w_images(layout, subject_label, session_label)
            if not t1w_images:
                logger.error(
                    f"No T1w images found for sub-{subject_label}{' ses-' + session_label if session_label else ''}"
                )
                self.results["skipped"].append(subject_key)
                return False

            # Find T2w images (optional)
            t2w_images = self._find_t2w_images(layout, subject_label, session_label)
            if t2w_images:
                logger.info(
                    f"Found {len(t2w_images)} T2w images for sub-{subject_label}{' ses-' + session_label if session_label else ''}"
                )

            # Set up subject ID and output directory
            if session_label:
                subject_id = f"{subject_label}_{session_label}"
            else:
                subject_id = subject_label

            # Check if subject already processed
            subject_dir = self.freesurfer_dir / subject_id
            if (
                subject_dir.exists()
                and (subject_dir / "scripts" / "recon-all.done").exists()
            ):
                logger.info(f"Subject {subject_id} already processed. Skipping...")
                self.results["skipped"].append(subject_key)
                return True

            # Generate recon-all command
            cmd = self._create_recon_all_command(subject_id, t1w_images, t2w_images)

            # Start progress monitoring in a separate thread
            import threading
            monitor_thread = threading.Thread(
                target=self._monitor_progress,
                args=(subject_id,),
                daemon=True
            )
            monitor_thread.start()

            # Run recon-all
            logger.info(f"Running command: {' '.join(cmd)}")
            process = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Log output for debugging
            logger.debug(f"Command output: {process.stdout}")
            if process.stderr:
                logger.debug(f"Command stderr: {process.stderr}")

            # Organize outputs in BIDS format
            self._organize_bids_output(subject_id, subject_label, session_label)

            # Record success
            self.results["success"].append(subject_key)
            logger.info(f"Successfully processed subject {subject_id}")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Error running recon-all for {subject_key}: {str(e)}")
            if e.stderr:
                logger.error(f"Error details: {e.stderr}")

            # Create error log file
            error_dir = self.output_dir / "logs"
            error_dir.mkdir(parents=True, exist_ok=True)
            error_file = error_dir / f"{subject_key}_error.log"

            with open(error_file, "w") as f:
                f.write(f"Error running recon-all for {subject_key}\n")
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"Error: {str(e)}\n")
                if e.stderr:
                    f.write(f"Error details: {e.stderr}\n")

            logger.info(f"Error details saved to {error_file}")
            self.results["failure"].append(subject_key)
            return False

        except Exception as e:
            logger.error(f"Unexpected error processing {subject_key}: {str(e)}")
            self.results["failure"].append(subject_key)
            return False

        finally:
            # Clean up temporary files
            self.cleanup()

    def _find_t1w_images(self, layout, subject_label, session_label=None):
        """
        Find T1w images for a subject.

        Parameters
        ----------
        layout : BIDSLayout
            BIDS layout object
        subject_label : str
            Subject label without 'sub-' prefix
        session_label : str, optional
            Session label without 'ses-' prefix

        Returns
        -------
        list
            List of paths to T1w images
        """
        query = {
            "subject": subject_label,
            "datatype": "anat",
            "suffix": "T1w",
            "extension": [".nii", ".nii.gz"],
        }

        if session_label:
            query["session"] = session_label

        files = layout.get(return_type="file", **query)
        logger.info(
            f"Found {len(files)} T1w images for sub-{subject_label}{' ses-' + session_label if session_label else ''}"
        )
        for f in files:
            logger.debug(f"T1w image: {f}")
        return files

    def _find_t2w_images(self, layout, subject_label, session_label=None):
        """
        Find T2w images for a subject.

        Parameters
        ----------
        layout : BIDSLayout
            BIDS layout object
        subject_label : str
            Subject label without 'sub-' prefix
        session_label : str, optional
            Session label without 'ses-' prefix

        Returns
        -------
        list
            List of paths to T2w images
        """
        query = {
            "subject": subject_label,
            "datatype": "anat",
            "suffix": "T2w",
            "extension": [".nii", ".nii.gz"],
        }

        if session_label:
            query["session"] = session_label

        files = layout.get(return_type="file", **query)
        for f in files:
            logger.debug(f"T2w image: {f}")
        return files

    def get_processing_summary(self):
        """
        Get summary of processing results.

        Returns
        -------
        dict
            Dictionary with processing statistics
        """
        return {
            "total": len(self.results["success"])
            + len(self.results["failure"])
            + len(self.results["skipped"]),
            "success": len(self.results["success"]),
            "failure": len(self.results["failure"]),
            "skipped": len(self.results["skipped"]),
            "success_list": self.results["success"],
            "failure_list": self.results["failure"],
            "skipped_list": self.results["skipped"],
        }

    def save_processing_summary(self, output_path=None):
        """
        Save processing summary to JSON file.

        Parameters
        ----------
        output_path : str, optional
            Path to save summary JSON (default: {output_dir}/processing_summary.json)

        Returns
        -------
        str
            Path to saved summary file
        """
        if output_path is None:
            output_path = os.path.join(self.output_dir, "processing_summary.json")

        summary = self.get_processing_summary()
        summary["timestamp"] = datetime.now().isoformat()
        summary["freesurfer_version"] = get_freesurfer_version()

        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Processing summary saved to {output_path}")
        return output_path


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
    type=click.Path(exists=True, resolve_path=True),
    help="Path to FreeSurfer license file.",
)
@click.option("--skip_bids_validator", is_flag=True, help="Skip BIDS validation.")
@click.option(
    "--fs_options",
    help='Additional options for recon-all, e.g. "--fs_options=-parallel"',
)
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
    verbose,
):
    """
    FreeSurfer wrapper for BIDS datasets.

    This command runs FreeSurfer's recon-all on BIDS formatted T1w and T2w data.
    """
    # Set up logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    try:
        # Initialize wrapper
        wrapper = FreeSurferWrapper(
            bids_dir, output_dir, freesurfer_license, fs_options
        )

        # Load BIDS dataset
        logger.info(f"Loading BIDS dataset from {bids_dir}")
        layout = BIDSLayout(bids_dir, validate=not skip_bids_validator)

        # Get subjects to analyze
        if participant_label:
            subjects = list(participant_label)
            logger.info(f"Processing specific subjects: {', '.join(subjects)}")
        else:
            subjects = layout.get_subjects()
            logger.info(f"Processing all {len(subjects)} subjects in dataset")

        # Get sessions to analyze
        if session_label:
            sessions = list(session_label)
            logger.info(f"Processing specific sessions: {', '.join(sessions)}")
        else:
            sessions = layout.get_sessions()
            if not sessions:
                # If no sessions found, use None for session-less datasets
                logger.info("No sessions found in dataset, using session-less mode")
                sessions = [None]
            else:
                logger.info(f"Processing all {len(sessions)} sessions in dataset")

        # Run participant level analysis
        if analysis_level == "participant":
            for subject in subjects:
                for session in sessions:
                    wrapper.process_subject(subject, session, layout)

            # Save summary
            wrapper.save_processing_summary()

            # Print summary
            summary = wrapper.get_processing_summary()
            logger.info("================================")
            logger.info("Processing complete!")
            logger.info(f"Total subjects/sessions processed: {summary['total']}")
            logger.info(f"Successfully processed: {summary['success']}")
            logger.info(f"Failed to process: {summary['failure']}")
            logger.info(
                f"Skipped (already processed or missing data): {summary['skipped']}"
            )
            logger.info("================================")

        # Run group level analysis
        elif analysis_level == "group":
            logger.info("Group level analysis not implemented yet")

        logger.info("FreeSurfer processing complete")
        return 0

    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
