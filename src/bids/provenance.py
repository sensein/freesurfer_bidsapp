#!/usr/bin/env python3
"""
BIDS Provenance Implementation

This module implements BIDS provenance for the FreeSurfer BIDS App,
ensuring that all processing steps are fully documented according to
BIDS specifications for reproducibility.
"""

import os
import json
import logging
import datetime
import hashlib
import subprocess
import platform
import sys
from pathlib import Path

from src.utils import get_freesurfer_version

# Configure logging
logger = logging.getLogger('bids-freesurfer.bids.provenance')


class BIDSProvenance:
    """BIDS Provenance Handler for FreeSurfer BIDS App."""
    
    def __init__(self, bids_dir, output_dir, version="0.1.0"):
        """
        Initialize BIDS Provenance handler.
        
        Parameters
        ----------
        bids_dir : str
            Path to BIDS dataset directory
        output_dir : str
            Path to output derivatives directory
        version : str, optional
            Version of the BIDS App
        """
        self.bids_dir = Path(bids_dir)
        self.output_dir = Path(output_dir)
        self.version = version
        
        # Initialize success/failure tracking
        self.processed_subjects = []
        
        try:
            # Create output directory if it doesn't exist
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Create logs directory
            self.logs_dir = self.output_dir / "logs"
            os.makedirs(self.logs_dir, exist_ok=True)
            
            # Create provenance directory
            self.provenance_dir = self.output_dir / "provenance"
            os.makedirs(self.provenance_dir, exist_ok=True)
            
            # Initialize provenance by creating base files
            self.create_dataset_description()
            self.create_readme()
            self._record_environment()
            
            logger.info(f"BIDS provenance initialized for output at {self.output_dir}")
        except Exception as e:
            logger.error(f"Failed to initialize BIDS provenance: {str(e)}")
            raise
    
    def create_dataset_description(self):
        """
        Create BIDS-compliant dataset_description.json for derivatives.
        
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        try:
            logger.info("Creating dataset_description.json")
            
            # Get FreeSurfer version
            fs_version = get_freesurfer_version()
            
            # Determine if running in container
            container_info = self._detect_container()
            
            # Create dataset description
            dataset_description = {
                "Name": "FreeSurfer Output",
                "BIDSVersion": "1.8.0",
                "DatasetType": "derivative",
                "GeneratedBy": [
                    {
                        "Name": "BIDS-FreeSurfer",
                        "Version": self.version,
                        "CodeURL": "https://github.com/yourusername/bids-freesurfer"
                    },
                    {
                        "Name": "FreeSurfer",
                        "Version": fs_version,
                        "CodeURL": "https://surfer.nmr.mgh.harvard.edu/"
                    }
                ],
                "HowToAcknowledge": "Please cite the following papers: Fischl B. (2012). FreeSurfer. NeuroImage, 62(2), 774–781. https://doi.org/10.1016/j.neuroimage.2012.01.021"
            }
            
            # Add container information if available
            if container_info:
                dataset_description["GeneratedBy"][1]["Container"] = container_info
                
            # Include source dataset information
            dataset_description["SourceDatasets"] = [
                {
                    "URL": f"file://{self.bids_dir.absolute()}"
                }
            ]
            
            # Write dataset description to file
            output_file = self.output_dir / "dataset_description.json"
            with open(output_file, 'w') as f:
                json.dump(dataset_description, f, indent=4)
            
            # Also create dataset description in freesurfer and nidm subdirectories
            for subdir in ['freesurfer', 'nidm']:
                subdir_path = self.output_dir / subdir
                os.makedirs(subdir_path, exist_ok=True)
                
                # Customize names based on subdirectory
                subdir_desc = dataset_description.copy()
                if subdir == 'freesurfer':
                    subdir_desc["Name"] = "FreeSurfer Derivatives"
                elif subdir == 'nidm':
                    subdir_desc["Name"] = "NIDM Outputs from FreeSurfer"
                
                # Write dataset description to file
                output_file = subdir_path / "dataset_description.json"
                with open(output_file, 'w') as f:
                    json.dump(subdir_desc, f, indent=4)
            
            logger.info("dataset_description.json created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create dataset_description.json: {str(e)}")
            return False
    
    def _detect_container(self):
        """
        Detect if running in a container environment.
        
        Returns
        -------
        dict or None
            Container information or None if not in a container
        """
        # Check for Docker
        if os.path.exists('/.dockerenv'):
            # Try to get Docker image information
            try:
                container_id = subprocess.check_output(
                    ['cat', '/proc/self/cgroup'], 
                    universal_newlines=True
                ).split('\n')[0].split('/')[-1]
                
                return {
                    "Type": "docker",
                    "ID": container_id
                }
            except:
                return {"Type": "docker"}
        
        # Check for Singularity
        elif os.environ.get('SINGULARITY_NAME'):
            return {
                "Type": "singularity",
                "Name": os.environ.get('SINGULARITY_NAME'),
                "Version": os.environ.get('SINGULARITY_VERSION', '')
            }
        
        return None
    
    def create_readme(self):
        """
        Create README file in BIDS derivatives directory.
        
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        try:
            logger.info("Creating README file")
            
            # Create README content
            readme_content = f"""# FreeSurfer BIDS App Derivatives

This directory contains derivatives from the FreeSurfer BIDS App (version {self.version}).

## Directory Structure

- `freesurfer/`: Contains standard FreeSurfer outputs organized by subject/session
- `nidm/`: Contains NIDM outputs for interoperability
- `provenance/`: Contains detailed provenance information for reproducibility
- `logs/`: Contains processing logs and error reports

## FreeSurfer Version

These derivatives were generated using FreeSurfer version {get_freesurfer_version()}.

## Processing Pipeline

The processing pipeline includes the following steps:
1. T1w image processing with FreeSurfer's recon-all command
2. Automatic segmentation of subcortical structures
3. Reconstruction of white and pial surfaces
4. Cortical parcellation using the Desikan-Killiany and Destrieux atlases
5. Generation of morphometric statistics
6. Conversion to NIDM format for interoperability

## License

The derivative data are provided under the original terms applied to the source dataset.

## References

- Fischl B. (2012). FreeSurfer. NeuroImage, 62(2), 774–781. https://doi.org/10.1016/j.neuroimage.2012.01.021
- Gorgolewski, K. J., Auer, T., Calhoun, V. D., et al. (2016). The brain imaging data structure. Scientific data, 3, 160044. https://doi.org/10.1038/sdata.2016.44

## Processing Date

This dataset was processed on {datetime.datetime.now().strftime('%Y-%m-%d')} using the BIDS-FreeSurfer app.
"""
            
            # Write README to file
            output_file = self.output_dir / "README.md"
            with open(output_file, 'w') as f:
                f.write(readme_content)
            
            logger.info("README.md created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create README.md: {str(e)}")
            return False
    
    def _record_environment(self):
        """
        Record environment information for reproducibility.
        
        Returns
        -------
        dict
            Environment information
        """
        try:
            logger.info("Recording environment information")
            
            # Get environment information
            env_info = self._get_environment_info()
            
            # Add FreeSurfer-specific information
            env_info["FreeSurfer"] = {
                "Version": get_freesurfer_version(),
                "Home": os.environ.get("FREESURFER_HOME", ""),
                "SubjectsDir": os.environ.get("SUBJECTS_DIR", ""),
                "License": os.environ.get("FS_LICENSE", "")
            }
            
            # Add Python packages information
            try:
                import pkg_resources
                env_info["PythonPackages"] = {
                    pkg.key: pkg.version for pkg in pkg_resources.working_set
                }
            except Exception as e:
                logger.warning(f"Failed to get Python packages information: {str(e)}")
            
            # Write environment information to file
            output_file = self.provenance_dir / "environment.json"
            with open(output_file, 'w') as f:
                json.dump(env_info, f, indent=4)
            
            logger.info("Environment information recorded successfully")
            return env_info
            
        except Exception as e:
            logger.error(f"Failed to record environment information: {str(e)}")
            return {"Error": str(e)}
    
    def create_subject_provenance(self, subject_label, session_label=None, t1w_files=None, t2w_files=None, command=None):
        """
        Create provenance information for a subject.
        
        Parameters
        ----------
        subject_label : str
            Subject label (without 'sub-' prefix)
        session_label : str, optional
            Session label (without 'ses-' prefix)
        t1w_files : list, optional
            List of T1w image files used for processing
        t2w_files : list, optional
            List of T2w image files used for processing
        command : str, optional
            Command used for processing
            
        Returns
        -------
        dict
            Provenance information as a dictionary
        """
        # Generate subject identifier
        subject_id = f"{subject_label}{f'_{session_label}' if session_label else ''}"
        
        try:
            logger.info(f"Creating provenance for subject {subject_label}{' session ' + session_label if session_label else ''}")
            
            # Set up output directories
            if session_label:
                subject_dir = self.output_dir / f"sub-{subject_label}" / f"ses-{session_label}"
                prov_dir = self.provenance_dir / f"sub-{subject_label}" / f"ses-{session_label}"
            else:
                subject_dir = self.output_dir / f"sub-{subject_label}"
                prov_dir = self.provenance_dir / f"sub-{subject_label}"
            
            os.makedirs(subject_dir, exist_ok=True)
            os.makedirs(prov_dir, exist_ok=True)
            
            # Generate file hashes if input files are provided
            input_files = []
            if t1w_files:
                for file_path in t1w_files:
                    try:
                        file_hash = self._calculate_file_hash(file_path)
                        input_files.append({
                            "path": str(file_path),
                            "type": "T1w",
                            "hash": file_hash
                        })
                    except Exception as e:
                        logger.warning(f"Failed to hash file {file_path}: {str(e)}")
                        input_files.append({
                            "path": str(file_path),
                            "type": "T1w"
                        })
            
            if t2w_files:
                for file_path in t2w_files:
                    try:
                        file_hash = self._calculate_file_hash(file_path)
                        input_files.append({
                            "path": str(file_path),
                            "type": "T2w",
                            "hash": file_hash
                        })
                    except Exception as e:
                        logger.warning(f"Failed to hash file {file_path}: {str(e)}")
                        input_files.append({
                            "path": str(file_path),
                            "type": "T2w"
                        })
            
            # Create provenance information
            provenance = {
                "GeneratedBy": {
                    "Name": "BIDS-FreeSurfer",
                    "Version": self.version,
                    "CodeURL": "https://github.com/yourusername/bids-freesurfer",
                    "Environment": self._get_environment_info()
                },
                "Subject": subject_label,
                "Session": session_label,
                "InputFiles": input_files,
                "Configuration": {
                    "FreeSurferVersion": get_freesurfer_version(),
                    "CommandLine": command or f"recon-all -all -subjid {subject_id}"
                },
                "Parameters": {
                    "UseT2": bool(t2w_files)
                },
                "DateProcessed": datetime.datetime.now().isoformat()
            }
            
            # Write provenance to file
            output_file = prov_dir / "provenance.json"
            with open(output_file, 'w') as f:
                json.dump(provenance, f, indent=4)
            
            # Add to list of processed subjects
            self.processed_subjects.append(subject_id)
            
            logger.info(f"Provenance created for subject {subject_id}")
            return provenance
            
        except Exception as e:
            logger.error(f"Failed to create provenance for subject {subject_id}: {str(e)}")
            return {"Error": str(e)}
    
    def _calculate_file_hash(self, file_path, algorithm="sha256", buffer_size=65536):
        """
        Calculate hash for a file.
        
        Parameters
        ----------
        file_path : str
            Path to file
        algorithm : str, optional
            Hash algorithm to use (default: sha256)
        buffer_size : int, optional
            Buffer size for reading file
            
        Returns
        -------
        str
            Hexadecimal hash string
        """
        hash_obj = hashlib.new(algorithm)
        
        with open(file_path, 'rb') as f:
            buffer = f.read(buffer_size)
            while buffer:
                hash_obj.update(buffer)
                buffer = f.read(buffer_size)
        
        return hash_obj.hexdigest()
    
    def create_group_provenance(self, subjects, command=None):
        """
        Create provenance information for group analysis.
        
        Parameters
        ----------
        subjects : list
            List of subject labels (without 'sub-' prefix)
        command : str, optional
            Command used for processing
            
        Returns
        -------
        dict
            Provenance information as a dictionary
        """
        try:
            logger.info("Creating group provenance")
            
            # Ensure subjects is a list
            if isinstance(subjects, str):
                subjects = [subjects]
            
            # Create provenance information
            provenance = {
                "GeneratedBy": {
                    "Name": "BIDS-FreeSurfer",
                    "Version": self.version,
                    "CodeURL": "https://github.com/yourusername/bids-freesurfer",
                    "Environment": self._get_environment_info()
                },
                "Subjects": [f"sub-{subject}" for subject in subjects],
                "Configuration": {
                    "FreeSurferVersion": get_freesurfer_version(),
                    "CommandLine": command or "FreeSurfer group analysis"
                },
                "Parameters": {},
                "DateProcessed": datetime.datetime.now().isoformat()
            }
            
            # Write provenance to file
            output_file = self.provenance_dir / "group_provenance.json"
            with open(output_file, 'w') as f:
                json.dump(provenance, f, indent=4)
            
            logger.info("Group provenance created successfully")
            return provenance
            
        except Exception as e:
            logger.error(f"Failed to create group provenance: {str(e)}")
            return {"Error": str(e)}
    
    def _get_environment_info(self):
        """
        Get information about the execution environment.
        
        Returns
        -------
        dict
            Environment information
        """
        env_info = {
            "Platform": platform.platform(),
            "Python": platform.python_version(),
            "Timestamp": datetime.datetime.now().isoformat()
        }
        
        # Get hostname if available
        try:
            env_info["Hostname"] = platform.node()
        except:
            pass
        
        # Add processor information
        env_info["CPU"] = {
            "Architecture": platform.machine(),
            "Processor": platform.processor()
        }
        
        # Add OS information
        env_info["OS"] = {
            "System": platform.system(),
            "Release": platform.release(),
            "Version": platform.version()
        }
        
        # Check if running in container
        if os.path.exists('/.dockerenv'):
            env_info["Container"] = "Docker"
        elif os.environ.get('SINGULARITY_NAME'):
            env_info["Container"] = "Singularity"
            env_info["ContainerName"] = os.environ.get('SINGULARITY_NAME')
            env_info["ContainerVersion"] = os.environ.get('SINGULARITY_VERSION', '')
        
        return env_info
    
    def get_processing_summary(self):
        """
        Get summary of processed subjects.
        
        Returns
        -------
        dict
            Summary of processed subjects
        """
        return {
            "total_subjects": len(self.processed_subjects),
            "subjects": self.processed_subjects,
            "timestamp": datetime.datetime.now().isoformat()
        }


def create_bids_provenance(bids_dir, output_dir, version="0.1.0"):
    """
    Create BIDS provenance handler.
    
    Parameters
    ----------
    bids_dir : str
        Path to BIDS dataset directory
    output_dir : str
        Path to output derivatives directory
    version : str, optional
        Version of the BIDS App
        
    Returns
    -------
    BIDSProvenance
        BIDSProvenance instance
    """
    try:
        provenance = BIDSProvenance(bids_dir, output_dir, version)
        logger.info(f"BIDS provenance handler created for {bids_dir} -> {output_dir}")
        return provenance
    except Exception as e:
        logger.error(f"Failed to create BIDS provenance handler: {str(e)}")
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create BIDS provenance")
    parser.add_argument("bids_dir", help="Path to BIDS dataset directory")
    parser.add_argument("output_dir", help="Path to output derivatives directory")
    parser.add_argument("--version", default="0.1.0", help="Version of the BIDS App")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Create BIDS provenance
        provenance = create_bids_provenance(args.bids_dir, args.output_dir, args.version)
        logger.info("BIDS provenance setup completed successfully")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Failed to setup BIDS provenance: {str(e)}")
        sys.exit(1)