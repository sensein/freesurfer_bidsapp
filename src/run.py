#!/usr/bin/env python3
import os
import sys
import json
import subprocess
import shutil
import logging
import datetime
import click
import numpy as np
from bids import BIDSLayout
from nipype.interfaces.freesurfer import ReconAll
import rdflib
from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDF, RDFS, XSD

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('bids-freesurfer')

# Define BIDS App
@click.command()
@click.argument('bids_dir', type=click.Path(exists=True, file_okay=False, resolve_path=True))
@click.argument('output_dir', type=click.Path(file_okay=False, resolve_path=True))
@click.argument('analysis_level', type=click.Choice(['participant', 'group']))
@click.option('--participant_label', multiple=True, help='The label(s) of the participant(s) to analyze (without "sub-").')
@click.option('--session_label', multiple=True, help='The label(s) of the session(s) to analyze (without "ses-").')
@click.option('--freesurfer_license', type=click.Path(exists=True, resolve_path=True), help='Path to FreeSurfer license file.')
@click.option('--skip_bids_validator', is_flag=True, help='Skip BIDS validation.')
@click.option('--fs_options', help='Additional options for recon-all, e.g. "--fs_options=-parallel"')
@click.option('--skip_nidm', is_flag=True, help='Skip NIDM output generation.')
def main(bids_dir, output_dir, analysis_level, participant_label, session_label, freesurfer_license, skip_bids_validator, fs_options, skip_nidm):
    """FreeSurfer BIDS App with NIDM Output.
    
    This BIDS App runs FreeSurfer's recon-all pipeline on T1w and optionally T2w images
    from a BIDS dataset and outputs both standard FreeSurfer results and NIDM format results.
    """
    # Print version and check environment
    version = "0.1.0"
    logger.info(f"BIDS-FreeSurfer version: {version}")
    
    # Check FreeSurfer environment
    check_freesurfer_env(freesurfer_license)
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    freesurfer_dir = os.path.join(output_dir, 'freesurfer')
    os.makedirs(freesurfer_dir, exist_ok=True)
    nidm_dir = os.path.join(output_dir, 'nidm')
    os.makedirs(nidm_dir, exist_ok=True)
    
    # Create dataset_description.json
    create_dataset_description(output_dir, version)
    
    # Load BIDS dataset
    layout = BIDSLayout(bids_dir, validate=not skip_bids_validator)
    
    # Get subjects to analyze
    subjects_to_analyze = get_subjects_to_analyze(layout, participant_label)
    sessions_to_analyze = ['ses-' + label for label in session_label] if session_label else []
    
    # Run analysis
    if analysis_level == "participant":
        run_participant_level(layout, subjects_to_analyze, sessions_to_analyze, 
                              freesurfer_dir, nidm_dir, fs_options, skip_nidm)
    else:  # group level
        run_group_level(subjects_to_analyze, freesurfer_dir, nidm_dir, skip_nidm)
    
    logger.info("BIDS-FreeSurfer processing complete.")
    return 0


def check_freesurfer_env(freesurfer_license):
    """Check FreeSurfer environment and license."""
    if 'FREESURFER_HOME' not in os.environ:
        logger.error("FREESURFER_HOME environment variable not set.")
        sys.exit(1)
    
    # Check license file
    license_file = os.environ.get('FS_LICENSE')
    if freesurfer_license:
        license_file = freesurfer_license
        os.environ['FS_LICENSE'] = str(freesurfer_license)
    
    if not license_file or not os.path.exists(license_file):
        logger.error("FreeSurfer license file not found.")
        sys.exit(1)


def create_dataset_description(output_dir, version):
    """Create dataset_description.json for derivatives."""
    freesurfer_version = get_freesurfer_version()
    
    dataset_description = {
        "Name": "FreeSurfer Output",
        "BIDSVersion": "1.8.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": "BIDS-FreeSurfer",
                "Version": version,
                "CodeURL": "https://github.com/yourusername/bids-freesurfer"
            },
            {
                "Name": "FreeSurfer",
                "Version": freesurfer_version,
                "CodeURL": "https://surfer.nmr.mgh.harvard.edu/"
            }
        ]
    }
    
    with open(os.path.join(output_dir, 'dataset_description.json'), 'w') as f:
        json.dump(dataset_description, f, indent=4)


def get_freesurfer_version():
    """Get FreeSurfer version.
    
    Returns the version of FreeSurfer from the installed environment, handling
    different output formats across versions.
    
    Returns:
        str: Version number (e.g., "7.3.2" or "8.0.0")
    """
    try:
        output = subprocess.check_output(['recon-all', '--version'], 
                                        stderr=subprocess.STDOUT).decode().strip()
        
        # Pattern 1: "FreeSurfer X.Y.Z" format
        if output.startswith('FreeSurfer '):
            version_parts = output.split('FreeSurfer ')[1].split('\n')[0].strip()
            return version_parts
        
        # Pattern 2: "recon-all vX.Y.Z (additional info)" format
        if 'recon-all v' in output:
            return output.split('v')[1].split(' ')[0].strip()
        
        # Pattern 3: Just extract the last part of the first line as a fallback
        return output.split('\n')[0].split(' ')[-1].strip()
        
    except Exception as e:
        # Log the error
        logger.warning(f"Could not determine FreeSurfer version: {str(e)}")
        
        # Try to extract version from FREESURFER_HOME
        try:
            fs_home = os.environ.get('FREESURFER_HOME', '')
            version_file = os.path.join(fs_home, 'build-stamp.txt')
            
            if os.path.exists(version_file):
                with open(version_file, 'r') as f:
                    version_content = f.read().strip()
                    # Extract version number using regex
                    import re
                    version_match = re.search(r'v?(\d+\.\d+\.\d+)', version_content)
                    if version_match:
                        return version_match.group(1)
        except:
            pass
            
        # Default to Unknown if all else fails
        return "unknown"


def get_subjects_to_analyze(layout, participant_label):
    """Get list of subjects to analyze."""
    if participant_label:
        return ['sub-' + label for label in participant_label]
    else:
        return ['sub-' + s for s in layout.get_subjects()]


def run_participant_level(layout, subjects, sessions, freesurfer_dir, nidm_dir, fs_options, skip_nidm):
    """Run participant level analysis."""
    logger.info(f"Running participant level analysis on {len(subjects)} subjects")
    
    for subject in subjects:
        subject_label = subject.replace('sub-', '')
        
        # Get sessions for this subject
        subject_sessions = layout.get_sessions(subject=subject_label)
        if sessions:
            subject_sessions = [ses for ses in subject_sessions if f'ses-{ses}' in sessions]
        
        # Handle case where no sessions are present
        if not subject_sessions:
            subject_sessions = [None]
        
        for session in subject_sessions:
            # Process this subject/session
            process_subject_session(layout, subject_label, session, freesurfer_dir, fs_options)
        
        # Create NIDM output if requested
        if not skip_nidm:
            create_nidm_output(subject_label, freesurfer_dir, nidm_dir)


def process_subject_session(layout, subject_label, session_label, freesurfer_dir, fs_options):
    """Process a single subject/session with FreeSurfer recon-all."""
    # Build query for T1w images
    query = {'subject': subject_label, 'datatype': 'anat', 'suffix': 'T1w', 'extension': ['.nii', '.nii.gz']}
    if session_label:
        query['session'] = session_label

    # Get T1w images
    t1w_files = layout.get(return_type='file', **query)
    
    if not t1w_files:
        logger.warning(f"No T1w images found for subject {subject_label}{' session ' + session_label if session_label else ''}")
        return
    
    # Get T2w images (optional)
    query['suffix'] = 'T2w'
    t2w_files = layout.get(return_type='file', **query)
    
    # Set up subject ID and output directory
    if session_label:
        subject_id = f"{subject_label}_{session_label}"
        output_subject_dir = os.path.join(freesurfer_dir, f"sub-{subject_label}", f"ses-{session_label}")
    else:
        subject_id = subject_label
        output_subject_dir = os.path.join(freesurfer_dir, f"sub-{subject_label}")
    
    os.makedirs(output_subject_dir, exist_ok=True)
    
    # Configure recon-all
    reconall = ReconAll()
    reconall.inputs.subject_id = subject_id
    reconall.inputs.T1_files = t1w_files
    if t2w_files:
        reconall.inputs.T2_file = t2w_files[0]  # Use first T2w if multiple exist
    reconall.inputs.directive = 'all'
    reconall.inputs.subjects_dir = freesurfer_dir
    
    # Add additional options if provided
    if fs_options:
        for option in fs_options.split():
            if option.startswith('-'):
                option_name = option.lstrip('-')
                setattr(reconall.inputs, option_name, True)
    
    # Run recon-all
    logger.info(f"Running recon-all for subject {subject_id}")
    try:
        reconall.run()
        logger.info(f"FreeSurfer processing complete for subject {subject_id}")
        
        # Create BIDS-compliant structure
        copy_freesurfer_outputs(freesurfer_dir, subject_id, output_subject_dir)
        
        # Create provenance files
        create_provenance(subject_label, session_label, t1w_files, t2w_files, output_subject_dir)
    
    except Exception as e:
        logger.error(f"Error processing subject {subject_id}: {str(e)}")


def copy_freesurfer_outputs(freesurfer_dir, subject_id, output_subject_dir):
    """Copy relevant FreeSurfer outputs to BIDS-compliant structure."""
    source_dir = os.path.join(freesurfer_dir, subject_id)
    if not os.path.exists(source_dir):
        logger.warning(f"FreeSurfer output directory {source_dir} not found.")
        return
    
    # Create output subdirectories
    for subdir in ['anat', 'surf', 'stats', 'label']:
        os.makedirs(os.path.join(output_subject_dir, subdir), exist_ok=True)
    
    # Essential outputs to copy
    file_mapping = {
        # Critical volumes
        'mri/T1.mgz': 'anat/T1.mgz',
        'mri/aparc+aseg.mgz': 'anat/aparc+aseg.mgz',
        'mri/brainmask.mgz': 'anat/brainmask.mgz',
        'mri/aseg.mgz': 'anat/aseg.mgz',
        
        # Essential surfaces
        'surf/lh.white': 'surf/lh.white',
        'surf/rh.white': 'surf/rh.white',
        'surf/lh.pial': 'surf/lh.pial',
        'surf/rh.pial': 'surf/rh.pial',
        
        # Important stats
        'stats/aseg.stats': 'stats/aseg.stats',
        'stats/lh.aparc.stats': 'stats/lh.aparc.stats',
        'stats/rh.aparc.stats': 'stats/rh.aparc.stats',
    }
    
    for src, dst in file_mapping.items():
        src_path = os.path.join(source_dir, src)
        dst_path = os.path.join(output_subject_dir, dst)
        
        if os.path.exists(src_path):
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy2(src_path, dst_path)


def create_provenance(subject_label, session_label, t1w_files, t2w_files, output_dir):
    """Create BIDS provenance JSON file."""
    provenance = {
        "Sources": [os.path.basename(f) for f in t1w_files + t2w_files],
        "SoftwareVersion": get_freesurfer_version(),
        "CommandLine": f"recon-all -all -subjid {subject_label}",
        "DateProcessed": datetime.datetime.now().isoformat()
    }
    
    with open(os.path.join(output_dir, "provenance.json"), 'w') as f:
        json.dump(provenance, f, indent=4)


def create_nidm_output(subject_label, freesurfer_dir, nidm_dir):
    """Create NIDM output in JSON-LD format."""
    logger.info(f"Creating NIDM output for subject {subject_label}")
    
    subject_nidm_dir = os.path.join(nidm_dir, f"sub-{subject_label}")
    os.makedirs(subject_nidm_dir, exist_ok=True)
    
    # Create RDF graph
    g = Graph()
    
    # Define namespaces
    nidm = Namespace("http://purl.org/nidash/nidm#")
    prov = Namespace("http://www.w3.org/ns/prov#")
    fs = Namespace("http://surfer.nmr.mgh.harvard.edu/fs/terms/")
    
    g.bind("nidm", nidm)
    g.bind("prov", prov)
    g.bind("fs", fs)
    
    # Create basic provenance entities
    subject_uri = URIRef(f"http://example.org/subjects/{subject_label}")
    fs_process = URIRef("http://example.org/processes/freesurfer")
    
    # Add basic provenance
    g.add((subject_uri, RDF.type, prov.Entity))
    g.add((subject_uri, RDFS.label, Literal(subject_label)))
    g.add((fs_process, RDF.type, prov.Activity))
    g.add((fs_process, prov.startedAtTime, Literal(datetime.datetime.now().isoformat(), datatype=XSD.dateTime)))
    
    # Add FreeSurfer stats
    stats_file = os.path.join(freesurfer_dir, f"sub-{subject_label}", "stats", "aseg.stats")
    if os.path.exists(stats_file):
        add_stats_to_graph(g, stats_file, subject_uri, fs)
    
    # Serialize to JSON-LD
    jsonld_file = os.path.join(subject_nidm_dir, "prov.jsonld")
    g.serialize(destination=jsonld_file, format="json-ld", indent=4)
    
    logger.info(f"NIDM output created at {jsonld_file}")


def add_stats_to_graph(graph, stats_file, subject_uri, fs_namespace):
    """Add FreeSurfer stats to RDF graph."""
    try:
        with open(stats_file, 'r') as f:
            lines = f.readlines()
        
        # Process aseg.stats file
        volumes = {}
        for line in lines:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 5:
                structure = parts[4]
                volume = float(parts[3])
                volumes[structure] = volume
        
        # Add to graph
        for structure, volume in volumes.items():
            structure_uri = URIRef(f"{fs_namespace}{structure}")
            graph.add((subject_uri, fs_namespace.hasVolume, structure_uri))
            graph.add((structure_uri, RDF.type, fs_namespace.BrainStructure))
            graph.add((structure_uri, RDFS.label, Literal(structure)))
            graph.add((structure_uri, fs_namespace.volume, Literal(volume, datatype=XSD.float)))
    
    except Exception as e:
        logger.error(f"Error parsing stats file {stats_file}: {e}")


def run_group_level(subjects, freesurfer_dir, nidm_dir, skip_nidm):
    """Run group level analysis."""
    logger.info("Running group level analysis")
    # FreeSurfer group analysis would go here
    
    # Create group NIDM
    if not skip_nidm:
        create_group_nidm(subjects, nidm_dir)


def create_group_nidm(subjects, nidm_dir):
    """Create group-level NIDM output."""
    logger.info("Creating group NIDM output")
    
    # Create a simple group-level NIDM document
    g = Graph()
    
    # Define namespaces
    nidm = Namespace("http://purl.org/nidash/nidm#")
    prov = Namespace("http://www.w3.org/ns/prov#")
    
    g.bind("nidm", nidm)
    g.bind("prov", prov)
    
    # Create basic group structure
    group_uri = URIRef("http://example.org/study/group")
    g.add((group_uri, RDF.type, nidm.Group))
    
    # Add subjects to group
    for subject in subjects:
        subject_uri = URIRef(f"http://example.org/subjects/{subject.replace('sub-', '')}")
        g.add((group_uri, prov.hadMember, subject_uri))
    
    # Serialize to JSON-LD
    jsonld_file = os.path.join(nidm_dir, "group_prov.jsonld")
    g.serialize(destination=jsonld_file, format="json-ld", indent=4)


if __name__ == "__main__":
    sys.exit(main())