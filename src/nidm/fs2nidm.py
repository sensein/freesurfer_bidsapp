#!/usr/bin/env python3
"""
FreeSurfer to NIDM Converter

This module converts FreeSurfer outputs to NIDM (Neuroimaging Data Model) format
using JSON-LD serialization to enable interoperability and standardized representation
of neuroimaging results.
"""

import os
import json
import logging
import datetime
import re
import sys
from pathlib import Path
from uuid import uuid4
from rdflib import Graph, URIRef, Literal, Namespace, BNode
from rdflib.namespace import RDF, RDFS, XSD, PROV, DCTERMS

# Import utility functions
from src.nidm.utils import parse_fs_stats_file, safe_id, load_fs_mapping, map_fs_term

# Configure logging
logger = logging.getLogger('bids-freesurfer.fs2nidm')

# Define namespaces
NIDM = Namespace("http://purl.org/nidash/nidm#")
NIIRI = Namespace("http://iri.nidash.org/")
FS = Namespace("http://surfer.nmr.mgh.harvard.edu/fs/terms/")
BIDS = Namespace("http://bids.neuroimaging.io/terms/")


class FreeSurferToNIDM:
    """Convert FreeSurfer outputs to NIDM format."""
    
    def __init__(self, freesurfer_dir, subject_label, session_label=None, output_dir=None):
        """
        Initialize the converter.
        
        Parameters
        ----------
        freesurfer_dir : str
            Path to FreeSurfer derivatives directory
        subject_label : str
            Subject label (without 'sub-' prefix)
        session_label : str, optional
            Session label (without 'ses-' prefix)
        output_dir : str, optional
            Output directory for NIDM files (default is freesurfer_dir/../nidm)
        """
        # Set up paths
        self.freesurfer_dir = Path(freesurfer_dir)
        self.subject_label = subject_label
        self.session_label = session_label
        
        # Validate directories
        if not self.freesurfer_dir.exists():
            raise ValueError(f"FreeSurfer directory does not exist: {self.freesurfer_dir}")
        
        # Set up subject ID and paths
        if session_label:
            self.subject_id = f"{subject_label}_{session_label}"
            self.fs_subject_dir = self.freesurfer_dir / f"sub-{subject_label}" / f"ses-{session_label}"
        else:
            self.subject_id = subject_label
            self.fs_subject_dir = self.freesurfer_dir / f"sub-{subject_label}"
        
        # Validate subject directory
        if not self.fs_subject_dir.exists():
            raise ValueError(f"Subject directory does not exist: {self.fs_subject_dir}")
        
        # Set up output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.freesurfer_dir.parent / "nidm"
        
        # Create output directory
        if session_label:
            self.nidm_subject_dir = self.output_dir / f"sub-{subject_label}" / f"ses-{session_label}"
        else:
            self.nidm_subject_dir = self.output_dir / f"sub-{subject_label}"
        
        os.makedirs(self.nidm_subject_dir, exist_ok=True)
        
        # Initialize RDF graph
        self.graph = Graph()
        self._bind_namespaces()
        
        # Create URIs using simpler approach
        self.subject_uri = NIIRI[f"subject-{subject_label}"]
        self.fs_process = NIIRI[f"fs-process-{subject_label}"]
        self.fs_software = URIRef("http://surfer.nmr.mgh.harvard.edu/")
        
        # Track processing
        self.processed_files = []
        
        # Load term mappings
        self.fs_mappings = load_fs_mapping()
    
    def _bind_namespaces(self):
        """Bind namespaces to the RDF graph."""
        self.graph.bind("nidm", NIDM)
        self.graph.bind("niiri", NIIRI)
        self.graph.bind("prov", PROV)
        self.graph.bind("fs", FS)
        self.graph.bind("bids", BIDS)
        self.graph.bind("dcterms", DCTERMS)
        self.graph.bind("rdfs", RDFS)
        self.graph.bind("xsd", XSD)
    
    def convert(self):
        """
        Convert FreeSurfer outputs to NIDM format.
        
        Returns
        -------
        str
            Path to the output NIDM file
        """
        try:
            logger.info(f"Converting FreeSurfer outputs for {self.subject_id} to NIDM")
            
            # Add basic provenance information
            self._add_basic_provenance()
            
            # Process FreeSurfer stats files
            self._process_stats_files()
            
            # Serialize to JSON-LD and Turtle
            output_file = self.nidm_subject_dir / "prov.jsonld"
            self.graph.serialize(destination=str(output_file), format="json-ld", indent=4)
            
            turtle_file = self.nidm_subject_dir / "prov.ttl"
            self.graph.serialize(destination=str(turtle_file), format="turtle")
            
            logger.info(f"NIDM output created at {output_file}")
            logger.info(f"Processed {len(self.processed_files)} FreeSurfer stats files")
            
            return str(output_file)
        
        except Exception as e:
            logger.error(f"Error converting FreeSurfer outputs: {str(e)}")
            raise
    
    def _add_basic_provenance(self):
        """Add basic provenance information to the RDF graph."""
        # Add subject information
        self.graph.add((self.subject_uri, RDF.type, PROV.Entity))
        self.graph.add((self.subject_uri, RDF.type, NIDM.Subject))
        self.graph.add((self.subject_uri, RDFS.label, Literal(f"Subject {self.subject_label}")))
        self.graph.add((self.subject_uri, BIDS.subject_id, Literal(self.subject_label)))
        
        if self.session_label:
            self.graph.add((self.subject_uri, BIDS.session_id, Literal(self.session_label)))
        
        # Add FreeSurfer process information
        self.graph.add((self.fs_process, RDF.type, PROV.Activity))
        self.graph.add((self.fs_process, RDF.type, NIDM.FreeSurferAnalysis))
        self.graph.add((self.fs_process, PROV.used, self.subject_uri))
        
        # Add FreeSurfer software information
        self.graph.add((self.fs_software, RDF.type, PROV.SoftwareAgent))
        self.graph.add((self.fs_software, RDFS.label, Literal("FreeSurfer")))
        self.graph.add((self.fs_process, PROV.wasAssociatedWith, self.fs_software))
        
        # Try to get version information
        version = self._get_fs_version()
        self.graph.add((self.fs_software, PROV.version, Literal(version or "unknown")))
        
        # Add timestamp
        current_time = datetime.datetime.now().isoformat()
        self.graph.add((self.fs_process, PROV.startedAtTime, Literal(current_time, datatype=XSD.dateTime)))
    
    def _get_fs_version(self):
        """Get FreeSurfer version from logs or other sources."""
        # Try to find version in recon-all.log
        log_file = self.fs_subject_dir / "scripts" / "recon-all.log"
        if log_file.exists():
            try:
                with open(log_file) as f:
                    content = f.read()
                
                # Look for version in log
                match = re.search(r'freesurfer-darwin-darwin.+(\d+\.\d+\.\d+)', content)
                if match:
                    return match.group(1)
                
                match = re.search(r'freesurfer-linux.+(\d+\.\d+\.\d+)', content)
                if match:
                    return match.group(1)
                    
                match = re.search(r'recon-all.+v(\d+\.\d+\.\d+)', content)
                if match:
                    return match.group(1)
            except:
                pass
        
        # Try provenance file
        for prov_file in [
            self.fs_subject_dir / "provenance.json",
            self.freesurfer_dir.parent / "provenance" / f"sub-{self.subject_label}" / "provenance.json"
        ]:
            if prov_file.exists():
                try:
                    with open(prov_file) as f:
                        data = json.load(f)
                    
                    # Check different formats
                    if "Configuration" in data and "FreeSurferVersion" in data["Configuration"]:
                        return data["Configuration"]["FreeSurferVersion"]
                    
                    if "SoftwareVersion" in data:
                        return data["SoftwareVersion"]
                except:
                    pass
        
        return None
    
    def _process_stats_files(self):
        """Process FreeSurfer stats files and add to RDF graph."""
        stats_dir = self.fs_subject_dir / "stats"
        if not stats_dir.exists():
            logger.warning(f"Stats directory not found: {stats_dir}")
            return
            
        # Process aseg.stats
        aseg_file = stats_dir / "aseg.stats"
        if aseg_file.exists():
            self._process_stats_file(aseg_file, "aseg")
            self.processed_files.append("aseg.stats")
        
        # Process parcellation files
        for filename in stats_dir.glob("*.aparc*.stats"):
            name = filename.name
            parts = name.split(".")
            
            if len(parts) == 3:  # Standard aparc (lh.aparc.stats)
                hemisphere = parts[0]  # lh or rh
                atlas = "aparc"
            else:  # Extended atlas (lh.aparc.a2009s.stats)
                hemisphere = parts[0]  # lh or rh
                atlas = parts[2]  # a2009s, DKTatlas, etc.
            
            self._process_stats_file(filename, "aparc", hemisphere=hemisphere, atlas=atlas)
            self.processed_files.append(name)
        
        # Process other stats files
        for special_file in ["wmparc.stats", "brainvol.stats"]:
            file_path = stats_dir / special_file
            if file_path.exists():
                self._process_stats_file(file_path, special_file.split(".")[0])
                self.processed_files.append(special_file)
    
    def _process_stats_file(self, stats_file, stats_type, hemisphere=None, atlas=None):
        """
        Process a FreeSurfer stats file.
        
        Parameters
        ----------
        stats_file : Path
            Path to stats file
        stats_type : str
            Type of stats file ('aseg', 'aparc', etc.)
        hemisphere : str, optional
            Hemisphere ('lh' or 'rh') for cortical stats
        atlas : str, optional
            Atlas name for cortical stats
        """
        logger.info(f"Processing {stats_file.name}")
        
        # Parse the stats file
        data = parse_fs_stats_file(stats_file)
        if not data:
            logger.warning(f"Failed to parse {stats_file}")
            return
        
        # Create a container for the stats
        container_id = f"{stats_type}-{self.subject_label}"
        if hemisphere:
            container_id = f"{hemisphere}-{atlas}-{self.subject_label}"
        
        container_uri = NIIRI[container_id]
        
        # Add container metadata
        if stats_type == "aseg":
            self.graph.add((container_uri, RDF.type, FS.SegmentationStatistics))
            self.graph.add((container_uri, RDFS.label, Literal("Subcortical Segmentation Statistics")))
            self.graph.add((self.subject_uri, FS.hasSegmentationStatistics, container_uri))
        elif stats_type == "aparc":
            self.graph.add((container_uri, RDF.type, FS.CorticalParcellationStatistics))
            self.graph.add((container_uri, RDFS.label, Literal(f"{hemisphere.upper()} {atlas} Cortical Parcellation Statistics")))
            self.graph.add((container_uri, FS.hemisphere, Literal(hemisphere)))
            self.graph.add((container_uri, FS.atlas, Literal(atlas)))
            self.graph.add((self.subject_uri, FS.hasCorticalParcellationStatistics, container_uri))
        else:
            self.graph.add((container_uri, RDF.type, FS.SpecializedStatistics))
            self.graph.add((container_uri, RDFS.label, Literal(f"{stats_type.capitalize()} Statistics")))
            self.graph.add((self.subject_uri, FS.hasSpecializedStatistics, container_uri))
        
        # Add file info
        self.graph.add((container_uri, PROV.wasGeneratedBy, self.fs_process))
        self.graph.add((container_uri, PROV.atLocation, Literal(str(stats_file))))
        
        # Add global measures
        for name, value in data['global_measures'].items():
            measure_uri = NIIRI[f"{container_id}-measure-{safe_id(name)}"]
            self.graph.add((container_uri, FS[safe_id(name)], measure_uri))
            self.graph.add((measure_uri, RDF.type, FS.Measurement))
            self.graph.add((measure_uri, RDFS.label, Literal(name)))
            self.graph.add((measure_uri, FS.value, Literal(value, datatype=XSD.float)))
            
            # Add standard term mapping if available
            std_term = map_fs_term(name, self.fs_mappings, "measure")
            if std_term:
                self.graph.add((measure_uri, RDFS.seeAlso, URIRef(std_term)))
        
        # Add structures/regions
        for i, struct in enumerate(data['structures']):
            if 'StructName' not in struct:
                continue
                
            name = struct['StructName']
            struct_uri = NIIRI[f"{container_id}-region-{i}-{safe_id(name)}"]
            
            # Add appropriate type based on stats_type
            if stats_type == "aseg":
                self.graph.add((container_uri, FS.hasStructure, struct_uri))
                self.graph.add((struct_uri, RDF.type, FS.BrainStructure))
            elif stats_type == "aparc":
                self.graph.add((container_uri, FS.hasRegion, struct_uri))
                self.graph.add((struct_uri, RDF.type, FS.CorticalRegion))
                self.graph.add((struct_uri, FS.hemisphere, Literal(hemisphere)))
            else:
                self.graph.add((container_uri, FS.hasStatisticsItem, struct_uri))
                self.graph.add((struct_uri, RDF.type, FS.StatisticsItem))
            
            # Add label
            self.graph.add((struct_uri, RDFS.label, Literal(name)))
            
            # Add properties
            for key, value in struct.items():
                if key == 'StructName':
                    continue
                    
                property_id = safe_id(key)
                try:
                    if isinstance(value, (int, float)):
                        self.graph.add((struct_uri, FS[property_id], Literal(value, 
                            datatype=XSD.float if isinstance(value, float) else XSD.integer)))
                    else:
                        self.graph.add((struct_uri, FS[property_id], Literal(value)))
                except Exception as e:
                    logger.warning(f"Error adding property {key}: {e}")
            
            # Add standard term mapping if available
            std_term = map_fs_term(name, self.fs_mappings, "region")
            if std_term:
                self.graph.add((struct_uri, RDFS.seeAlso, URIRef(std_term)))


def convert_subject(freesurfer_dir, subject_label, session_label=None, output_dir=None):
    """
    Convert FreeSurfer outputs for a subject to NIDM format.
    
    Parameters
    ----------
    freesurfer_dir : str
        Path to FreeSurfer derivatives directory
    subject_label : str
        Subject label (without 'sub-' prefix)
    session_label : str, optional
        Session label (without 'ses-' prefix)
    output_dir : str, optional
        Output directory for NIDM files
        
    Returns
    -------
    str
        Path to the output NIDM file
    """
    try:
        # Initialize and run converter
        converter = FreeSurferToNIDM(freesurfer_dir, subject_label, session_label, output_dir)
        output_file = converter.convert()
        
        return output_file
    except Exception as e:
        logger.error(f"Error converting subject {subject_label}: {str(e)}")
        raise


def create_group_nidm(subjects, nidm_dir):
    """
    Create group-level NIDM output by aggregating subject-level data.
    
    Parameters
    ----------
    subjects : list
        List of subject labels (with 'sub-' prefix)
    nidm_dir : str
        Output directory for NIDM files
        
    Returns
    -------
    str
        Path to the output group NIDM file
    """
    try:
        # Create output directory
        nidm_dir = Path(nidm_dir)
        os.makedirs(nidm_dir, exist_ok=True)
        
        # Initialize graph
        g = Graph()
        g.bind("nidm", NIDM)
        g.bind("niiri", NIIRI)
        g.bind("prov", PROV)
        g.bind("fs", FS)
        g.bind("bids", BIDS)
        
        # Create group entity
        group_uri = NIIRI[f"group-{uuid4().hex[:8]}"]
        g.add((group_uri, RDF.type, NIDM.Group))
        g.add((group_uri, RDFS.label, Literal("Study Group")))
        
        # Add creation info
        g.add((group_uri, PROV.generatedAtTime, 
               Literal(datetime.datetime.now().isoformat(), datatype=XSD.dateTime)))
        
        # Track processing results
        successful = []
        failed = []
        
        # Add each subject to the group
        for subject in subjects:
            subject_label = subject.replace('sub-', '')
            subject_uri = NIIRI[f"subject-group-{subject_label}"]
            
            # Link subject to group
            g.add((group_uri, PROV.hadMember, subject_uri))
            g.add((subject_uri, BIDS.subject_id, Literal(subject_label)))
            
            # Try to reference subject data
            subject_file = nidm_dir / subject / "prov.jsonld"
            if os.path.exists(subject_file):
                try:
                    # Load subject graph and link to group
                    g.add((subject_uri, RDFS.seeAlso, URIRef(f"file://{subject_file.absolute()}")))
                    successful.append(subject_label)
                except:
                    failed.append(subject_label)
            else:
                logger.warning(f"No data found for subject {subject}")
                failed.append(subject_label)
        
        # Add summary metadata
        g.add((group_uri, NIDM.numberOfSubjects, Literal(len(subjects), datatype=XSD.integer)))
        g.add((group_uri, NIDM.subjectsWithData, Literal(len(successful), datatype=XSD.integer)))
        
        # Write output files
        output_file = nidm_dir / "group_prov.jsonld"
        g.serialize(destination=str(output_file), format="json-ld", indent=4)
        
        turtle_file = nidm_dir / "group_prov.ttl"
        g.serialize(destination=str(turtle_file), format="turtle")
        
        # Create simple summary file
        summary = {
            "timestamp": datetime.datetime.now().isoformat(),
            "total_subjects": len(subjects),
            "successful": len(successful),
            "failed": len(failed),
            "successful_list": successful,
            "failed_list": failed
        }
        
        with open(nidm_dir / "group_summary.json", 'w') as f:
            json.dump(summary, f, indent=4)
        
        logger.info(f"Group NIDM created with {len(successful)} of {len(subjects)} subjects")
        return str(output_file)
        
    except Exception as e:
        logger.error(f"Error creating group NIDM: {str(e)}")
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert FreeSurfer outputs to NIDM format")
    parser.add_argument("freesurfer_dir", help="Path to FreeSurfer derivatives directory")
    parser.add_argument("subject_label", help="Subject label (without 'sub-' prefix)")
    parser.add_argument("--session", help="Session label (without 'ses-' prefix)")
    parser.add_argument("--output", help="Output directory for NIDM files")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Convert subject
        output_file = convert_subject(args.freesurfer_dir, args.subject_label, args.session, args.output)
        print(f"NIDM output created at: {output_file}")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)