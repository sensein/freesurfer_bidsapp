#!/usr/bin/env python3
"""
FreeSurfer to NIDM Converter

This module converts FreeSurfer outputs to NIDM (Neuroimaging Data Model) format
using JSON-LD serialization to enable interoperability and standardized representation
of neuroimaging results. This implementation integrates features from the original
fs2nidm.py and the segstats_jsonld library.
"""

import os
import json
import logging
import datetime
import re
import sys
import tempfile
import urllib.request as ur
from urllib.parse import urlparse
from pathlib import Path
from uuid import uuid4
from io import StringIO

import pandas as pd
from rapidfuzz import fuzz
from rdflib import Graph, URIRef, Literal, Namespace, BNode
from rdflib.namespace import RDF, RDFS, XSD, PROV, DCTERMS
from rdflib.util import guess_format

# Import utility functions
from src.nidm.utils import parse_fs_stats_file, safe_id, load_fs_mapping, map_fs_term

# Configure logging
logger = logging.getLogger('bids-freesurfer.fs2nidm')

# Define namespaces
NIDM = Namespace("http://purl.org/nidash/nidm#")
NIIRI = Namespace("http://iri.nidash.org/")
FS = Namespace("http://surfer.nmr.mgh.harvard.edu/fs/terms/")
BIDS = Namespace("http://bids.neuroimaging.io/terms/")
NDAR = Namespace("https://ndar.nih.gov/api/datadictionary/v2/dataelement/")
SIO = Namespace("http://semanticscience.org/resource/")
NFO = Namespace("http://www.semanticdesktop.org/ontologies/2007/03/22/nfo#")

# Constants
FREESURFER_CDE = 'https://raw.githubusercontent.com/ReproNim/segstats_jsonld/master/segstats_jsonld/mapping_data/fs_cde.ttl'
MIN_MATCH_SCORE = 30  # minimum match score for fuzzy matching NIDM terms

# Mapping for measurement types
MEASURE_OF_KEY = {
    "http://purl.obolibrary.org/obo/PATO_0001591": "curvature",
    "http://purl.obolibrary.org/obo/PATO_0001323": "area",
    "http://uri.interlex.org/base/ilx_0111689": "thickness",
    "http://uri.interlex.org/base/ilx_0738276": "scalar",
    "http://uri.interlex.org/base/ilx_0112559": "volume",
    "https://surfer.nmr.mgh.harvard.edu/folding": "folding",
    "https://surfer.nmr.mgh.harvard.edu/BrainSegVol-to-eTIV": "BrainSegVol-to-eTIV",
    "https://surfer.nmr.mgh.harvard.edu/MaskVol-to-eTIV": "MaskVol-to-eTIV"
}


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
        
        # Try to load CDE graph
        try:
            self.cde_graph = self._load_cde_graph()
        except Exception as e:
            logger.warning(f"Could not load CDE graph: {str(e)}. Will use local mappings only.")
            self.cde_graph = None
    
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
        self.graph.bind("ndar", NDAR)
        self.graph.bind("sio", SIO)
        self.graph.bind("nfo", NFO)
    
    def _load_cde_graph(self):
        """Load the FreeSurfer CDE graph from the web."""
        g = Graph()
        try:
            g.parse(location=FREESURFER_CDE, format="turtle")
            return g
        except Exception as e:
            logger.warning(f"Could not load FreeSurfer CDE graph: {str(e)}")
            return None
    
    def _get_cde_mappings(self):
        """Get CDE mappings from the loaded CDE graph."""
        if not self.cde_graph:
            return {}
        
        query = '''
            prefix fs: <https://surfer.nmr.mgh.harvard.edu/>
            prefix nidm: <http://purl.org/nidash/nidm#>
            prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            prefix xsd: <http://www.w3.org/2001/XMLSchema#>

            select distinct ?uuid ?measure ?structure ?label
            where {
                ?uuid a fs:DataElement ;
                    nidm:measureOf ?measure ;
                    rdfs:label ?label ;
                    fs:structure ?structure .
            } order by ?uuid
        '''
        
        results = []
        qres = self.cde_graph.query(query)
        columns = [str(var) for var in qres.vars]
        
        for row in qres:
            results.append(list(row))
        
        return pd.DataFrame(results, columns=columns)
    
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
        
        # Try build-stamp.txt file
        build_stamp = self.fs_subject_dir / "scripts" / "build-stamp.txt"
        if build_stamp.exists():
            try:
                with open(build_stamp) as f:
                    return f.readline().strip()
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
    
    def _add_basic_provenance(self):
        """Add basic provenance information to the RDF graph."""
        # Add subject information
        self.graph.add((self.subject_uri, RDF.type, PROV.Entity))
        self.graph.add((self.subject_uri, RDF.type, NIDM.Subject))
        self.graph.add((self.subject_uri, RDFS.label, Literal(f"Subject {self.subject_label}")))
        self.graph.add((self.subject_uri, BIDS.subject_id, Literal(self.subject_label)))
        self.graph.add((self.subject_uri, NDAR.src_subject_id, Literal(self.subject_label)))
        
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
        self.graph.add((self.fs_software, NIDM.neuroimagingAnalysisSoftware, URIRef("http://surfer.nmr.mgh.harvard.edu/")))
        
        # Try to get version information
        version = self._get_fs_version()
        self.graph.add((self.fs_software, PROV.version, Literal(version or "unknown")))
        
        # Add timestamp
        current_time = datetime.datetime.now().isoformat()
        self.graph.add((self.fs_process, PROV.startedAtTime, Literal(current_time, datatype=XSD.dateTime)))
        
        # Add qualified association between subject and process
        association_bnode = BNode()
        self.graph.add((self.fs_process, PROV.qualifiedAssociation, association_bnode))
        self.graph.add((association_bnode, RDF.type, PROV.Association))
        self.graph.add((association_bnode, PROV.hadRole, SIO.Subject))
        self.graph.add((association_bnode, PROV.agent, self.subject_uri))
        
        # Create project and session
        project_uri = NIIRI[f"project-{uuid4().hex[:8]}"]
        self.graph.add((project_uri, RDF.type, NIDM.Project))
        
        session_uri = NIIRI[f"session-{self.subject_label}"]
        self.graph.add((session_uri, RDF.type, PROV.Activity))
        self.graph.add((session_uri, RDF.type, NIDM.Session))
        self.graph.add((session_uri, DCTERMS.isPartOf, project_uri))
        
        # Associate process with session
        self.graph.add((self.fs_process, DCTERMS.isPartOf, session_uri))
    
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
            
            # Try to find match in CDE graph using fuzzy matching
            if self.cde_graph:
                # Find matching term in CDE graph
                matches = self._find_matching_cde_terms(name)
                if matches:
                    best_match = matches[0]  # Use the best match
                    self.graph.add((measure_uri, RDFS.seeAlso, URIRef(best_match)))
        
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
            
            # Try to find match in CDE graph using fuzzy matching
            if self.cde_graph:
                # Find matching term in CDE graph
                matches = self._find_matching_cde_terms(name, type="region")
                if matches:
                    best_match = matches[0]  # Use the best match
                    self.graph.add((struct_uri, RDFS.seeAlso, URIRef(best_match)))
    
    def _find_matching_cde_terms(self, term, type="measure"):
        """
        Find matching terms in the CDE graph using fuzzy matching.
        
        Parameters
        ----------
        term : str
            The term to match
        type : str
            The type of term to match ('measure' or 'region')
            
        Returns
        -------
        list
            List of matching URIs, ordered by match score
        """
        if not self.cde_graph:
            return []
        
        # Get all labels from CDE graph
        query = '''
            prefix fs: <https://surfer.nmr.mgh.harvard.edu/>
            prefix nidm: <http://purl.org/nidash/nidm#>
            prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            
            select ?uuid ?label
            where {
                ?uuid a fs:DataElement ;
                    rdfs:label ?label .
            }
        '''
        
        qres = self.cde_graph.query(query)
        
        # Calculate fuzzy match scores
        match_scores = {}
        for row in qres:
            uuid = str(row[0])
            label = str(row[1]).lower()
            
            # Calculate match score
            score = fuzz.token_sort_ratio(term.lower(), label)
            
            if score > MIN_MATCH_SCORE:
                match_scores[uuid] = score
        
        # Sort by score and return URIs
        return [k for k, v in sorted(match_scores.items(), key=lambda item: item[1], reverse=True)]
    
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


def convert_from_url(url, subject_id, output_dir):
    """
    Convert FreeSurfer stats from a URL.
    
    Parameters
    ----------
    url : str
        URL to FreeSurfer stats file
    subject_id : str
        Subject ID to use for the NIDM file
    output_dir : str
        Output directory for NIDM files
        
    Returns
    -------
    str
        Path to the output NIDM file
    """
    try:
        # Check if it's a supported stats file
        supported_files = ['aseg.stats', 'lh.aparc.stats', 'rh.aparc.stats', 'wmparc.stats', 'brainvol.stats']
        if not any(ext in url for ext in supported_files):
            raise ValueError(f"Unsupported stats file URL: {url}. Supported files: {supported_files}")
        
        # Download the file
        try:
            opener = ur.urlopen(url)
            temp = tempfile.NamedTemporaryFile(delete=False)
            temp.write(opener.read())
            temp.close()
            stats_file = temp.name
        except Exception as e:
            raise ValueError(f"Failed to download stats file from URL: {url}. Error: {str(e)}")
        
        # Parse the file to get header information for provenance
        # We'll need to implement this based on the format of your stats files
        
        # Create a temporary directory for the subject
        tmp_subj_dir = tempfile.mkdtemp()
        
        # Create a stats directory inside the temporary subject directory
        stats_dir = os.path.join(tmp_subj_dir, "stats")
        os.makedirs(stats_dir, exist_ok=True)
        
        # Copy the downloaded file to the stats directory with the original name
        filename = os.path.basename(urlparse(url).path)
        dest_file = os.path.join(stats_dir, filename)
        os.rename(stats_file, dest_file)
        
        # Run the conversion
        output_file = convert_subject(tmp_subj_dir, subject_id, output_dir=output_dir)
        
        # Clean up
        import shutil
        shutil.rmtree(tmp_subj_dir)
        
        return output_file
    
    except Exception as e:
        logger.error(f"Error converting from URL: {str(e)}")
        raise


def convert_from_csv(csv_file, id_field, output_dir, json_map=None, add_to_nidm=None, forcenidm=False):
    """
    Convert FreeSurfer stats from a CSV file to NIDM format.
    
    Parameters
    ----------
    csv_file : str
        Path to CSV file containing FreeSurfer stats
    id_field : str
        Column name in CSV file that contains subject IDs
    output_dir : str
        Output directory for NIDM files
    json_map : str, optional
        Path to JSON mapping file to use for CSV variable mapping
    add_to_nidm : str, optional
        Path to existing NIDM file to add the stats to
    forcenidm : bool, optional
        Force adding to NIDM file even if subject doesn't exist
        
    Returns
    -------
    str
        Path to the output NIDM file
    """
    try:
        # Load the CSV file
        df = pd.read_csv(csv_file, dtype={id_field: str})
        
        # Create output directory
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Load JSON mapping if provided
        if json_map:
            with open(json_map) as f:
                mapping = json.load(f)
        else:
            # Create a mapping dictionary
            # This is a simplified version - the real implementation would be more complex
            mapping = {}
            for column in df.columns:
                if column == id_field:
                    continue
                
                # For demonstration, just use a simple mapping
                mapping[column] = {
                    "label": column,
                    "description": f"FreeSurfer measure: {column}",
                    "isAbout": None  # This should be filled with actual term
                }
        
        # Create or load NIDM graph
        if add_to_nidm:
            graph = Graph()
            graph.parse(add_to_nidm, format=guess_format(add_to_nidm))
        else:
            graph = Graph()
            graph.bind("nidm", NIDM)
            graph.bind("niiri", NIIRI)
            graph.bind("prov", PROV)
            graph.bind("fs", FS)
            graph.bind("bids", BIDS)
            graph.bind("dcterms", DCTERMS)
            graph.bind("rdfs", RDFS)
            graph.bind("xsd", XSD)
            
            # Create project
            project_uri = NIIRI[f"project-{uuid4().hex[:8]}"]
            graph.add((project_uri, RDF.type, NIDM.Project))
            graph.add((project_uri, RDFS.label, Literal("FreeSurfer Project")))
            
        # Process each row (subject) in the CSV
        for _, row in df.iterrows():
            subject_id = row[id_field]
            
            # Check if subject exists in NIDM file if adding to existing
            if add_to_nidm and not forcenidm:
                # Query for subject
                query = f"""
                    PREFIX ndar: <https://ndar.nih.gov/api/datadictionary/v2/dataelement/>
                    SELECT ?s WHERE {{ ?s ndar:src_subject_id "{subject_id}"^^xsd:string }}
                """
                qres = graph.query(query)
                if len(qres) == 0:
                    logger.warning(f"Subject {subject_id} not found in NIDM file, skipping")
                    continue
            
            # Create or retrieve subject URI
            if add_to_nidm and not forcenidm:
                for res in qres:
                    subject_uri = res[0]
                    break
            else:
                subject_uri = NIIRI[f"subject-{subject_id}"]
                graph.add((subject_uri, RDF.type, PROV.Agent))
                graph.add((subject_uri, RDF.type, PROV.Person))
                graph.add((subject_uri, NDAR.src_subject_id, Literal(subject_id)))
                
            # Create activity
            activity_uri = NIIRI[f"activity-{subject_id}-{uuid4().hex[:8]}"]
            graph.add((activity_uri, RDF.type, PROV.Activity))
            graph.add((activity_uri, RDF.type, NIDM.FreeSurferAnalysis))
            
            # Associate activity with subject
            association_bnode = BNode()
            graph.add((activity_uri, PROV.qualifiedAssociation, association_bnode))
            graph.add((association_bnode, RDF.type, PROV.Association))
            graph.add((association_bnode, PROV.hadRole, SIO.Subject))
            graph.add((association_bnode, PROV.agent, subject_uri))
            
            # Add measurements
            container_uri = NIIRI[f"fs-stats-{subject_id}"]
            graph.add((container_uri, RDF.type, FS.SegmentationStatistics))
            graph.add((container_uri, PROV.wasGeneratedBy, activity_uri))
            
            # Add each measurement from the CSV
            for column in df.columns:
                if column == id_field:
                    continue
                
                value = row[column]
                if pd.isna(value):
                    continue
                
                # Create measurement URI
                measure_uri = NIIRI[f"fs-measure-{subject_id}-{safe_id(column)}"]
                graph.add((container_uri, FS[safe_id(column)], measure_uri))
                graph.add((measure_uri, RDF.type, FS.Measurement))
                graph.add((measure_uri, RDFS.label, Literal(column)))
                
                # Add value with appropriate datatype
                if isinstance(value, (int, float)):
                    graph.add((measure_uri, FS.value, Literal(value, 
                        datatype=XSD.float if isinstance(value, float) else XSD.integer)))
                else:
                    graph.add((measure_uri, FS.value, Literal(value)))
                
                # Add mapping information if available
                if column in mapping:
                    if "isAbout" in mapping[column] and mapping[column]["isAbout"]:
                        graph.add((measure_uri, RDFS.seeAlso, URIRef(mapping[column]["isAbout"])))
        
        # Write output
        output_file = output_dir / "fs_csv_nidm.jsonld"
        graph.serialize(destination=str(output_file), format="json-ld", indent=4)
        
        turtle_file = output_dir / "fs_csv_nidm.ttl"
        graph.serialize(destination=str(turtle_file), format="turtle")
        
        return str(output_file)
    
    except Exception as e:
        logger.error(f"Error converting from CSV: {str(e)}")
        raise


def url_validator(url):
    """
    Validate a URL.
    
    Parameters
    ----------
    url : str
        URL to validate
    
    Returns
    -------
    bool
        True if URL is valid, False otherwise
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc, result.path])
    except:
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert FreeSurfer outputs to NIDM format")
    
    # Create mutually exclusive groups for input types
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("-s", "--subject_dir", help="FreeSurfer subject directory")
    input_group.add_argument("-f", "--seg_file", help="Path or URL to FreeSurfer stats file")
    input_group.add_argument("-csv", "--csv_file", help="CSV file with FreeSurfer stats")
    
    # Subject ID required when using stats file
    parser.add_argument("-subjid", "--subjid", help="Subject ID (required when using -f/--seg_file)")
    
    # Output options
    parser.add_argument("-o", "--output", dest="output_dir", help="Output directory for NIDM files")
    parser.add_argument("-j", "--jsonld", action="store_true", help="Output in JSON-LD format (default is Turtle)")
    
    # Session information
    parser.add_argument("--session", help="Session label (without 'ses-' prefix)")
    
    # CSV options
    parser.add_argument("-idfield", "--idfield", help="Column name in CSV that contains subject IDs")
    parser.add_argument("-json_map", "--json_map", help="JSON mapping file for CSV variables")
    
    # NIDM options
    parser.add_argument("-n", "--nidm", help="Existing NIDM file to add data to")
    parser.add_argument("-forcenidm", "--forcenidm", action="store_true", 
                      help="Force adding to NIDM file even if subject doesn't exist")
    parser.add_argument("-add_de", "--add_de", action="store_true",
                      help="Add data element definitions to NIDM file")
    
    # Verbosity
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Validate arguments
    if args.seg_file and not args.subjid:
        parser.error("-f/--seg_file requires -subjid/--subjid")
    
    if args.csv_file and not args.idfield:
        parser.error("-csv/--csv_file requires -idfield/--idfield")
    
    try:
        # Process based on input type
        if args.subject_dir:
            # Convert from FreeSurfer subject directory
            output_file = convert_subject(
                args.subject_dir, 
                args.subjid or os.path.basename(args.subject_dir).replace('sub-', ''),
                args.session,
                args.output_dir
            )
            print(f"NIDM output created at: {output_file}")
            
        elif args.seg_file:
            # Check if it's a URL
            if url_validator(args.seg_file):
                # Convert from URL
                output_file = convert_from_url(args.seg_file, args.subjid, args.output_dir)
            else:
                # Convert from local file
                # Create a temporary directory structure
                tmp_dir = tempfile.mkdtemp()
                subj_dir = os.path.join(tmp_dir, f"sub-{args.subjid}")
                stats_dir = os.path.join(subj_dir, "stats")
                os.makedirs(stats_dir, exist_ok=True)
                
                # Copy the stats file to the temporary directory
                import shutil
                shutil.copy(args.seg_file, os.path.join(stats_dir, os.path.basename(args.seg_file)))
                
                # Convert from the temporary directory
                output_file = convert_subject(tmp_dir, args.subjid, args.session, args.output_dir)
                
                # Clean up
                shutil.rmtree(tmp_dir)
                
            print(f"NIDM output created at: {output_file}")
            
        elif args.csv_file:
            # Convert from CSV file
            output_file = convert_from_csv(
                args.csv_file,
                args.idfield,
                args.output_dir,
                args.json_map,
                args.nidm,
                args.forcenidm
            )
            print(f"NIDM output created at: {output_file}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
        
    sys.exit(0)