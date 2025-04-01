#!/usr/bin/env python3
"""
FreeSurfer to NIDM Converter

This module converts FreeSurfer outputs to NIDM (Neuroimaging Data Model) format
using JSON-LD serialization to enable interoperability and standardized representation
of neuroimaging results. This implementation integrates features from the original
fs2nidm.py and the segstats_jsonld library.
"""

import datetime
import json
import logging
import os
import re
import sys
import tempfile
import urllib.request as ur
from io import StringIO
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

import pandas as pd
from rapidfuzz import fuzz
from rdflib import BNode, Graph, Literal, Namespace, URIRef
from rdflib.namespace import DCTERMS, PROV, RDF, RDFS, XSD
from rdflib.util import guess_format

# Import utility functions
from src.nidm.utils import load_fs_mapping, map_fs_term, parse_fs_stats_file, safe_id

# Configure logging
logger = logging.getLogger("bids-freesurfer.fs2nidm")

# Define namespaces
NIDM = Namespace("http://purl.org/nidash/nidm#")
NIIRI = Namespace("http://iri.nidash.org/")
FS = Namespace("http://purl.org/nidash/freesurfer#")
BIDS = Namespace("http://bids.neuroimaging.io/terms/")
NDAR = Namespace("https://ndar.nih.gov/api/datadictionary/v2/dataelement/")
SIO = Namespace("http://semanticscience.org/resource/")
NFO = Namespace("http://www.semanticdesktop.org/ontologies/2007/03/22/nfo#")

# Constants
FREESURFER_CDE = "https://raw.githubusercontent.com/ReproNim/segstats_jsonld/master/segstats_jsonld/mapping_data/fs_cde.ttl"
MIN_MATCH_SCORE = 30  # minimum match score for fuzzy matching NIDM terms
CDE_TIMEOUT = 5  # timeout in seconds for CDE download

# Mapping for measurement types
MEASURE_OF_KEY = {
    "http://purl.obolibrary.org/obo/PATO_0001591": "curvature",
    "http://purl.obolibrary.org/obo/PATO_0001323": "area",
    "http://uri.interlex.org/base/ilx_0111689": "thickness",
    "http://uri.interlex.org/base/ilx_0738276": "scalar",
    "http://uri.interlex.org/base/ilx_0112559": "volume",
    "https://surfer.nmr.mgh.harvard.edu/folding": "folding",
    "https://surfer.nmr.mgh.harvard.edu/BrainSegVol-to-eTIV": "BrainSegVol-to-eTIV",
    "https://surfer.nmr.mgh.harvard.edu/MaskVol-to-eTIV": "MaskVol-to-eTIV",
}


class FreeSurferToNIDM:
    """Convert FreeSurfer outputs to NIDM format."""

    def __init__(self, freesurfer_dir, subject_id, session_label=None, output_dir=None):
        """Initialize the FreeSurfer to NIDM converter.
        
        Args:
            freesurfer_dir (str): Path to FreeSurfer subjects directory
            subject_id (str): Subject ID (with or without 'sub-' prefix)
            session_label (str, optional): BIDS session label (with or without 'ses-' prefix)
            output_dir (str, optional): Directory to save NIDM output
        """
        self.freesurfer_dir = Path(freesurfer_dir)
        if not self.freesurfer_dir.exists():
            raise ValueError(f"FreeSurfer directory not found: {freesurfer_dir}")
        
        # Handle subject ID with or without prefix
        self.subject_id = subject_id.replace("sub-", "")  # Remove prefix if present
        
        # Handle session label with or without prefix
        self.session_label = session_label.replace("ses-", "") if session_label else None
        
        # Construct BIDS-compliant paths
        self.subject_label = self.subject_id  # Don't add 'sub-' prefix
        self.fs_subject_dir = self.freesurfer_dir / f"sub-{self.subject_id}"
        if not self.fs_subject_dir.exists():
            raise ValueError(f"Subject directory not found: {self.fs_subject_dir}")
        
        if self.session_label:
            self.fs_subject_dir = self.fs_subject_dir / f"ses-{self.session_label}"
            if not self.fs_subject_dir.exists():
                raise ValueError(f"Session directory not found: {self.fs_subject_dir}")
        
        # Initialize RDF graph
        self.graph = Graph()
        self.processed_files = set()
        
        # Set up output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            # Default to a 'nidm' directory next to the FreeSurfer directory
            self.output_dir = self.freesurfer_dir.parent / "nidm"
        
        logger.info(f"Using output directory: {self.output_dir}")
        
        # Load FreeSurfer mappings and CDE graph
        self.fs_mappings = load_fs_mapping()
        self.cde_graph = self._load_cde_graph()
        
        # Bind namespaces to the graph
        self._bind_namespaces()
        
        # Add basic provenance information
        self._add_basic_provenance()

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
            # Add timeout to the request
            response = ur.urlopen(FREESURFER_CDE, timeout=CDE_TIMEOUT)
            content = response.read().decode('utf-8')
            g.parse(data=content, format="turtle")
            return g
        except Exception as e:
            logger.warning(f"Could not load FreeSurfer CDE graph: {str(e)}")
            return None

    def _get_cde_mappings(self):
        """Get CDE mappings from the loaded CDE graph."""
        if not self.cde_graph:
            return pd.DataFrame()  # Return empty DataFrame instead of None

        try:
            query = """
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
            """

            results = []
            qres = self.cde_graph.query(query)
            columns = [str(var) for var in qres.vars]

            for row in qres:
                results.append(list(row))

            return pd.DataFrame(results, columns=columns)
        except Exception as e:
            logger.warning(f"Error querying CDE graph: {str(e)}")
            return pd.DataFrame()  # Return empty DataFrame on error

    def _get_fs_version(self):
        """Get FreeSurfer version from logs or other sources."""
        # Try to find version in recon-all.log
        log_file = self.fs_subject_dir / "scripts" / "recon-all.log"
        if log_file.exists():
            try:
                with open(log_file) as f:
                    content = f.read()

                # Look for version in log
                match = re.search(r"freesurfer-darwin-darwin.+(\d+\.\d+\.\d+)", content)
                if match:
                    return match.group(1)

                match = re.search(r"freesurfer-linux.+(\d+\.\d+\.\d+)", content)
                if match:
                    return match.group(1)

                match = re.search(r"recon-all.+v(\d+\.\d+\.\d+)", content)
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
            self.freesurfer_dir.parent
            / "provenance"
            / f"sub-{self.subject_label}"
            / "provenance.json",
        ]:
            if prov_file.exists():
                try:
                    with open(prov_file) as f:
                        data = json.load(f)

                    # Check different formats
                    if (
                        "Configuration" in data
                        and "FreeSurferVersion" in data["Configuration"]
                    ):
                        return data["Configuration"]["FreeSurferVersion"]

                    if "SoftwareVersion" in data:
                        return data["SoftwareVersion"]
                except:
                    pass

        return None

    def _add_basic_provenance(self):
        """Add basic provenance information to the graph."""
        # Get version information
        from src.utils import get_version_info
        version_info = get_version_info()

        # Add FreeSurfer software information
        self.fs_software = NIIRI[f"software-{uuid4()}"]
        self.graph.add((self.fs_software, RDF.type, PROV.SoftwareAgent))
        self.graph.add((self.fs_software, NIDM.label, Literal("FreeSurfer")))
        self.graph.add((self.fs_software, NIDM.neuroimagingAnalysisSoftware, URIRef("http://surfer.nmr.mgh.harvard.edu/")))
        
        # Add FreeSurfer version information
        self.graph.add((self.fs_software, DCTERMS.hasVersion, Literal(version_info["freesurfer"]["version"])))
        self.graph.add((self.fs_software, NIDM.versionSource, Literal(version_info["freesurfer"]["source"])))
        
        # Add container image if available
        if version_info["freesurfer"]["image"]:
            self.graph.add((self.fs_software, FS.containerImage, Literal(version_info["freesurfer"]["image"])))
        
        # Add build stamp if available
        if version_info["freesurfer"]["build_stamp"]:
            self.graph.add((self.fs_software, FS.buildStamp, Literal(version_info["freesurfer"]["build_stamp"])))

        # Add BIDS-FreeSurfer software information
        bids_software = NIIRI[f"software-{uuid4()}"]
        self.graph.add((bids_software, RDF.type, PROV.SoftwareAgent))
        self.graph.add((bids_software, NIDM.label, Literal("BIDS-FreeSurfer")))
        self.graph.add((bids_software, DCTERMS.hasVersion, Literal(version_info["bids_freesurfer"]["version"])))
        self.graph.add((bids_software, NIDM.versionSource, Literal(version_info["bids_freesurfer"]["source"])))

        # Add Python environment information
        env = NIIRI[f"environment-{uuid4()}"]
        self.graph.add((env, RDF.type, PROV.Location))
        self.graph.add((env, NIDM.label, Literal("Python Environment")))
        
        # Add Python version as a property of the environment
        self.graph.add((env, NIDM.pythonVersion, Literal(version_info["python"]["version"])))
        
        # Add Python package versions
        for package, version in version_info["python"]["packages"].items():
            self.graph.add((env, NIDM.packageVersion, Literal(f"{package}:{version}")))

        # Add subject information
        self.subject_uri = NIIRI[f"subject-{self.subject_id}"]
        self.graph.add((self.subject_uri, RDF.type, NIDM.Subject))
        self.graph.add((self.subject_uri, NIDM.label, Literal(self.subject_id)))

        # Add session information if available
        if self.session_label:
            session_uri = NIIRI[f"session-{self.session_label}"]
            self.graph.add((session_uri, RDF.type, NIDM.Session))
            self.graph.add((session_uri, NIDM.label, Literal(self.session_label)))
            self.graph.add((self.subject_uri, URIRef("http://bids.neuroimaging.io/terms/session_id"), Literal(self.session_label)))
            self.graph.add((session_uri, PROV.wasAssociatedWith, self.subject_uri))

        # Add process information
        self.fs_process = NIIRI[f"process-{uuid4()}"]
        self.graph.add((self.fs_process, RDF.type, NIDM.FreeSurferAnalysis))
        self.graph.add((self.fs_process, PROV.wasAssociatedWith, self.fs_software))
        self.graph.add((self.fs_process, PROV.wasAssociatedWith, bids_software))
        self.graph.add((self.fs_process, PROV.used, self.subject_uri))
        if self.session_label:
            self.graph.add((self.fs_process, PROV.used, session_uri))

        # Add timestamp to process
        self.graph.add((self.fs_process, PROV.startedAtTime, Literal(version_info["bids_freesurfer"]["timestamp"])))

        # Add project information
        project_uri = NIIRI[f"project-{uuid4()}"]
        self.graph.add((project_uri, RDF.type, NIDM.Project))
        self.graph.add((project_uri, NIDM.label, Literal("FreeSurfer Analysis")))
        self.graph.add((project_uri, PROV.wasAssociatedWith, self.fs_software))
        self.graph.add((project_uri, PROV.wasAssociatedWith, bids_software))
        self.graph.add((project_uri, PROV.wasGeneratedBy, self.fs_process))

    def _process_stats_files(self):
        """Process FreeSurfer stats files and add them to the graph."""
        stats_dir = self.fs_subject_dir / "stats"
        if not stats_dir.exists():
            logger.warning(f"Stats directory not found: {stats_dir}")
            return

        # Process aseg.stats
        aseg_file = stats_dir / "aseg.stats"
        if aseg_file.exists():
            container_id = f"aseg-{self.subject_id}"
            container_uri = NIIRI[container_id]
            
            # Add container to graph with proper type and connections
            self.graph.add((container_uri, RDF.type, FS.SegmentationStatistics))
            self.graph.add((container_uri, NIDM.label, Literal(f"ASEG Statistics for {self.subject_id}")))
            self.graph.add((container_uri, PROV.wasGeneratedBy, self.fs_process))
            self.graph.add((self.subject_uri, FS.hasSegmentationStatistics, container_uri))
            self.graph.add((container_uri, NIDM.hasFile, Literal(str(aseg_file))))  # Add file path
            
            # Process the stats file
            stats_data = parse_fs_stats_file(aseg_file)
            if stats_data:
                # Add global measures
                for measure, value in stats_data.get('global_measures', {}).items():
                    measure_uri = NIIRI[f"{container_id}-{safe_id(measure)}"]
                    self.graph.add((measure_uri, RDF.type, NIDM.Measure))
                    self.graph.add((measure_uri, NIDM.label, Literal(measure)))
                    self.graph.add((measure_uri, NIDM.value, Literal(value)))
                    self.graph.add((container_uri, NIDM.hasMeasure, measure_uri))
                
                # Add structure data
                for struct in stats_data.get('structures', []):
                    struct_uri = NIIRI[f"{container_id}-{safe_id(struct.get('StructName', ''))}"]
                    self.graph.add((struct_uri, RDF.type, NIDM.Structure))
                    self.graph.add((struct_uri, NIDM.label, Literal(struct.get('StructName', ''))))
                    self.graph.add((container_uri, NIDM.hasStructure, struct_uri))
                    
                    # Add structure measures
                    for key, value in struct.items():
                        if key != 'StructName':
                            measure_uri = NIIRI[f"{struct_uri}-{safe_id(key)}"]
                            self.graph.add((measure_uri, RDF.type, NIDM.Measure))
                            self.graph.add((measure_uri, NIDM.label, Literal(key)))
                            self.graph.add((measure_uri, NIDM.value, Literal(value)))
                            self.graph.add((struct_uri, NIDM.hasMeasure, measure_uri))
            
            self.processed_files.add("aseg.stats")

        # Process aparc.stats for each hemisphere
        for hemi in ['lh', 'rh']:
            aparc_file = stats_dir / f"{hemi}.aparc.stats"
            if aparc_file.exists():
                container_id = f"aparc-{hemi}-{self.subject_id}"
                container_uri = NIIRI[container_id]
                
                # Add container to graph with proper type and connections
                self.graph.add((container_uri, RDF.type, FS.CorticalStatistics))
                self.graph.add((container_uri, NIDM.label, Literal(f"{hemi.upper()} Cortical Statistics for {self.subject_id}")))
                self.graph.add((container_uri, PROV.wasGeneratedBy, self.fs_process))
                self.graph.add((self.subject_uri, FS.hasCorticalStatistics, container_uri))
                self.graph.add((container_uri, FS.hemisphere, Literal(hemi)))
                self.graph.add((container_uri, NIDM.hasFile, Literal(str(aparc_file))))  # Add file path
                
                # Process the stats file
                stats_data = parse_fs_stats_file(aparc_file)
                if stats_data:
                    # Add global measures
                    for measure, value in stats_data.get('global_measures', {}).items():
                        measure_uri = NIIRI[f"{container_id}-{safe_id(measure)}"]
                        self.graph.add((measure_uri, RDF.type, NIDM.Measure))
                        self.graph.add((measure_uri, NIDM.label, Literal(measure)))
                        self.graph.add((measure_uri, NIDM.value, Literal(value)))
                        self.graph.add((container_uri, NIDM.hasMeasure, measure_uri))
                    
                    # Add structure data
                    for struct in stats_data.get('structures', []):
                        struct_uri = NIIRI[f"{container_id}-{safe_id(struct.get('StructName', ''))}"]
                        self.graph.add((struct_uri, RDF.type, NIDM.Structure))
                        self.graph.add((struct_uri, NIDM.label, Literal(struct.get('StructName', ''))))
                        self.graph.add((container_uri, NIDM.hasStructure, struct_uri))
                        
                        # Add structure measures
                        for key, value in struct.items():
                            if key != 'StructName':
                                measure_uri = NIIRI[f"{struct_uri}-{safe_id(key)}"]
                                self.graph.add((measure_uri, RDF.type, NIDM.Measure))
                                self.graph.add((measure_uri, NIDM.label, Literal(key)))
                                self.graph.add((measure_uri, NIDM.value, Literal(value)))
                                self.graph.add((struct_uri, NIDM.hasMeasure, measure_uri))
                
                self.processed_files.add(f"{hemi}.aparc.stats")

    def _process_surface_file(self, surface_file, measure_type, hemisphere):
        """
        Process a surface measurement file.

        Parameters
        ----------
        surface_file : Path
            Path to surface measurement file
        measure_type : str
            Type of measurement ('thickness', 'area', 'curvature')
        hemisphere : str
            Hemisphere ('lh' or 'rh')
        """
        logger.info(f"Processing {surface_file.name}")

        try:
            # Create a container for the surface measurements
            container_id = f"{hemisphere}-{measure_type}-{self.subject_id}"
            container_uri = NIIRI[container_id]

            # Add container metadata
            self.graph.add((container_uri, RDF.type, FS.SurfaceMeasurement))
            self.graph.add(
                (
                    container_uri,
                    RDFS.label,
                    Literal(f"{hemisphere.upper()} {measure_type.capitalize()} Measurements"),
                )
            )
            self.graph.add((container_uri, FS.hemisphere, Literal(hemisphere)))
            self.graph.add((container_uri, FS.measurementType, Literal(measure_type)))
            self.graph.add(
                (self.subject_uri, FS.hasSurfaceMeasurement, container_uri)
            )

            # Add file info
            self.graph.add((container_uri, PROV.wasGeneratedBy, self.fs_process))
            self.graph.add((container_uri, NIDM.hasFile, Literal(str(surface_file))))

            # Add standard term mapping if available
            std_term = map_fs_term(measure_type, self.fs_mappings, "measure")
            if std_term:
                self.graph.add((container_uri, RDFS.seeAlso, URIRef(std_term)))

        except Exception as e:
            logger.warning(f"Error processing surface file {surface_file}: {str(e)}")

    def _process_mri_files(self):
        """Process FreeSurfer MRI volume files and add to RDF graph."""
        mri_dir = self.fs_subject_dir / "mri"
        if not mri_dir.exists():
            logger.warning(f"MRI directory not found: {mri_dir}")
            return

        # Process segmentation volumes
        for filename in [
            "aparc+aseg.mgz",
            "aparc.a2009s+aseg.mgz",
            "aparc.DKTatlas+aseg.mgz",
            "wmparc.mgz"
        ]:
            file_path = mri_dir / filename
            if file_path.exists():
                self._process_mri_file(file_path)
                self.processed_files.add(filename)

    def _process_mri_file(self, mri_file):
        """
        Process an MRI volume file.

        Parameters
        ----------
        mri_file : Path
            Path to MRI volume file
        """
        logger.info(f"Processing {mri_file.name}")

        try:
            # Create a container for the MRI volume
            container_id = f"mri-{mri_file.stem}-{self.subject_id}"
            container_uri = NIIRI[container_id]

            # Determine the type of volume
            if "aparc+aseg" in mri_file.name:
                vol_type = "CombinedSegmentation"
                atlas = "aparc"
            elif "aparc.a2009s+aseg" in mri_file.name:
                vol_type = "CombinedSegmentation"
                atlas = "a2009s"
            elif "aparc.DKTatlas+aseg" in mri_file.name:
                vol_type = "CombinedSegmentation"
                atlas = "DKTatlas"
            elif "wmparc" in mri_file.name:
                vol_type = "WhiteMatterSegmentation"
                atlas = None
            else:
                vol_type = "MRI"
                atlas = None

            # Add container metadata
            self.graph.add((container_uri, RDF.type, FS.MRIVolume))
            self.graph.add(
                (
                    container_uri,
                    RDFS.label,
                    Literal(f"{vol_type} Volume"),
                )
            )
            if atlas:
                self.graph.add((container_uri, FS.atlas, Literal(atlas)))
            self.graph.add(
                (self.subject_uri, FS.hasMRIVolume, container_uri)
            )

            # Add file info
            self.graph.add((container_uri, PROV.wasGeneratedBy, self.fs_process))
            self.graph.add((container_uri, NIDM.hasFile, Literal(str(mri_file))))

            # Add standard term mapping if available
            std_term = map_fs_term(vol_type, self.fs_mappings, "volume")
            if std_term:
                self.graph.add((container_uri, RDFS.seeAlso, URIRef(std_term)))

        except Exception as e:
            logger.warning(f"Error processing MRI file {mri_file}: {str(e)}")

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
            logger.info(f"Output directory: {self.output_dir}")

            # Process FreeSurfer stats files
            self._process_stats_files()

            # Process MRI volume files
            self._process_mri_files()

            # Create JSON-LD context
            context = {
                "@context": {
                    "nidm": "http://purl.org/nidash/nidm#",
                    "niiri": "http://iri.nidash.org/",
                    "prov": "http://www.w3.org/ns/prov#",
                    "fs": "http://purl.org/nidash/freesurfer#",
                    "bids": "http://bids.neuroimaging.io/terms/",
                    "dcterms": "http://purl.org/dc/terms/",
                    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
                    "xsd": "http://www.w3.org/2001/XMLSchema#",
                    "ndar": "https://ndar.nih.gov/api/datadictionary/v2/dataelement/",
                    "sio": "http://semanticscience.org/resource/",
                    "nfo": "http://www.semanticdesktop.org/ontologies/2007/03/22/nfo#"
                }
            }

            # Ensure output directory exists
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created output directory: {self.output_dir}")

            # Serialize to JSON-LD and Turtle
            if self.session_label:
                output_file = self.output_dir / f"prov-sub-{self.subject_id}_ses-{self.session_label}.jsonld"
                turtle_file = self.output_dir / f"prov-sub-{self.subject_id}_ses-{self.session_label}.ttl"
            else:
                output_file = self.output_dir / f"prov-sub-{self.subject_id}.jsonld"
                turtle_file = self.output_dir / f"prov-sub-{self.subject_id}.ttl"
            
            # First serialize to JSON-LD
            jsonld_data = self.graph.serialize(format="json-ld", indent=4)
            
            # Parse the graph data and merge with context
            graph_data = json.loads(jsonld_data)
            if "@graph" in graph_data:
                context["@graph"] = graph_data["@graph"]
            else:
                context["@graph"] = graph_data
            
            # Write the complete JSON-LD data
            with open(output_file, 'w') as f:
                json.dump(context, f, indent=4)
            logger.info(f"Wrote JSON-LD file to: {output_file}")

            # Also serialize to Turtle for verification
            self.graph.serialize(destination=str(turtle_file), format="turtle")
            logger.info(f"Wrote Turtle file to: {turtle_file}")

            logger.info(f"Processed {len(self.processed_files)} FreeSurfer files")

            return str(output_file)

        except Exception as e:
            logger.error(f"Error converting FreeSurfer outputs: {str(e)}")
            raise


def convert_subject(freesurfer_dir, subject_id, session_label=None, output_dir=None):
    """
    Convert a single subject's FreeSurfer outputs to NIDM format.

    Parameters
    ----------
    freesurfer_dir : str
        Path to FreeSurfer derivatives directory
    subject_id : str
        Subject ID (with or without 'sub-' prefix)
    session_label : str, optional
        Session label (without 'ses-' prefix)
    output_dir : str, optional
        Output directory for NIDM files (default is freesurfer_dir/../nidm)
    """
    converter = FreeSurferToNIDM(
        freesurfer_dir, subject_id, session_label, output_dir
    )
    return converter.convert()


def convert_from_url(url, subject_id, output_dir):
    """
    Convert FreeSurfer outputs from a URL to NIDM format.

    Parameters
    ----------
    url : str
        URL to FreeSurfer outputs
    subject_id : str
        Subject ID
    output_dir : str
        Output directory for NIDM files
    """
    try:
        # Check if it's a supported stats file
        supported_files = [
            "aseg.stats",
            "lh.aparc.stats",
            "rh.aparc.stats",
            "wmparc.stats",
            "brainvol.stats",
        ]
        if not any(ext in url for ext in supported_files):
            raise ValueError(
                f"Unsupported stats file URL: {url}. Supported files: {supported_files}"
            )

        # Download the file
        try:
            opener = ur.urlopen(url)
            temp = tempfile.NamedTemporaryFile(delete=False)
            temp.write(opener.read())
            temp.close()
            stats_file = temp.name
        except Exception as e:
            raise ValueError(
                f"Failed to download stats file from URL: {url}. Error: {str(e)}"
            )

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


def convert_from_csv(
    csv_file, id_field, output_dir, json_map=None, add_to_nidm=None, forcenidm=False
):
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
                    "isAbout": None,  # This should be filled with actual term
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
                    logger.warning(
                        f"Subject {subject_id} not found in NIDM file, skipping"
                    )
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
                graph.add((measure_uri, NIDM.label, Literal(column)))

                # Add value with appropriate datatype
                if isinstance(value, (int, float)):
                    graph.add(
                        (
                            measure_uri,
                            FS.value,
                            Literal(
                                value,
                                datatype=(
                                    XSD.float
                                    if isinstance(value, float)
                                    else XSD.integer
                                ),
                            ),
                        )
                    )
                else:
                    graph.add((measure_uri, FS.value, Literal(value)))

                # Add mapping information if available
                if column in mapping:
                    if "isAbout" in mapping[column] and mapping[column]["isAbout"]:
                        graph.add(
                            (
                                measure_uri,
                                RDFS.seeAlso,
                                URIRef(mapping[column]["isAbout"]),
                            )
                        )

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

    parser = argparse.ArgumentParser(
        description="Convert FreeSurfer outputs to NIDM format"
    )

    # Create mutually exclusive groups for input types
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("-s", "--subject_dir", help="FreeSurfer subject directory")
    input_group.add_argument(
        "-f", "--seg_file", help="Path or URL to FreeSurfer stats file"
    )
    input_group.add_argument(
        "-csv", "--csv_file", help="CSV file with FreeSurfer stats"
    )

    # Subject ID required when using stats file
    parser.add_argument(
        "-subjid", "--subjid", help="Subject ID (required when using -f/--seg_file)"
    )

    # Output options
    parser.add_argument(
        "-o", "--output", dest="output_dir", help="Output directory for NIDM files"
    )
    parser.add_argument(
        "-j",
        "--jsonld",
        action="store_true",
        help="Output in JSON-LD format (default is Turtle)",
    )

    # Session information
    parser.add_argument("--session", help="Session label (without 'ses-' prefix)")

    # CSV options
    parser.add_argument(
        "-idfield", "--idfield", help="Column name in CSV that contains subject IDs"
    )
    parser.add_argument(
        "-json_map", "--json_map", help="JSON mapping file for CSV variables"
    )

    # NIDM options
    parser.add_argument("-n", "--nidm", help="Existing NIDM file to add data to")
    parser.add_argument(
        "-forcenidm",
        "--forcenidm",
        action="store_true",
        help="Force adding to NIDM file even if subject doesn't exist",
    )
    parser.add_argument(
        "-add_de",
        "--add_de",
        action="store_true",
        help="Add data element definitions to NIDM file",
    )

    # Verbosity
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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
                args.subjid or os.path.basename(args.subject_dir).replace("sub-", ""),
                args.session,
                args.output_dir,
            )
            print(f"NIDM output created at: {output_file}")

        elif args.seg_file:
            # Check if it's a URL
            if url_validator(args.seg_file):
                # Convert from URL
                output_file = convert_from_url(
                    args.seg_file, args.subjid, args.output_dir
                )
            else:
                # Convert from local file
                # Create a temporary directory structure
                tmp_dir = tempfile.mkdtemp()
                subj_dir = os.path.join(tmp_dir, f"sub-{args.subjid}")
                stats_dir = os.path.join(subj_dir, "stats")
                os.makedirs(stats_dir, exist_ok=True)

                # Copy the stats file to the temporary directory
                import shutil

                shutil.copy(
                    args.seg_file,
                    os.path.join(stats_dir, os.path.basename(args.seg_file)),
                )

                # Convert from the temporary directory
                output_file = convert_subject(
                    tmp_dir, args.subjid, args.session, args.output_dir
                )

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
                args.forcenidm,
            )
            print(f"NIDM output created at: {output_file}")

    except Exception as e:
        print(f"Error: {str(e)}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)

    sys.exit(0)
