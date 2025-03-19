#!/usr/bin/env python3
"""
Utility functions for FreeSurfer to NIDM conversion

This module provides helper functions for parsing FreeSurfer outputs,
handling RDF graphs, and generating NIDM-compatible identifiers.
"""

import json
import logging
import os
import re
import tempfile
import urllib.request as ur
from collections import namedtuple
from io import StringIO
from pathlib import Path

import pandas as pd

try:
    from rapidfuzz import fuzz
except ImportError:
    fuzz = None

from rdflib import RDF, RDFS, XSD, BNode, Graph, Literal, Namespace, URIRef

# Configure logging
logger = logging.getLogger("bids-freesurfer.nidm.utils")

# Define common namespaces
NIDM = Namespace("http://purl.org/nidash/nidm#")
NIIRI = Namespace("http://iri.nidash.org/")
FS = Namespace("http://surfer.nmr.mgh.harvard.edu/fs/terms/")
NDAR = Namespace("https://ndar.nih.gov/api/datadictionary/v2/dataelement/")
SIO = Namespace("http://semanticscience.org/resource/")
NFO = Namespace("http://www.semanticdesktop.org/ontologies/2007/03/22/nfo#")

# Define FreeSurfer structure
FSEntry = namedtuple("FSEntry", ["structure", "hemi", "measure", "unit"])

# Minimum match score for fuzzy matching
MIN_MATCH_SCORE = 30

# Paths to FreeSurfer mapping files
MODULE_DIR = Path(os.path.dirname(__file__))
MAPPINGS_DIR = MODULE_DIR / "mappings"
LUT_PATH = os.path.join(MAPPINGS_DIR, "FreeSurferColorLUT.txt")
FS_MAP_PATH = os.path.join(MAPPINGS_DIR, "fsmap.json")
FS_CDE_PATH = os.path.join(MAPPINGS_DIR, "fs_cde.json")
FS_CDE_TTL_PATH = os.path.join(MAPPINGS_DIR, "fs_cde.ttl")


def safe_id(text):
    """
    Create a safe identifier from text by removing special characters.

    Parameters
    ----------
    text : str
        Text to convert to a safe identifier

    Returns
    -------
    str
        Safe identifier string
    """
    if not text:
        return "unnamed"

    # Replace non-alphanumeric characters with underscores
    safe = re.sub(r"[^a-zA-Z0-9_]", "_", str(text)).lower()

    # Ensure it doesn't start with a number
    if safe and safe[0].isdigit():
        safe = "n" + safe

    return safe


def parse_fs_stats_file(stats_file):
    """
    Parse a FreeSurfer stats file and extract measurements.

    Parameters
    ----------
    stats_file : str or Path
        Path to FreeSurfer stats file

    Returns
    -------
    dict
        Dictionary with global measures and per-structure data
    """
    if not os.path.exists(stats_file):
        logger.error(f"Stats file not found: {stats_file}")
        return None

    result = {"global_measures": {}, "structures": []}

    try:
        with open(stats_file, "r") as f:
            lines = f.readlines()

        # Extract global measures (# Measure lines)
        for line in lines:
            if line.startswith("# Measure"):
                parts = line.strip().split(",")
                if len(parts) >= 4:
                    name = parts[1].strip()
                    try:
                        value = float(parts[3].strip())
                        result["global_measures"][name] = value
                    except ValueError:
                        logger.debug(
                            f"Could not convert measure value to float: {parts[3]}"
                        )

        # Find the column headers
        header_idx = None
        for i, line in enumerate(lines):
            if line.startswith("# ColHeaders"):
                header_idx = i
                break

        # Extract column data if headers are found
        if header_idx is not None:
            headers = lines[header_idx].replace("# ColHeaders", "").strip().split()

            # Process data rows
            for line in lines[header_idx + 1 :]:
                if line.startswith("#") or not line.strip():
                    continue

                parts = line.strip().split()
                if len(parts) < len(headers):
                    continue

                # Create a dictionary for this structure
                struct_data = {headers[i]: parts[i] for i in range(len(headers))}

                # Convert numeric values
                for key, value in list(struct_data.items()):
                    try:
                        struct_data[key] = float(value)
                    except ValueError:
                        # Keep as string if not a number
                        pass

                result["structures"].append(struct_data)
        else:
            # For files without column headers (like aseg.stats)
            # Extract volume data directly
            data_lines = [
                line
                for line in lines
                if not line.startswith("#") and len(line.strip()) > 0
            ]

            for line in data_lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    # Assumes standard aseg.stats format
                    struct_data = {
                        "Index": int(parts[0]),
                        "SegId": int(parts[1]),
                        "NVoxels": int(parts[2]),
                        "Volume_mm3": float(parts[3]),
                        "StructName": parts[4],
                    }

                    # Add normalized volume if ICV is available
                    if "EstimatedTotalIntraCranialVol" in result["global_measures"]:
                        icv = result["global_measures"]["EstimatedTotalIntraCranialVol"]
                        if icv > 0:
                            struct_data["Volume_Normalized"] = (
                                struct_data["Volume_mm3"] / icv
                            ) * 100

                    result["structures"].append(struct_data)

        return result

    except Exception as e:
        logger.error(f"Error parsing stats file {stats_file}: {e}")
        return None


def extract_brain_structures(aseg_data):
    """
    Extract brain structure information from aseg.stats data.

    Parameters
    ----------
    aseg_data : dict
        Parsed aseg.stats data from parse_fs_stats_file

    Returns
    -------
    dict
        Dictionary mapping structure names to their properties
    """
    if not aseg_data or "structures" not in aseg_data:
        return {}

    structures = {}

    for struct in aseg_data["structures"]:
        if "StructName" in struct and "Volume_mm3" in struct:
            name = struct["StructName"]
            structures[name] = {
                "volume": struct["Volume_mm3"],
                "index": struct.get("Index", None),
                "nvoxels": struct.get("NVoxels", None),
                "normalized_volume": struct.get("Volume_Normalized", None),
            }

    return structures


def extract_cortical_measures(aparc_data, hemisphere):
    """
    Extract cortical measures from aparc.stats data.

    Parameters
    ----------
    aparc_data : dict
        Parsed aparc.stats data from parse_fs_stats_file
    hemisphere : str
        Hemisphere identifier ('lh' or 'rh')

    Returns
    -------
    dict
        Dictionary with cortical measures by region
    """
    if not aparc_data or "structures" not in aparc_data:
        return {}

    measures = {"hemisphere": hemisphere, "regions": {}}

    # Add global measures
    if "global_measures" in aparc_data:
        measures["global"] = aparc_data["global_measures"]

    # Process each region
    for region in aparc_data["structures"]:
        if "StructName" in region:
            name = region["StructName"]
            region_data = {}

            # Extract common metrics
            metrics = [
                "NumVert",
                "SurfArea",
                "GrayVol",
                "ThickAvg",
                "ThickStd",
                "MeanCurv",
                "GausCurv",
            ]
            for metric in metrics:
                if metric in region:
                    region_data[metric] = region[metric]

            measures["regions"][name] = region_data

    return measures


def load_fs_mapping(mapping_file=None):
    """
    Load FreeSurfer to standard terminology mapping.

    Parameters
    ----------
    mapping_file : str, optional
        Path to mapping file (JSON)

    Returns
    -------
    dict
        Mapping dictionary
    """
    # Default mapping file location
    if not mapping_file:
        mapping_file = FS_MAP_PATH

    # If mapping file exists, load it
    if os.path.exists(mapping_file):
        try:
            with open(mapping_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading mapping file {mapping_file}: {e}")

    # Return empty mapping if file not found or error
    return {}


def map_fs_term(term, mapping_dict, category=None):
    """
    Map a FreeSurfer term to standard terminology.

    Parameters
    ----------
    term : str
        FreeSurfer term to map
    mapping_dict : dict
        Mapping dictionary from load_fs_mapping
    category : str, optional
        Category to restrict mapping search (e.g., 'region', 'measure')

    Returns
    -------
    str or None
        Mapped term if found, otherwise None
    """
    if not mapping_dict or not term:
        return None

    # Normalize the term for lookup
    norm_term = safe_id(term)

    # First check if this is a measure
    if category == "measure" and "Measures" in mapping_dict:
        if term in mapping_dict["Measures"]:
            measure_info = mapping_dict["Measures"][term]
            if "measureOf" in measure_info:
                return measure_info["measureOf"]

    # Check if this is a region/structure
    if category == "region" and "Structures" in mapping_dict:
        # First check for direct match
        if term in mapping_dict["Structures"]:
            struct_info = mapping_dict["Structures"][term]
            if "isAbout" in struct_info and struct_info["isAbout"]:
                if not struct_info["isAbout"].startswith("CUSTOM") and not struct_info[
                    "isAbout"
                ].startswith("<UNKNOWN"):
                    return struct_info["isAbout"]

        # Check for match in fskey lists
        for struct_name, struct_info in mapping_dict["Structures"].items():
            if "fskey" in struct_info and term in struct_info["fskey"]:
                if "isAbout" in struct_info and struct_info["isAbout"]:
                    if not struct_info["isAbout"].startswith(
                        "CUSTOM"
                    ) and not struct_info["isAbout"].startswith("<UNKNOWN"):
                        return struct_info["isAbout"]

    # Legacy approach for simpler mappings
    if norm_term in mapping_dict:
        entry = mapping_dict[norm_term]

        # If category specified, check category
        if category and "category" in entry and entry["category"] != category:
            return None

        # Return the standard term
        return entry.get("standard_term", None)

    return None


def get_segid(filename, structure):
    """
    Get the segmentation ID of a FreeSurfer structure

    Parameters
    ----------
    filename : str
        Path to the stats file
    structure : str
        Structure name

    Returns
    -------
    int or None
        Segmentation ID if found, otherwise None
    """
    structure = structure.replace("&", "_and_")
    filename = str(filename)
    label = structure

    # Determine hemisphere and adjust label
    hemi = None
    if "lh" in filename:
        hemi = "lh"
    if "rh" in filename:
        hemi = "rh"

    # Handle special file formats
    if hemi and (
        "aparc.stats" in filename
        or "a2005" in filename
        or "DKT" in filename
        or "aparc.pial" in filename
        or "w-g.pct" in filename
    ):
        label = f"ctx-{hemi}-{structure}"

    if hemi and "a2009" in filename:
        label = f"ctx_{hemi}_{structure}"

    if "BA" in filename:
        label = structure.split("_exvivo")[0]

    # Look up in the color table
    try:
        if os.path.exists(LUT_PATH):
            with open(LUT_PATH, "rt") as fp:
                for line in fp.readlines():
                    if line.startswith("#") or not line.strip():
                        continue
                    vals = line.split()
                    if len(vals) > 2 and vals[1] == label:
                        return int(vals[0])
    except Exception as e:
        logger.warning(f"Error finding segID for {filename} - {structure}: {e}")

    return None


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
        from urllib.parse import urlparse

        result = urlparse(url)
        return all([result.scheme, result.netloc, result.path])
    except:
        return False


def find_matching_terms(term, cde_graph, type="measure", threshold=MIN_MATCH_SCORE):
    """
    Find matching terms in a CDE graph using fuzzy matching.

    Parameters
    ----------
    term : str
        The term to match
    cde_graph : Graph
        RDFLib graph containing CDE definitions
    type : str, optional
        Type of term to match (measure or region)
    threshold : int, optional
        Minimum match score (0-100)

    Returns
    -------
    list
        List of matching URIs, ordered by match score
    """
    if not cde_graph or not fuzz:
        return []

    # Query for term labels
    query = """
        PREFIX fs: <http://surfer.nmr.mgh.harvard.edu/fs/terms/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?uri ?label
        WHERE {
            ?uri a fs:DataElement ;
                 rdfs:label ?label .
        }
    """

    qres = cde_graph.query(query)

    # Calculate match scores
    matches = {}
    for row in qres:
        uri = str(row[0])
        label = str(row[1]).lower()

        score = fuzz.token_sort_ratio(term.lower(), label)

        if score >= threshold:
            matches[uri] = score

    # Sort by score (highest first)
    return [
        k for k, v in sorted(matches.items(), key=lambda item: item[1], reverse=True)
    ]


def create_cde_graph(fs_mapping=None):
    """
    Create a graph containing FreeSurfer Common Data Elements.

    Parameters
    ----------
    fs_mapping : dict, optional
        FreeSurfer mapping dictionary

    Returns
    -------
    Graph
        RDFLib graph with CDE definitions
    """
    # If CDE TTL file exists, load it directly
    if os.path.exists(FS_CDE_TTL_PATH):
        try:
            g = Graph()
            g.parse(FS_CDE_TTL_PATH, format="turtle")
            logger.info(f"Loaded CDE graph from {FS_CDE_TTL_PATH}")
            return g
        except Exception as e:
            logger.warning(
                f"Error loading CDE TTL file: {e}. Will create graph from scratch."
            )

    # Otherwise, create the graph from scratch
    g = Graph()

    # Bind namespaces
    g.bind("fs", FS)
    g.bind("nidm", NIDM)
    g.bind("rdfs", RDFS)
    g.bind("xsd", XSD)

    # Define relationship between FS DataElement and NIDM DataElement
    g.add((FS["DataElement"], RDFS.subClassOf, NIDM["DataElement"]))

    # If fs_cde.json exists, load and process it
    if os.path.exists(FS_CDE_PATH):
        try:
            with open(FS_CDE_PATH, "r") as f:
                fs_cde = json.load(f)

            # Add each CDE
            for key, value in fs_cde.items():
                if key == "count":
                    continue

                cde_id = f"fs_{value['id']}"
                g.add((FS[cde_id], RDF.type, FS["DataElement"]))

                # Add properties
                for prop, val in value.items():
                    if prop == "id" or val is None:
                        continue

                    if prop == "label":
                        g.add((FS[cde_id], RDFS.label, Literal(val)))
                    elif prop == "description":
                        g.add((FS[cde_id], RDFS.comment, Literal(val)))
                    elif prop in ["isAbout", "datumType", "measureOf"]:
                        g.add((FS[cde_id], NIDM[prop], URIRef(val)))
                    elif prop == "hasUnit":
                        g.add((FS[cde_id], NIDM[prop], Literal(val)))
                    else:
                        g.add((FS[cde_id], FS[prop], Literal(val)))
        except Exception as e:
            logger.warning(f"Error loading CDE file: {e}")

    # If mapping provided, add information from fsmap.json
    if fs_mapping:
        try:
            # Add measures
            if "Measures" in fs_mapping:
                for measure_name, measure_info in fs_mapping["Measures"].items():
                    measure_id = f"measure_{safe_id(measure_name)}"
                    g.add((FS[measure_id], RDF.type, FS["Measure"]))
                    g.add((FS[measure_id], RDFS.label, Literal(measure_name)))

                    for prop, val in measure_info.items():
                        if val is None:
                            continue

                        if prop == "measureOf":
                            if val.startswith("fs:"):
                                val = FS[val.replace("fs:", "")]
                            else:
                                val = URIRef(val)
                            g.add((FS[measure_id], NIDM.measureOf, val))
                        elif prop == "datumType":
                            g.add((FS[measure_id], NIDM.datumType, URIRef(val)))
                        elif prop == "hasUnit" or prop == "fsunit":
                            if isinstance(val, str) and val.startswith("fs:"):
                                unit_val = FS[val.replace("fs:", "")]
                            else:
                                unit_val = Literal(val)
                            g.add((FS[measure_id], NIDM.hasUnit, unit_val))

            # Add structures
            if "Structures" in fs_mapping:
                for struct_name, struct_info in fs_mapping["Structures"].items():
                    struct_id = f"struct_{safe_id(struct_name)}"
                    g.add((FS[struct_id], RDF.type, FS["Structure"]))
                    g.add((FS[struct_id], RDFS.label, Literal(struct_name)))

                    if "isAbout" in struct_info and struct_info["isAbout"]:
                        if not (
                            isinstance(struct_info["isAbout"], str)
                            and (
                                struct_info["isAbout"].startswith("CUSTOM")
                                or struct_info["isAbout"].startswith("<UNKNOWN")
                            )
                        ):
                            g.add(
                                (
                                    FS[struct_id],
                                    NIDM.isAbout,
                                    URIRef(struct_info["isAbout"]),
                                )
                            )

                    if "fskey" in struct_info:
                        for key in struct_info["fskey"]:
                            g.add((FS[struct_id], FS.fskey, Literal(key)))

                            # Handle hemispheric information
                            if key.startswith("Left-") or key.startswith("lh"):
                                g.add(
                                    (FS[struct_id], NIDM.hasLaterality, Literal("Left"))
                                )
                            elif key.startswith("Right-") or key.startswith("rh"):
                                g.add(
                                    (
                                        FS[struct_id],
                                        NIDM.hasLaterality,
                                        Literal("Right"),
                                    )
                                )

        except Exception as e:
            logger.warning(f"Error processing mapping: {e}")

    # Save the graph to turtle format for future use
    try:
        g.serialize(destination=FS_CDE_TTL_PATH, format="turtle")
        logger.info(f"Saved CDE graph to {FS_CDE_TTL_PATH}")
    except Exception as e:
        logger.warning(f"Error saving CDE graph: {e}")

    return g


def get_measure_details(measure_name, fs_mapping=None):
    """
    Get detailed information about a FreeSurfer measure.

    Parameters
    ----------
    measure_name : str
        Name of the FreeSurfer measure
    fs_mapping : dict, optional
        FreeSurfer mapping dictionary

    Returns
    -------
    dict
        Dictionary with measure details
    """
    if not fs_mapping:
        fs_mapping = load_fs_mapping()

    if not fs_mapping or "Measures" not in fs_mapping:
        return {}

    # Check for direct match
    if measure_name in fs_mapping["Measures"]:
        return fs_mapping["Measures"][measure_name]

    # Try some common variations
    variations = [
        measure_name.lower(),
        measure_name.upper(),
        measure_name.replace("-", "_"),
        measure_name.replace("_", "-"),
    ]

    for var in variations:
        if var in fs_mapping["Measures"]:
            return fs_mapping["Measures"][var]

    return {}


def get_structure_details(structure_name, fs_mapping=None):
    """
    Get detailed information about a FreeSurfer structure.

    Parameters
    ----------
    structure_name : str
        Name of the FreeSurfer structure
    fs_mapping : dict, optional
        FreeSurfer mapping dictionary

    Returns
    -------
    dict
        Dictionary with structure details
    """
    if not fs_mapping:
        fs_mapping = load_fs_mapping()

    if not fs_mapping or "Structures" not in fs_mapping:
        return {}

    # Check for direct match
    if structure_name in fs_mapping["Structures"]:
        return fs_mapping["Structures"][structure_name]

    # Check if this is in any structure's fskey list
    for struct_name, struct_info in fs_mapping["Structures"].items():
        if "fskey" in struct_info and structure_name in struct_info["fskey"]:
            return struct_info

    # Remove hemisphere prefix and try again
    clean_name = structure_name
    for prefix in ["Left-", "Right-", "lh-", "rh-"]:
        if structure_name.startswith(prefix):
            clean_name = structure_name[len(prefix) :]
            break

    if clean_name != structure_name and clean_name in fs_mapping["Structures"]:
        return fs_mapping["Structures"][clean_name]

    return {}


def convert_stats_to_nidm(stats_data):
    """
    Convert FreeSurfer stats to NIDM entities

    Parameters
    ----------
    stats_data : list
        List of tuples (id, value)

    Returns
    -------
    tuple
        (entity, document) pair
    """
    try:
        import prov.model as prov

        from nidm.core import Constants
        from nidm.experiment.Core import getUUID

        # Set up namespaces
        fs = prov.Namespace("fs", str(Constants.FREESURFER))
        niiri = prov.Namespace("niiri", str(Constants.NIIRI))
        nidm_ns = prov.Namespace("nidm", "http://purl.org/nidash/nidm#")

        # Create document
        doc = prov.ProvDocument()

        # Create entity
        entity_id = niiri[getUUID()]
        entity = doc.entity(identifier=entity_id)
        entity.add_asserted_type(nidm_ns["FSStatsCollection"])

        # Add attributes
        entity.add_attributes(
            {
                fs[f"fs_{val[0]}"]: prov.Literal(
                    val[1],
                    datatype=(
                        prov.XSD["float"] if "." in str(val[1]) else prov.XSD["integer"]
                    ),
                )
                for val in stats_data
            }
        )

        return entity, doc

    except ImportError:
        logger.warning("PyNIDM not available. Cannot convert stats to NIDM.")
        return None, None


def convert_csv_stats_to_nidm(row, var_to_cde_mapping, filename, id_column):
    """
    Convert CSV-based FreeSurfer stats to NIDM

    Parameters
    ----------
    row : pandas.Series
        Row of data from CSV
    var_to_cde_mapping : dict
        Mapping from variables to CDEs
    filename : str
        Source filename
    id_column : str
        Column containing subject IDs

    Returns
    -------
    tuple
        (entity, document) pair
    """
    try:
        import prov.model as prov

        from nidm.core import Constants
        from nidm.core.Constants import DD
        from nidm.experiment.Core import getUUID

        # Set up namespaces
        fs = prov.Namespace("fs", str(Constants.FREESURFER))
        niiri = prov.Namespace("niiri", str(Constants.NIIRI))
        nidm_ns = prov.Namespace("nidm", "http://purl.org/nidash/nidm#")
        nfo = prov.Namespace(
            "nfo", "http://www.semanticdesktop.org/ontologies/2007/03/22/nfo#"
        )

        # Create document
        doc = prov.ProvDocument()

        # Create entity
        entity_id = niiri[getUUID()]
        entity = doc.entity(identifier=entity_id)
        entity.add_asserted_type(nidm_ns["FSStatsCollection"])

        # Add filename
        entity.add_attributes({nfo["filename"]: prov.Literal(filename)})

        # Add measurements
        for colname, colval in row.items():
            if colname == id_column:
                continue

            if pd.isna(colval):
                continue

            # Create tuple for lookup
            current_tuple = str(DD(source=filename, variable=colname))

            if current_tuple in var_to_cde_mapping:
                cde_id = var_to_cde_mapping[current_tuple]["sameAs"].rsplit("/", 1)[-1]

                # Add value with appropriate datatype
                entity.add_attributes(
                    {
                        fs[cde_id]: prov.Literal(
                            colval,
                            datatype=(
                                prov.XSD["float"]
                                if "." in str(colval)
                                else prov.XSD["integer"]
                            ),
                        )
                    }
                )

        return entity, doc

    except ImportError:
        logger.warning("PyNIDM not available. Cannot convert CSV stats to NIDM.")
        return None, None


def stats_files_exist(fs_subject_dir):
    """
    Check if required FreeSurfer stats files exist for a subject.

    Parameters
    ----------
    fs_subject_dir : str or Path
        Path to subject's FreeSurfer directory

    Returns
    -------
    bool
        True if essential stats files exist, False otherwise
    """
    fs_subject_dir = Path(fs_subject_dir)
    stats_dir = fs_subject_dir / "stats"

    if not stats_dir.exists():
        return False

    # Check for at least one stats file (simplified check)
    return any(stats_dir.glob("*.stats"))


def get_fs_version_from_recon_log(fs_subject_dir):
    """
    Extract FreeSurfer version from recon-all.log file.

    Parameters
    ----------
    fs_subject_dir : str or Path
        Path to subject's FreeSurfer directory

    Returns
    -------
    str or None
        FreeSurfer version if found, None otherwise
    """
    fs_subject_dir = Path(fs_subject_dir)
    log_file = fs_subject_dir / "scripts" / "recon-all.log"

    if not log_file.exists():
        return None

    try:
        with open(log_file, "r") as f:
            content = f.read()

        # Look for version in log file using common patterns
        patterns = [
            r"FreeSurfer\s+[vV]ersion\s+([0-9.]+)",
            r"freesurfer.+?(\d+\.\d+\.\d+)",
            r"recon-all\s+v([0-9.]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, content)
            if match:
                return match.group(1)

        return None

    except Exception as e:
        logger.warning(f"Error reading recon-all.log: {e}")
        return None
