#!/usr/bin/env python3
"""
Utility functions for FreeSurfer to NIDM conversion

This module provides helper functions for parsing FreeSurfer outputs,
handling RDF graphs, and generating NIDM-compatible identifiers.
"""

import os
import re
import json
import logging
from pathlib import Path
from rdflib import Graph, URIRef, Namespace

# Configure logging
logger = logging.getLogger('bids-freesurfer.nidm.utils')

# Define common namespaces
NIDM = Namespace("http://purl.org/nidash/nidm#")
NIIRI = Namespace("http://iri.nidash.org/")
FS = Namespace("http://surfer.nmr.mgh.harvard.edu/fs/terms/")


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
    safe = re.sub(r'[^a-zA-Z0-9_]', '_', str(text)).lower()
    
    # Ensure it doesn't start with a number
    if safe and safe[0].isdigit():
        safe = 'n' + safe
        
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
    
    result = {
        'global_measures': {},
        'structures': []
    }
    
    try:
        with open(stats_file, 'r') as f:
            lines = f.readlines()
        
        # Extract global measures (# Measure lines)
        for line in lines:
            if line.startswith('# Measure'):
                parts = line.strip().split(',')
                if len(parts) >= 4:
                    name = parts[1].strip()
                    try:
                        value = float(parts[3].strip())
                        result['global_measures'][name] = value
                    except ValueError:
                        logger.debug(f"Could not convert measure value to float: {parts[3]}")
        
        # Find the column headers
        header_idx = None
        for i, line in enumerate(lines):
            if line.startswith('# ColHeaders'):
                header_idx = i
                break
        
        # Extract column data if headers are found
        if header_idx is not None:
            headers = lines[header_idx].replace('# ColHeaders', '').strip().split()
            
            # Process data rows
            for line in lines[header_idx + 1:]:
                if line.startswith('#') or not line.strip():
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
                
                result['structures'].append(struct_data)
        else:
            # For files without column headers (like aseg.stats)
            # Extract volume data directly
            data_lines = [line for line in lines if not line.startswith('#') and len(line.strip()) > 0]
            
            for line in data_lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    # Assumes standard aseg.stats format
                    struct_data = {
                        'Index': int(parts[0]),
                        'SegId': int(parts[1]),
                        'NVoxels': int(parts[2]),
                        'Volume_mm3': float(parts[3]),
                        'StructName': parts[4],
                    }
                    
                    # Add normalized volume if ICV is available
                    if 'EstimatedTotalIntraCranialVol' in result['global_measures']:
                        icv = result['global_measures']['EstimatedTotalIntraCranialVol']
                        if icv > 0:
                            struct_data['Volume_Normalized'] = (struct_data['Volume_mm3'] / icv) * 100
                    
                    result['structures'].append(struct_data)
        
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
    if not aseg_data or 'structures' not in aseg_data:
        return {}
    
    structures = {}
    
    for struct in aseg_data['structures']:
        if 'StructName' in struct and 'Volume_mm3' in struct:
            name = struct['StructName']
            structures[name] = {
                'volume': struct['Volume_mm3'],
                'index': struct.get('Index', None),
                'nvoxels': struct.get('NVoxels', None),
                'normalized_volume': struct.get('Volume_Normalized', None)
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
    if not aparc_data or 'structures' not in aparc_data:
        return {}
    
    measures = {
        'hemisphere': hemisphere,
        'regions': {}
    }
    
    # Add global measures
    if 'global_measures' in aparc_data:
        measures['global'] = aparc_data['global_measures']
    
    # Process each region
    for region in aparc_data['structures']:
        if 'StructName' in region:
            name = region['StructName']
            region_data = {}
            
            # Extract common metrics
            metrics = ['NumVert', 'SurfArea', 'GrayVol', 'ThickAvg', 'ThickStd', 'MeanCurv', 'GausCurv']
            for metric in metrics:
                if metric in region:
                    region_data[metric] = region[metric]
            
            measures['regions'][name] = region_data
    
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
        module_dir = Path(__file__).parent
        mapping_file = module_dir / 'mappings' / 'fsmap.json'
    
    # If mapping file exists, load it
    if os.path.exists(mapping_file):
        try:
            with open(mapping_file, 'r') as f:
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
        Category to restrict mapping search (e.g., 'region', 'metric')
    
    Returns
    -------
    str or None
        Mapped term if found, otherwise None
    """
    if not mapping_dict or not term:
        return None
    
    # Normalize the term for lookup
    norm_term = safe_id(term)
    
    # Check if term exists in mapping
    if norm_term in mapping_dict:
        entry = mapping_dict[norm_term]
        
        # If category specified, check category
        if category and 'category' in entry and entry['category'] != category:
            return None
        
        # Return the standard term
        return entry.get('standard_term', None)
    
    return None


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
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Look for version in log file using common patterns
        patterns = [
            r'FreeSurfer\s+[vV]ersion\s+([0-9.]+)',
            r'freesurfer.+?(\d+\.\d+\.\d+)',
            r'recon-all\s+v([0-9.]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content)
            if match:
                return match.group(1)
        
        return None
        
    except Exception as e:
        logger.warning(f"Error reading recon-all.log: {e}")
        return None