import os
import json
import tempfile
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock
import sys

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

# Import the module to test
from run import (
    create_dataset_description, 
    get_freesurfer_version, 
    create_provenance,
    create_nidm_output,
    add_stats_to_graph,
    get_subjects_to_analyze
)


def test_create_dataset_description():
    """Test creation of dataset_description.json file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        create_dataset_description(tmpdir, "0.1.0")
        
        # Check if file exists
        desc_file = os.path.join(tmpdir, 'dataset_description.json')
        assert os.path.exists(desc_file)
        
        # Check content
        with open(desc_file, 'r') as f:
            data = json.load(f)
        
        assert data["Name"] == "FreeSurfer Output"
        assert data["BIDSVersion"] == "1.8.0"
        assert data["DatasetType"] == "derivative"
        assert len(data["GeneratedBy"]) == 2
        assert data["GeneratedBy"][0]["Name"] == "BIDS-FreeSurfer"
        assert data["GeneratedBy"][0]["Version"] == "0.1.0"


@patch('subprocess.check_output')
def test_get_freesurfer_version(mock_check_output):
    """Test getting FreeSurfer version."""
    # Test FreeSurfer 8.0.0 detection
    mock_check_output.return_value = b"FreeSurfer 8.0.0\nrecon-all\n"
    version = get_freesurfer_version()
    assert version == "8.0.0"
    
    # Test older FreeSurfer format
    mock_check_output.return_value = b"recon-all v7.3.2 (July 22, 2022)\n"
    version = get_freesurfer_version()
    assert version == "7.3.2"
    
    # Test error handling
    mock_check_output.side_effect = Exception("Command failed")
    version = get_freesurfer_version()
    assert version == "8.0.0"  # Default to 8.0.0 in our streamlined version


def test_create_provenance():
    """Test creation of provenance files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        t1w_files = ["/data/sub-01/anat/sub-01_T1w.nii.gz"]
        t2w_files = ["/data/sub-01/anat/sub-01_T2w.nii.gz"]
        
        create_provenance("01", None, t1w_files, t2w_files, tmpdir)
        
        # Check if provenance file exists
        prov_file = os.path.join(tmpdir, "provenance.json")
        assert os.path.exists(prov_file)
        
        # Check content
        with open(prov_file, 'r') as f:
            data = json.load(f)
        
        assert "Sources" in data
        assert len(data["Sources"]) == 2
        assert "sub-01_T1w.nii.gz" in data["Sources"]
        assert "SoftwareVersion" in data
        assert "CommandLine" in data
        assert "DateProcessed" in data


@patch('rdflib.Graph')
def test_create_nidm_output(mock_graph_class):
    """Test creation of NIDM output."""
    # Mock the RDF graph
    mock_graph = MagicMock()
    mock_graph_class.return_value = mock_graph
    
    with tempfile.TemporaryDirectory() as tmpdir:
        freesurfer_dir = os.path.join(tmpdir, 'freesurfer')
        nidm_dir = os.path.join(tmpdir, 'nidm')
        
        os.makedirs(os.path.join(freesurfer_dir, 'sub-01', 'stats'), exist_ok=True)
        
        # Create a mock stats file
        with open(os.path.join(freesurfer_dir, 'sub-01', 'stats', 'aseg.stats'), 'w') as f:
            f.write("# Header\n")
            f.write("1 10 11 123.4 Left-Lateral-Ventricle\n")
        
        create_nidm_output("01", freesurfer_dir, nidm_dir)
        
        # Check if directory was created
        assert os.path.exists(os.path.join(nidm_dir, 'sub-01'))
        
        # Check if serialize was called
        mock_graph.serialize.assert_called_once()


def test_add_stats_to_graph():
    """Test adding FreeSurfer stats to RDF graph."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock stats file
        stats_file = os.path.join(tmpdir, 'aseg.stats')
        with open(stats_file, 'w') as f:
            f.write("# Volume information\n")
            f.write("# Measure, Value\n")
            f.write("1 10 11 123.4 Left-Lateral-Ventricle\n")
            f.write("2 20 21 456.7 Right-Lateral-Ventricle\n")
        
        # Create mock graph and namespace
        graph = MagicMock()
        subject_uri = MagicMock()
        fs_namespace = MagicMock()
        
        # Call function
        add_stats_to_graph(graph, stats_file, subject_uri, fs_namespace)
        
        # Check that add was called for each structure found
        assert graph.add.call_count >= 6  # At least 6 calls for 2 structures (3 triples each)


def test_get_subjects_to_analyze():
    """Test getting subjects to analyze."""
    # Mock BIDSLayout
    mock_layout = MagicMock()
    mock_layout.get_subjects.return_value = ['01', '02', '03']
    
    # Test with specific participant labels
    subjects = get_subjects_to_analyze(mock_layout, ['01', '03'])
    assert subjects == ['sub-01', 'sub-03']
    
    # Test with no participant labels (all subjects)
    subjects = get_subjects_to_analyze(mock_layout, [])
    assert subjects == ['sub-01', 'sub-02', 'sub-03']
    mock_layout.get_subjects.assert_called_once()


if __name__ == '__main__':
    pytest.main(['-xvs', __file__])