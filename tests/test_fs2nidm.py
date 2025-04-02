#!/usr/bin/env python3
"""
Tests for the FreeSurfer to NIDM conversion.
"""

import json
import os
import pytest
from pathlib import Path
from rdflib import Graph, Literal, RDF, URIRef, RDFS
from rdflib.namespace import DCTERMS, PROV, XSD

from src.nidm.fs2nidm import FreeSurferToNIDM
from src.nidm.utils import NIDM, NIIRI, FS, SIO, NDAR

# Test data paths
TEST_DATA_DIR = Path(__file__).parent / "data"
BIDS_DIR = TEST_DATA_DIR / "bids"  # Add BIDS directory
FS_DIR = TEST_DATA_DIR / "derivatives" / "freesurfer"
SUBJECT_ID = "sub-0050663"  # Include sub- prefix

@pytest.fixture(scope="session", autouse=True)
def setup_test_data():
    """Set up test data directory structure."""
    # Create BIDS directory structure
    bids_subj_dir = BIDS_DIR / SUBJECT_ID
    bids_subj_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dataset_description.json - required for BIDS validation
    dataset_description = {
        "Name": "Test BIDS Dataset",
        "BIDSVersion": "1.8.0",
        "DatasetType": "raw"
    }
    BIDS_DIR.mkdir(parents=True, exist_ok=True)
    with open(BIDS_DIR / "dataset_description.json", "w") as f:
        json.dump(dataset_description, f, indent=2)
    
    # Create FreeSurfer directory structure
    fs_subj_dir = FS_DIR / SUBJECT_ID
    for subdir in ["stats", "mri", "surf", "scripts"]:
        (fs_subj_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    # Create required files
    required_files = [
        "stats/aseg.stats",
        "stats/lh.aparc.stats",
        "stats/rh.aparc.stats",
        "mri/aparc+aseg.mgz",
        "mri/wmparc.mgz"
    ]
    
    for file in required_files:
        file_path = fs_subj_dir / file
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.touch()
    
    # Create dummy T1w file in BIDS directory (needed for validation)
    anat_dir = bids_subj_dir / "anat"
    anat_dir.mkdir(exist_ok=True)
    (anat_dir / f"{SUBJECT_ID}_T1w.nii.gz").touch()
    
    yield

@pytest.fixture
def converter():
    """Create a FreeSurferToNIDM converter instance for testing."""
    return FreeSurferToNIDM(
        freesurfer_dir=FS_DIR,
        bids_dir=BIDS_DIR,  # Required argument
        subject_id=SUBJECT_ID
    )

def test_converter_initialization():
    """Test FreeSurferToNIDM initialization."""
    # Test with valid inputs
    converter = FreeSurferToNIDM(
        freesurfer_dir=FS_DIR,
        bids_dir=BIDS_DIR,
        subject_id=SUBJECT_ID
    )
    assert converter.freesurfer_dir == FS_DIR
    assert converter.subject_id == SUBJECT_ID
    
    # Test with invalid FreeSurfer directory
    with pytest.raises(ValueError):
        FreeSurferToNIDM(
            freesurfer_dir="/nonexistent/dir",
            bids_dir=BIDS_DIR,
            subject_id=SUBJECT_ID
        )
    
    # Test with subject ID missing sub- prefix
    with pytest.raises(ValueError):
        FreeSurferToNIDM(
            freesurfer_dir=FS_DIR,
            bids_dir=BIDS_DIR,
            subject_id="0050663"  # Missing sub- prefix
        )

def test_basic_provenance(converter):
    """Test basic provenance information generation."""
    # Add basic provenance
    converter._add_basic_provenance()
    
    # Check software agent information
    assert (converter.fs_software, RDF.type, PROV.SoftwareAgent) in converter.graph
    assert (converter.fs_software, RDF.type, PROV.Agent) in converter.graph
    assert (converter.fs_software, NIDM['softwareName'], Literal("FreeSurfer")) in converter.graph
    
    # Get version from get_version_info
    from src.utils import get_version_info
    version_info = get_version_info()
    expected_version = version_info["freesurfer"]["version"]
    
    # Check version information
    assert (converter.fs_software, NIDM['softwareVersion'], 
            Literal(expected_version)) in converter.graph
    
    # Check subject information - updated to use PROV.Person
    assert (converter.subject_uri, RDF.type, PROV.Agent) in converter.graph
    assert (converter.subject_uri, RDF.type, PROV.Person) in converter.graph
    assert (converter.subject_uri, NDAR.src_subject_id, 
            Literal(converter.subject_id)) in converter.graph
    
    # Check process information
    assert (converter.fs_process, RDF.type, PROV.Activity) in converter.graph
    assert (converter.fs_process, RDF.type, NIDM.FreeSurferProcessing) in converter.graph
    
    # Check qualified associations
    # Find the association nodes
    software_association = None
    subject_association = None
    for s, p, o in converter.graph.triples((None, RDF.type, PROV.Association)):
        if (s, PROV.agent, converter.fs_software) in converter.graph:
            software_association = s
        elif (s, PROV.agent, converter.subject_uri) in converter.graph:
            subject_association = s
    
    assert software_association is not None, "Software association not found"
    assert subject_association is not None, "Subject association not found"
    
    # Check software association
    assert (converter.fs_process, PROV.qualifiedAssociation, software_association) in converter.graph
    assert (software_association, PROV.hadRole, NIDM.SoftwareAgent) in converter.graph
    
    # Check subject association
    assert (converter.fs_process, PROV.qualifiedAssociation, subject_association) in converter.graph
    assert (subject_association, PROV.hadRole, SIO.Subject) in converter.graph

def test_stats_file_processing(converter):
    """Test processing of FreeSurfer stats files."""
    # Process stats files
    converter._process_stats_files()
    
    # Check that stats were added to graph
    container_uri = NIIRI[f"aseg-{converter.subject_id}"]
    assert (container_uri, RDF.type, FS.SegmentationStatistics) in converter.graph
    assert (container_uri, PROV.wasGeneratedBy, converter.fs_process) in converter.graph

def test_mri_file_processing(converter):
    """Test processing of FreeSurfer MRI volume files."""
    # Process MRI files
    converter._process_mri_files()
    
    # Check that files were processed
    assert "aparc+aseg.mgz" in converter.processed_files
    assert "wmparc.mgz" in converter.processed_files
    
    # Check that volumes were added to graph
    mri_dir = converter.fs_subject_dir / "mri"
    aparc_file = mri_dir / "aparc+aseg.mgz"
    if aparc_file.exists():
        container_id = f"mri-aparc+aseg-{SUBJECT_ID}"
        container_uri = NIIRI[container_id]
        assert (container_uri, RDF.type, FS.MRIVolume) in converter.graph

def test_jsonld_output(converter):
    """Test JSON-LD output generation."""
    # Convert to NIDM
    output_file = converter.convert()
    
    # Check output file exists
    assert os.path.exists(output_file)
    
    # Load and validate JSON-LD
    with open(output_file) as f:
        data = json.load(f)
    
    # Check required fields
    assert "@context" in data
    assert "@graph" in data
    assert isinstance(data["@graph"], list)
    
    # Check context contains required namespaces
    context = data["@context"]
    assert "nidm" in context
    assert "niiri" in context
    assert "prov" in context
    assert "dcterms" in context
    assert "xsd" in context

def test_basic_conversion(converter):
    """Test basic conversion of a FreeSurfer subject directory."""
    # Convert subject
    output_file = converter.convert()
    
    # Check output file exists
    assert os.path.exists(output_file)
    
    # Load the NIDM graph
    graph = Graph()
    graph.parse(output_file, format="json-ld")
    
    # Check basic structure
    assert len(graph) > 0
    
    # Check that we have a subject instance
    subjects = list(graph.subjects(RDF.type, PROV.Person))
    assert len(subjects) > 0, "Should have at least one subject"

def test_stats_files_processing(converter):
    """Test processing of FreeSurfer stats files."""
    output_file = converter.convert()
    
    # Load the NIDM graph
    graph = Graph()
    graph.parse(output_file, format="json-ld")
    
    # Check stats files are included
    stats_files = list(graph.objects(predicate=NIDM.hasFile))
    assert any("aseg.stats" in str(f) for f in stats_files)
    assert any("lh.aparc.stats" in str(f) for f in stats_files)
    assert any("rh.aparc.stats" in str(f) for f in stats_files)
    
    # Also check that the files were processed
    assert "aseg.stats" in converter.processed_files
    assert "lh.aparc.stats" in converter.processed_files
    assert "rh.aparc.stats" in converter.processed_files

def test_mri_files_processing(converter):
    """Test processing of FreeSurfer MRI volume files."""
    output_file = converter.convert()
    
    # Load the NIDM graph
    graph = Graph()
    graph.parse(output_file, format="json-ld")
    
    # Check MRI files are included
    mri_files = list(graph.objects(predicate=NIDM.hasFile))
    assert any("aparc+aseg.mgz" in str(f) for f in mri_files)
    assert any("wmparc.mgz" in str(f) for f in mri_files)
    
    # Also check that the files were processed
    assert "aparc+aseg.mgz" in converter.processed_files
    assert "wmparc.mgz" in converter.processed_files

def test_provenance_information(converter):
    """Test that provenance information is correctly captured."""
    output_file = converter.convert()
    
    # Load the NIDM graph
    graph = Graph()
    graph.parse(output_file, format="json-ld")
    
    # Check software information
    software = list(graph.subjects(RDF.type, PROV.SoftwareAgent))[0]
    
    # Check that software has a version
    versions = list(graph.objects(software, NIDM['softwareVersion']))
    assert len(versions) == 1, "Software should have exactly one version"
    
    # Check software URI
    assert (software, NIDM.neuroimagingAnalysisSoftware, 
            URIRef("http://surfer.nmr.mgh.harvard.edu/")) in graph
    
    # Check process information
    processes = list(graph.subjects(RDF.type, NIDM.FreeSurferProcessing))
    assert len(processes) > 0, "No FreeSurfer processing found in graph"
    process = processes[0]
    
    # Check qualified associations
    associations = list(graph.objects(process, PROV.qualifiedAssociation))
    assert len(associations) >= 2, "Should have at least two qualified associations"
    
    # Find software association
    software_association = None
    for assoc in associations:
        if (assoc, PROV.agent, software) in graph:
            software_association = assoc
            break
    
    assert software_association is not None, "No qualified association with software found"
    assert (software_association, PROV.hadRole, NIDM.SoftwareAgent) in graph
    

def test_error_handling():
    """Test error handling for invalid input."""
    # Test with non-existent directory
    with pytest.raises(ValueError):
        FreeSurferToNIDM(
            freesurfer_dir="/nonexistent/dir",
            bids_dir=BIDS_DIR,
            subject_id="sub-001"
        )
    
    # Test with invalid subject ID (missing sub- prefix)
    with pytest.raises(ValueError, match="Subject ID must include 'sub-' prefix"):
        FreeSurferToNIDM(
            freesurfer_dir=FS_DIR,
            bids_dir=BIDS_DIR,
            subject_id="001"
        )

def test_data_validation(converter):
    """Test that the converted data contains expected values."""
    output_file = converter.convert()
    
    # Load the NIDM graph
    graph = Graph()
    graph.parse(output_file, format="json-ld")
    
    # Check subject information - updated to use PROV.Person
    subjects = list(graph.subjects(RDF.type, PROV.Person))
    assert len(subjects) == 1, "Should have exactly one subject"
    subject = subjects[0]
    assert (subject, NDAR.src_subject_id, Literal(SUBJECT_ID)) in graph
    
    # Check software agent
    software_agents = list(graph.subjects(RDF.type, PROV.SoftwareAgent))
    assert len(software_agents) == 1, "Should have exactly one software agent"
    software = software_agents[0]
    assert (software, NIDM['softwareName'], Literal("FreeSurfer")) in graph
    
    # Check process information
    processes = list(graph.subjects(RDF.type, NIDM.FreeSurferProcessing))
    assert len(processes) == 1, "Should have exactly one FreeSurfer process"
    process = processes[0]
    
    # Check qualified associations
    associations = list(graph.objects(process, PROV.qualifiedAssociation))
    assert len(associations) >= 2, "Should have at least two qualified associations"

def test_output_format(converter):
    """Test that the output format is valid JSON-LD."""
    output_file = converter.convert()
    
    # Try to load as JSON
    with open(output_file) as f:
        data = json.load(f)
    
    # Check required fields
    assert "@context" in data
    assert "@graph" in data
    assert isinstance(data["@graph"], list)
    
    # Check context contains required namespaces
    context = data["@context"]
    assert "nidm" in context
    assert "niiri" in context
    assert "prov" in context
    assert "dcterms" in context
    assert "xsd" in context

def test_file_tracking(converter):
    """Test that all files are properly tracked."""
    output_file = converter.convert()
    
    # Load the NIDM graph
    graph = Graph()
    graph.parse(output_file, format="json-ld")
    
    # Get all files
    files = list(graph.objects(predicate=NIDM.hasFile))
    
    # Check that all expected files are present
    expected_files = [
        "aseg.stats",
        "lh.aparc.stats",
        "rh.aparc.stats",
        "aparc+aseg.mgz",
        "wmparc.mgz"
    ]
    
    for expected in expected_files:
        assert any(expected in str(f) for f in files)

def test_session_handling():
    """Test session information handling from BIDS dataset."""
    # Create test session in BIDS directory with proper BIDS structure
    session = "ses-01"
    subject_dir = BIDS_DIR / SUBJECT_ID
    session_dir = subject_dir / session
    session_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a dummy T1w file in the session directory (needed for BIDS validation)
    anat_dir = session_dir / "anat"
    anat_dir.mkdir(parents=True, exist_ok=True)
    (anat_dir / f"{SUBJECT_ID}_{session}_T1w.nii.gz").touch()
    
    # Initialize converter
    converter = FreeSurferToNIDM(
        freesurfer_dir=FS_DIR,
        bids_dir=BIDS_DIR,
        subject_id=SUBJECT_ID
    )
    
    # Check that session was detected from BIDS dataset
    assert converter.session_label == "01"

def test_custom_output_directory():
    """Test using a custom output directory."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        converter = FreeSurferToNIDM(
            freesurfer_dir=FS_DIR,
            bids_dir=BIDS_DIR,
            subject_id=SUBJECT_ID,
            output_dir=tmp_dir
        )
        output_file = converter.convert()
        
        # Check output file is in custom directory
        assert str(output_file).startswith(tmp_dir)
        assert os.path.exists(output_file)

def test_subject_id_handling():
    """Test handling of subject IDs with and without 'sub-' prefix."""
    # Test with subject ID with required sub- prefix
    converter = FreeSurferToNIDM(
        freesurfer_dir=FS_DIR,
        bids_dir=BIDS_DIR,
        subject_id=SUBJECT_ID  # Already includes sub- prefix
    )
    
    # Verify it's stored correctly
    assert converter.subject_id == SUBJECT_ID
    
    # Test that invalid subject ID raises error
    with pytest.raises(ValueError):
        FreeSurferToNIDM(
            freesurfer_dir=FS_DIR,
            bids_dir=BIDS_DIR,
            subject_id="0050663"  # Missing sub- prefix
        )

def test_version_tracking(converter):
    """Test version tracking in NIDM output."""
    output_file = converter.convert()
    
    # Load the NIDM graph
    graph = Graph()
    graph.parse(output_file, format="json-ld")
    
    # Get version information from the graph
    from src.utils import get_version_info
    version_info = get_version_info()
    
    # Check FreeSurfer software version - only check essential information
    fs_software = list(graph.subjects(RDF.type, PROV.SoftwareAgent))[0]
    
    # Check essential version information
    assert (fs_software, NIDM['softwareVersion'], 
            Literal(version_info["freesurfer"]["version"])) in graph
    
    # Check software name
    assert (fs_software, NIDM['softwareName'], Literal("FreeSurfer")) in graph
    
    # Check software URI
    assert (fs_software, NIDM.neuroimagingAnalysisSoftware, 
            URIRef("http://surfer.nmr.mgh.harvard.edu/")) in graph

def test_helper_methods(converter):
    """Test the helper methods for adding entities and associations."""
    # Test _add_entity
    test_uri = NIIRI["test-entity"]
    converter._add_entity(
        test_uri,
        types=[PROV.Entity, NIDM.Measurement],
        labels=["Test Entity"]
    )
    
    assert (test_uri, RDF.type, PROV.Entity) in converter.graph
    assert (test_uri, RDF.type, NIDM.Measurement) in converter.graph
    assert (test_uri, RDFS.label, Literal("Test Entity")) in converter.graph
    
    # Test _add_qualified_association
    activity_uri = NIIRI["test-activity"]
    agent_uri = NIIRI["test-agent"]
    role_uri = NIDM["TestRole"]
    
    bnode = converter._add_qualified_association(
        activity=activity_uri,
        agent=agent_uri,
        role=role_uri
    )
    
    assert (activity_uri, PROV.qualifiedAssociation, bnode) in converter.graph
    assert (bnode, RDF.type, PROV.Association) in converter.graph
    assert (bnode, PROV.hadRole, role_uri) in converter.graph
    assert (bnode, PROV.agent, agent_uri) in converter.graph 