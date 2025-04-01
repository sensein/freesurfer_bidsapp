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
from src.nidm.utils import NIDM, NIIRI, FS

# Test data paths
TEST_DATA_DIR = Path(__file__).parent / "data"
FS_SUBJECT_DIR = TEST_DATA_DIR / "freesurfer-subjects-dir"
SUBJECT_ID = "0050663"
SESSION_LABEL = "01"  # BIDS session label

@pytest.fixture(scope="session", autouse=True)
def setup_test_data():
    """Set up test data directory structure."""
    # Use existing BIDS-formatted data structure
    subject_dir = FS_SUBJECT_DIR / f"sub-{SUBJECT_ID}"
    session_dir = subject_dir / f"ses-{SESSION_LABEL}"
    
    assert subject_dir.exists(), f"Subject directory not found: {subject_dir}"
    assert session_dir.exists(), f"Session directory not found: {session_dir}"
    
    # Verify required directories exist
    for subdir in ["stats", "mri", "surf", "scripts"]:
        assert (session_dir / subdir).exists(), f"Required directory not found: {subdir}"
    
    # Verify required files exist
    required_files = [
        "stats/aseg.stats",
        "stats/lh.aparc.stats",
        "stats/rh.aparc.stats",
        "mri/aparc+aseg.mgz",
        "mri/wmparc.mgz"
    ]
    
    for file in required_files:
        assert (session_dir / file).exists(), f"Required file not found: {file}"
    
    yield
    
    # No cleanup needed since we're using existing data

@pytest.fixture
def converter():
    """Create a FreeSurferToNIDM converter instance for testing."""
    return FreeSurferToNIDM(FS_SUBJECT_DIR, SUBJECT_ID, session_label=SESSION_LABEL)

def test_converter_initialization():
    """Test FreeSurferToNIDM initialization."""
    # Test with valid directory
    converter = FreeSurferToNIDM(FS_SUBJECT_DIR, SUBJECT_ID, session_label=SESSION_LABEL)
    assert converter.freesurfer_dir == FS_SUBJECT_DIR
    assert converter.subject_label == SUBJECT_ID
    assert converter.session_label == SESSION_LABEL
    assert converter.subject_id == SUBJECT_ID  # subject_id should not include session
    
    # Test without session label
    converter = FreeSurferToNIDM(FS_SUBJECT_DIR, SUBJECT_ID)
    assert converter.session_label is None
    assert converter.subject_id == SUBJECT_ID
    
    # Test with invalid directory
    with pytest.raises(ValueError):
        FreeSurferToNIDM("/nonexistent/dir", SUBJECT_ID)
    
    # Test with invalid subject
    with pytest.raises(ValueError):
        FreeSurferToNIDM(FS_SUBJECT_DIR, "nonexistent")

def test_basic_provenance(converter):
    """Test basic provenance information generation."""
    # Add basic provenance
    converter._add_basic_provenance()
    
    # Check software information
    assert (converter.fs_software, RDF.type, PROV.SoftwareAgent) in converter.graph
    assert (converter.fs_software, NIDM.label, Literal("FreeSurfer")) in converter.graph
    assert (converter.fs_software, NIDM.neuroimagingAnalysisSoftware, URIRef("http://surfer.nmr.mgh.harvard.edu/")) in converter.graph
    
    # Check version information (should be "unknown" if no version found)
    version = converter._get_fs_version()
    expected_version = version if version else "unknown"
    assert (converter.fs_software, DCTERMS.hasVersion, Literal(expected_version)) in converter.graph
    
    # Check subject information
    subject_uri = URIRef(f"http://iri.nidash.org/subject-{converter.subject_id}")
    assert (subject_uri, RDF.type, NIDM.Subject) in converter.graph
    assert (subject_uri, NIDM.label, Literal(converter.subject_id)) in converter.graph
    
    # Check session information
    if converter.session_label:
        session_uri = URIRef(f"http://iri.nidash.org/session-{converter.session_label}")
        assert (session_uri, RDF.type, NIDM.Session) in converter.graph
        assert (session_uri, NIDM.label, Literal(converter.session_label)) in converter.graph
        assert (subject_uri, URIRef("http://bids.neuroimaging.io/terms/session_id"), Literal(converter.session_label)) in converter.graph
        assert (session_uri, PROV.wasAssociatedWith, subject_uri) in converter.graph
    
    # Check process information
    assert (converter.fs_process, RDF.type, NIDM.FreeSurferAnalysis) in converter.graph
    assert (converter.fs_process, PROV.wasAssociatedWith, converter.fs_software) in converter.graph
    assert (converter.fs_process, PROV.used, subject_uri) in converter.graph
    if converter.session_label:
        assert (converter.fs_process, PROV.used, session_uri) in converter.graph

def test_stats_file_processing(converter):
    """Test processing of FreeSurfer stats files."""
    # Process stats files
    converter._process_stats_files()
    
    # Check that files were processed
    assert "aseg.stats" in converter.processed_files
    assert "lh.aparc.stats" in converter.processed_files
    assert "rh.aparc.stats" in converter.processed_files
    
    # Check that stats were added to graph
    stats_dir = converter.fs_subject_dir / "stats"
    aseg_file = stats_dir / "aseg.stats"
    if aseg_file.exists():
        container_id = f"aseg-{SUBJECT_ID}"
        container_uri = NIIRI[container_id]
        assert (container_uri, RDF.type, FS.SegmentationStatistics) in converter.graph

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
    
    # Check that we have a project instance
    projects = list(graph.subjects(RDF.type, NIDM.Project))
    assert len(projects) > 0, "Should have at least one project"
    
    # Check that we have a subject instance
    subjects = list(graph.subjects(RDF.type, NIDM.Subject))
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
    software = list(graph.subjects(RDF.type, NIDM.Software))[0]
    
    # Check that software has a version
    versions = list(graph.objects(software, DCTERMS.hasVersion))
    assert len(versions) == 1, "Software should have exactly one version"
    
    # Check software URI
    assert (software, NIDM.neuroimagingAnalysisSoftware, URIRef("http://surfer.nmr.mgh.harvard.edu/")) in graph
    
    # Check process information
    process = list(graph.subjects(RDF.type, NIDM.FreeSurferAnalysis))[0]
    
    # Debug: Print all triples with the process as subject
    print("\nProcess triples:")
    for s, p, o in graph.triples((process, None, None)):
        print(f"({s}, {p}, {o})")
    
    # Debug: Print all triples with the software as object
    print("\nSoftware triples:")
    for s, p, o in graph.triples((None, None, software)):
        print(f"({s}, {p}, {o})")
    
    # Check the specific triple
    assert (process, PROV.wasAssociatedWith, software) in graph, "Process should be associated with software"

def test_error_handling():
    """Test error handling for invalid input."""
    # Test with non-existent directory
    with pytest.raises(ValueError):
        FreeSurferToNIDM("/nonexistent/dir", SUBJECT_ID)
    
    # Test with invalid subject ID
    with pytest.raises(ValueError):
        FreeSurferToNIDM(FS_SUBJECT_DIR, "")

def test_data_validation(converter):
    """Test that the converted data contains expected values."""
    output_file = converter.convert()
    
    # Load the NIDM graph
    graph = Graph()
    graph.parse(output_file, format="json-ld")
    
    # Check subject information
    subject = list(graph.subjects(RDF.type, NIDM.Subject))[0]
    assert (subject, NIDM.label, Literal(SUBJECT_ID)) in graph
    
    # Check session information if present
    if SESSION_LABEL:
        session = list(graph.subjects(RDF.type, NIDM.Session))[0]
        assert (session, NIDM.label, Literal(SESSION_LABEL)) in graph
    
    # Check project information
    project = list(graph.subjects(RDF.type, NIDM.Project))[0]
    assert (project, NIDM.label, Literal("FreeSurfer Analysis")) in graph

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
    """Test handling of session information in BIDS output."""
    # Test with session label
    converter = FreeSurferToNIDM(FS_SUBJECT_DIR, SUBJECT_ID, session_label=SESSION_LABEL)
    output_file = converter.convert()
    
    # Load the NIDM graph
    graph = Graph()
    graph.parse(output_file, format="json-ld")
    
    # Check that subject exists with correct label
    subject = list(graph.subjects(RDF.type, NIDM.Subject))[0]
    assert (subject, NIDM.label, Literal(SUBJECT_ID)) in graph
    
    # Check that session exists with correct label
    session = list(graph.subjects(RDF.type, NIDM.Session))[0]
    assert (session, NIDM.label, Literal(SESSION_LABEL)) in graph
    
    # Test without session label
    converter = FreeSurferToNIDM(FS_SUBJECT_DIR, SUBJECT_ID)
    output_file = converter.convert()
    
    # Load the NIDM graph
    graph = Graph()
    graph.parse(output_file, format="json-ld")
    
    # Check that subject exists with correct label
    subject = list(graph.subjects(RDF.type, NIDM.Subject))[0]
    assert (subject, NIDM.label, Literal(SUBJECT_ID)) in graph
    
    # Check that no session exists
    sessions = list(graph.subjects(RDF.type, NIDM.Session))
    assert len(sessions) == 0, "Should not have any sessions when no session label is provided"

def test_custom_output_directory():
    """Test using a custom output directory."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        converter = FreeSurferToNIDM(FS_SUBJECT_DIR, SUBJECT_ID, session_label=SESSION_LABEL, output_dir=tmp_dir)
        output_file = converter.convert()
        
        # Check output file is in custom directory
        assert str(output_file).startswith(tmp_dir)
        assert os.path.exists(output_file)

def test_subject_id_handling():
    """Test handling of subject IDs with and without 'sub-' prefix."""
    # Test with subject ID without prefix
    converter1 = FreeSurferToNIDM(FS_SUBJECT_DIR, SUBJECT_ID)
    assert converter1.subject_id == SUBJECT_ID
    
    # Test with subject ID with prefix
    converter2 = FreeSurferToNIDM(FS_SUBJECT_DIR, f"sub-{SUBJECT_ID}")
    assert converter2.subject_id == SUBJECT_ID
    
    # Test that both converters point to the same directory
    assert converter1.fs_subject_dir == converter2.fs_subject_dir 

def test_version_tracking(converter):
    """Test version tracking in NIDM output."""
    # Convert to NIDM
    output_file = converter.convert()
    
    # Load the NIDM graph
    graph = Graph()
    graph.parse(output_file, format="json-ld")
    
    # Get version information from the graph
    from src.utils import get_version_info
    version_info = get_version_info()
    
    # Check FreeSurfer software version
    fs_software = list(graph.subjects(RDF.type, PROV.SoftwareAgent))[0]
    assert (fs_software, DCTERMS.hasVersion, Literal(version_info["freesurfer"]["version"])) in graph
    
    # Check software source
    if version_info["freesurfer"]["source"] != "unknown":
        assert (fs_software, NIDM.versionSource, Literal(version_info["freesurfer"]["source"])) in graph
    
    # Check build stamp if available
    if version_info["freesurfer"]["build_stamp"]:
        assert (fs_software, FS.buildStamp, Literal(version_info["freesurfer"]["build_stamp"])) in graph
    
    # Check container image if available
    if version_info["freesurfer"]["image"]:
        assert (fs_software, FS.containerImage, Literal(version_info["freesurfer"]["image"])) in graph
    
    # Check BIDS-FreeSurfer version
    bids_software = list(graph.subjects(RDF.type, PROV.SoftwareAgent))[1]  # Second software agent
    assert (bids_software, DCTERMS.hasVersion, Literal(version_info["bids_freesurfer"]["version"])) in graph
    assert (bids_software, NIDM.versionSource, Literal(version_info["bids_freesurfer"]["source"])) in graph
    
    # Check Python environment
    env = list(graph.subjects(RDF.type, PROV.Location))[0]
    assert (env, NIDM.pythonVersion, Literal(version_info["python"]["version"])) in graph
    
    # Check Python package versions
    for package, version in version_info["python"]["packages"].items():
        assert (env, NIDM.packageVersion, Literal(f"{package}:{version}")) in graph 