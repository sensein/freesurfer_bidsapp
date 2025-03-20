"""
Tests for the BIDS App FreeSurfer run.py script.

This test suite covers the main functionality of the run.py script
including subject selection, session selection, and participant and
group level processing.
"""

import os
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call

# Import the functions to test
from src.run import (
    main,
    get_subjects_to_analyze,
    get_sessions_to_analyze,
    run_participant_level,
    run_group_level,
    copy_freesurfer_outputs
)

# Define fixtures and mock data for testing
@pytest.fixture
def mock_bids_layout():
    """Create a mock BIDS layout for testing."""
    mock_layout = MagicMock()
    mock_layout.get_subjects.return_value = ["001", "002", "003"]
    mock_layout.get_sessions.return_value = ["01", "02"]
    return mock_layout

@pytest.fixture
def mock_freesurfer_wrapper():
    """Create a mock FreeSurferWrapper for testing."""
    mock_wrapper = MagicMock()
    mock_wrapper.process_subject.return_value = True
    mock_wrapper._find_t1w_images.return_value = ["sub-001/anat/sub-001_T1w.nii.gz"]
    mock_wrapper._find_t2w_images.return_value = []
    mock_wrapper.fs_options = "--parallel"
    return mock_wrapper

@pytest.fixture
def mock_bids_provenance():
    """Create a mock BIDSProvenance for testing."""
    mock_provenance = MagicMock()
    return mock_provenance

@pytest.fixture
def test_directories(tmp_path):
    """Create test directories for outputs."""
    dirs = {
        "bids_dir": tmp_path / "bids",
        "output_dir": tmp_path / "output",
        "freesurfer_dir": tmp_path / "output" / "freesurfer",
        "nidm_dir": tmp_path / "output" / "nidm"
    }
    
    # Create the directories
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
        
    # Create a simple FreeSurfer output directory structure
    subject_dir = dirs["freesurfer_dir"] / "001"
    for subdir in ["mri", "surf", "stats", "label", "scripts"]:
        (subject_dir / subdir).mkdir(parents=True, exist_ok=True)
        
    # Create some mock files
    (subject_dir / "mri" / "T1.mgz").touch()
    (subject_dir / "mri" / "brain.mgz").touch()
    (subject_dir / "surf" / "lh.white").touch()
    (subject_dir / "surf" / "rh.white").touch()
    (subject_dir / "stats" / "aseg.stats").touch()
    (subject_dir / "scripts" / "recon-all.log").touch()
        
    return dirs


# Test the get_subjects_to_analyze function
def test_get_subjects_to_analyze_with_labels(mock_bids_layout):
    """Test subject selection with specific labels."""
    participant_label = ["001", "002"]
    subjects = get_subjects_to_analyze(mock_bids_layout, participant_label)
    assert subjects == ["sub-001", "sub-002"]
    
def test_get_subjects_to_analyze_all(mock_bids_layout):
    """Test subject selection without specific labels (all subjects)."""
    subjects = get_subjects_to_analyze(mock_bids_layout, [])
    assert subjects == ["sub-001", "sub-002", "sub-003"]
    
def test_get_subjects_to_analyze_missing_subjects(mock_bids_layout):
    """Test subject selection with non-existent subjects."""
    participant_label = ["001", "999"]  # 999 doesn't exist
    subjects = get_subjects_to_analyze(mock_bids_layout, participant_label)
    assert subjects == ["sub-001"]  # Only 001 should be included

# Test the get_sessions_to_analyze function
def test_get_sessions_to_analyze_with_labels(mock_bids_layout):
    """Test session selection with specific labels."""
    session_label = ["01"]
    sessions = get_sessions_to_analyze(mock_bids_layout, session_label)
    assert sessions == ["01"]
    
def test_get_sessions_to_analyze_all(mock_bids_layout):
    """Test session selection without specific labels (all sessions)."""
    sessions = get_sessions_to_analyze(mock_bids_layout, [])
    assert sessions == ["01", "02"]
    
def test_get_sessions_to_analyze_missing_sessions(mock_bids_layout):
    """Test session selection with non-existent sessions."""
    session_label = ["01", "99"]  # 99 doesn't exist
    sessions = get_sessions_to_analyze(mock_bids_layout, session_label)
    assert sessions == ["01"]  # Only 01 should be included

def test_get_sessions_to_analyze_no_sessions(mock_bids_layout):
    """Test session selection with a dataset that has no sessions."""
    mock_bids_layout.get_sessions.return_value = []
    sessions = get_sessions_to_analyze(mock_bids_layout, [])
    assert sessions == [None]  # Should return [None] when no sessions exist

# Test copy_freesurfer_outputs function
def test_copy_freesurfer_outputs(test_directories):
    """Test the function that copies FreeSurfer outputs to BIDS structure."""
    # Setup
    freesurfer_dir = test_directories["freesurfer_dir"]
    output_subject_dir = test_directories["output_dir"] / "sub-001"
    output_subject_dir.mkdir(parents=True, exist_ok=True)
    
    # Call the function
    copy_freesurfer_outputs(str(freesurfer_dir), "001", None, str(output_subject_dir))
    
    # Verify file copying worked
    assert (output_subject_dir / "anat" / "T1.mgz").exists()
    assert (output_subject_dir / "anat" / "brain.mgz").exists()
    assert (output_subject_dir / "surf" / "lh.white").exists()
    assert (output_subject_dir / "surf" / "rh.white").exists()
    assert (output_subject_dir / "stats" / "aseg.stats").exists()
    assert (output_subject_dir / "scripts" / "recon-all.log").exists()

# Test the run_participant_level function
@patch("src.run.convert_subject")
@patch("src.run.copy_freesurfer_outputs")
def test_run_participant_level(
    mock_copy_outputs, 
    mock_convert_subject,
    mock_bids_layout, 
    mock_freesurfer_wrapper, 
    mock_bids_provenance, 
    test_directories
):
    """Test the participant level processing function."""
    # Setup
    subjects = ["sub-001"]
    sessions = ["01", "02"]
    freesurfer_dir = str(test_directories["freesurfer_dir"])
    nidm_dir = str(test_directories["nidm_dir"])
    output_dir = str(test_directories["output_dir"])
    skip_nidm = False
    
    # Call the function
    run_participant_level(
        mock_bids_layout,
        subjects,
        sessions,
        mock_freesurfer_wrapper,
        mock_bids_provenance,
        freesurfer_dir,
        nidm_dir,
        output_dir,
        skip_nidm
    )
    
    # Verify the FreeSurferWrapper was called for each subject/session combination
    assert mock_freesurfer_wrapper.process_subject.call_count == 2
    mock_freesurfer_wrapper.process_subject.assert_has_calls([
        call("001", "01", mock_bids_layout),
        call("001", "02", mock_bids_layout)
    ])
    
    # Verify provenance creation
    assert mock_bids_provenance.create_subject_provenance.call_count == 2
    
    # Verify output copying
    assert mock_copy_outputs.call_count == 2
    
    # Verify NIDM conversion
    assert mock_convert_subject.call_count == 2

# Test the run_group_level function
@patch("src.run.create_group_nidm")
def test_run_group_level(
    mock_create_group_nidm,
    mock_bids_provenance, 
    test_directories
):
    """Test the group level processing function."""
    # Setup
    subjects = ["sub-001", "sub-002"]
    freesurfer_dir = str(test_directories["freesurfer_dir"])
    nidm_dir = str(test_directories["nidm_dir"])
    skip_nidm = False
    
    # Call the function
    run_group_level(
        subjects,
        mock_bids_provenance,
        freesurfer_dir,
        nidm_dir,
        skip_nidm
    )
    
    # Verify group provenance creation
    mock_bids_provenance.create_group_provenance.assert_called_once_with(subjects)
    
    # Verify NIDM group creation
    mock_create_group_nidm.assert_called_once_with(subjects, nidm_dir)

# Test the main click command function
@patch("src.run.run_participant_level")
@patch("src.run.run_group_level")
@patch("src.run.BIDSLayout")
@patch("src.run.FreeSurferWrapper")
@patch("src.run.create_bids_provenance")
def test_main_participant_level(
    mock_create_provenance,
    mock_fs_wrapper,
    mock_layout_class,
    mock_run_group,
    mock_run_participant,
    test_directories
):
    """Test the main command-line function for participant level."""
    # Setup
    bids_dir = str(test_directories["bids_dir"])
    output_dir = str(test_directories["output_dir"])
    analysis_level = "participant"
    
    # Mock return values
    mock_layout = MagicMock()
    mock_layout.get_subjects.return_value = ["001", "002"]
    mock_layout.get_sessions.return_value = []
    mock_layout_class.return_value = mock_layout
    
    mock_wrapper = MagicMock()
    mock_fs_wrapper.return_value = mock_wrapper
    
    mock_provenance = MagicMock()
    mock_create_provenance.return_value = mock_provenance
    
    # Call the function through the click runner
    from click.testing import CliRunner
    runner = CliRunner()
    result = runner.invoke(
        main, 
        [
            bids_dir, 
            output_dir, 
            analysis_level, 
            "--participant_label", "001",
            "--skip_bids_validator"
        ]
    )
    
    # Verify success
    assert result.exit_code == 0
    
    # Verify the right analysis was triggered
    mock_run_participant.assert_called_once()
    mock_run_group.assert_not_called()
    
    # Verify directories were created
    assert (test_directories["output_dir"] / "freesurfer").exists()
    assert (test_directories["output_dir"] / "nidm").exists()

@patch("src.run.run_participant_level")
@patch("src.run.run_group_level")
@patch("src.run.BIDSLayout")
@patch("src.run.FreeSurferWrapper")
@patch("src.run.create_bids_provenance")
def test_main_group_level(
    mock_create_provenance,
    mock_fs_wrapper,
    mock_layout_class,
    mock_run_group,
    mock_run_participant,
    test_directories
):
    """Test the main command-line function for group level."""
    # Setup
    bids_dir = str(test_directories["bids_dir"])
    output_dir = str(test_directories["output_dir"])
    analysis_level = "group"
    
    # Mock return values
    mock_layout = MagicMock()
    mock_layout.get_subjects.return_value = ["001", "002"]
    mock_layout.get_sessions.return_value = []
    mock_layout_class.return_value = mock_layout
    
    mock_wrapper = MagicMock()
    mock_fs_wrapper.return_value = mock_wrapper
    
    mock_provenance = MagicMock()
    mock_create_provenance.return_value = mock_provenance
    
    # Call the function through the click runner
    from click.testing import CliRunner
    runner = CliRunner()
    result = runner.invoke(
        main, 
        [
            bids_dir, 
            output_dir, 
            analysis_level, 
            "--skip_bids_validator"
        ]
    )
    
    # Verify success
    assert result.exit_code == 0
    
    # Verify the right analysis was triggered
    mock_run_participant.assert_not_called()
    mock_run_group.assert_called_once()