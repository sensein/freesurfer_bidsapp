#!/usr/bin/env python3
"""
Tests for the FreeSurfer BIDS App run script.
"""

import json
import os
import pytest
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock, call
import logging
import subprocess

from click.testing import CliRunner
from src.run import cli


@pytest.fixture
def bids_dataset(tmp_path):
    """Create a temporary BIDS dataset for testing."""
    # Create basic BIDS structure
    bids_dir = Path(tmp_path) / "bids_dataset"
    bids_dir.mkdir(parents=True)

    # Create dataset_description.json
    with open(bids_dir / "dataset_description.json", "w") as f:
        json.dump({
            "Name": "Test BIDS Dataset",
            "BIDSVersion": "1.4.0",
            "DatasetType": "raw"
        }, f)

    # Create subject directory
    subject_dir = Path(bids_dir) / "sub-001"
    subject_dir.mkdir()

    # Create anat directory
    anat_dir = Path(bids_dir) / "sub-001" / "anat"
    anat_dir.mkdir()

    # Create dummy T1w image
    t1w_file = Path(anat_dir) / "sub-001_T1w.nii.gz"
    t1w_file.touch()

    # Create dummy T2w image
    t2w_file = Path(anat_dir) / "sub-001_T2w.nii.gz"
    t2w_file.touch()

    return bids_dir


@pytest.fixture
def freesurfer_license(tmp_path):
    """Create a dummy FreeSurfer license file."""
    license_file = Path(tmp_path) / "license.txt"
    license_file.write_text("dummy license")
    return license_file


@pytest.fixture
def output_dir(tmp_path):
    """Create a temporary output directory."""
    output_dir = Path(tmp_path) / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def mock_layout():
    """Mock for BIDSLayout."""
    with patch("bids.layout.BIDSLayout") as mock:
        layout_instance = MagicMock()
        layout_instance.get_subjects.return_value = ["sub-001"]  # Only keep subjects
        mock.return_value = layout_instance
        yield mock


@pytest.fixture
def mock_wrapper():
    """Mock for FreeSurferWrapper."""
    with patch("src.freesurfer.wrapper.FreeSurferWrapper") as mock:
        wrapper_instance = MagicMock()
        # Mock successful subject processing
        wrapper_instance.process_subject.return_value = True
        # Mock the command execution to return success
        wrapper_instance.run_recon_all.return_value = True
        # Mock the processing summary
        wrapper_instance.get_processing_summary.return_value = {
            "total": 1,
            "success": 1,
            "failure": 0,
            "skipped": 0
        }
        mock.return_value = wrapper_instance
        
        # Mock environment variables
        with patch.dict('os.environ', {
            'FREESURFER_HOME': '/dummy/path',
            'SUBJECTS_DIR': '/dummy/subjects'
        }):
            yield mock


@pytest.fixture
def mock_nidm():
    """Mock for fs_to_nidm.py subprocess call."""
    with patch("subprocess.run") as mock:
        # Mock successful subprocess run
        mock.return_value = MagicMock(
            returncode=0,
            stdout="NIDM conversion successful",
            stderr=""
        )
        yield mock


@pytest.fixture
def mock_fs_version():
    """Mock for FreeSurfer version."""
    with patch("src.utils.get_freesurfer_version") as mock:
        mock.return_value = "7.3.2"
        yield mock


@patch('src.run.BIDSLayout')
@patch('src.run.FreeSurferWrapper')
@patch('subprocess.run')
@patch.dict('os.environ', {
    'FREESURFER_HOME': '/dummy/path',
    'SUBJECTS_DIR': '/dummy/subjects'
})
def test_basic_run(mock_subprocess_run, mock_wrapper_class, mock_layout, bids_dataset, output_dir, freesurfer_license,
                  mock_fs_version):
    """Test basic run with default options."""
    # Set up mock returns
    mock_layout_instance = MagicMock()
    # Mock both methods for subject handling
    mock_layout_instance.get_subjects.return_value = ["001"]  # Without sub- prefix
    mock_layout_instance.get.return_value = [MagicMock(subject="001")]  # Mock BIDS entities
    mock_layout_instance.get_sessions.return_value = []
    mock_layout.return_value = mock_layout_instance

    # Set up FreeSurferWrapper mock
    mock_wrapper_instance = MagicMock()
    mock_wrapper_instance.process_subject.return_value = True
    mock_wrapper_class.return_value = mock_wrapper_instance

    # Set up subprocess.run mock
    mock_subprocess_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

    # Create proper BIDS dataset structure
    subject_dir = Path(bids_dataset) / "sub-001"
    anat_dir = Path(subject_dir) / "anat"
    anat_dir.mkdir(parents=True, exist_ok=True)
    (anat_dir / "sub-001_T1w.nii.gz").touch()

    # Create dataset_description.json
    dataset_description = {
        "Name": "Test dataset",
        "BIDSVersion": "1.8.0",
        "DatasetType": "raw"
    }
    with open(Path(bids_dataset) / "dataset_description.json", "w") as f:
        json.dump(dataset_description, f)

    # Run the test
    runner = CliRunner()
    result = runner.invoke(cli, [
        str(bids_dataset),
        str(output_dir),
        'participant',
        '--participant_label', 'sub-001',
        '--freesurfer_license', str(freesurfer_license),
        '--skip-bids-validation'
    ], catch_exceptions=False)

    # Debug output
    print(f"\nExit code: {result.exit_code}")
    print(f"Output: {result.output}")
    if result.exception:
        print(f"Exception: {result.exception}")

    # Verify the result
    assert result.exit_code == 0

    # Verify FreeSurfer processing was called
    mock_wrapper_instance.process_subject.assert_called_once()

    # Verify NIDM conversion was called with correct arguments
    mock_subprocess_run.assert_called_once()
    call_args = mock_subprocess_run.call_args[0][0]
    assert call_args[0] == 'python3'
    assert call_args[1].endswith('fs_to_nidm.py')
    assert call_args[2] == '-s'
    assert call_args[4] == '-o'
    assert call_args[6] == '-j'


def test_custom_freesurfer_dir(bids_dataset, output_dir, freesurfer_license):
    """Test run with custom FreeSurfer directory."""
    custom_fs_dir = Path(output_dir) / "custom_fs"
    custom_fs_dir.mkdir()

    # Create proper BIDS dataset structure first
    subject_dir = Path(bids_dataset) / "sub-001"
    anat_dir = subject_dir / "anat"
    anat_dir.mkdir(parents=True, exist_ok=True)
    
    # Create required BIDS files
    (anat_dir / "sub-001_T1w.nii.gz").touch()
    
    # Create dataset_description.json
    dataset_description = {
        "Name": "Test dataset",
        "BIDSVersion": "1.8.0",
        "DatasetType": "raw"
    }
    with open(Path(bids_dataset) / "dataset_description.json", "w") as f:
        json.dump(dataset_description, f)

    with patch("src.run.BIDSLayout") as mock_layout, \
         patch("src.freesurfer.wrapper.FreeSurferWrapper") as mock_wrapper, \
         patch("subprocess.run") as mock_nidm, \
         patch("src.utils.get_freesurfer_version") as mock_fs_version, \
         patch("os.environ", {"FREESURFER_HOME": "/dummy/path"}):

        # Set up mock layout
        mock_layout_instance = MagicMock()
        mock_layout_instance.get_subjects.return_value = ["001"]
        mock_entity = MagicMock()
        mock_entity.subject = "001"
        mock_layout_instance.get.return_value = [mock_entity]
        mock_layout_instance.get_sessions.return_value = []
        mock_layout.return_value = mock_layout_instance

        # Run the script
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                str(bids_dataset),
                str(output_dir),
                'participant',
                "--participant_label", "sub-001",
                "--freesurfer_license", str(freesurfer_license),
                "--skip-bids-validation"
            ],
            catch_exceptions=False
        )

        # Debug output
        if result.exit_code != 0:
            print(f"\nExit code: {result.exit_code}")
            print(f"Output: {result.output}")
            if result.exception:
                print(f"Exception: {result.exception}")

        assert result.exit_code == 0


def test_skip_nidm(bids_dataset, output_dir, freesurfer_license):
    """Test that NIDM conversion is skipped when requested."""
    # Create proper BIDS dataset structure
    subject_dir = Path(bids_dataset) / "sub-001"
    anat_dir = subject_dir / "anat"
    anat_dir.mkdir(parents=True, exist_ok=True)
    (anat_dir / "sub-001_T1w.nii.gz").touch()

    # Create dataset_description.json
    dataset_description = {
        "Name": "Test dataset",
        "BIDSVersion": "1.8.0",
        "DatasetType": "raw"
    }
    with open(Path(bids_dataset) / "dataset_description.json", "w") as f:
        json.dump(dataset_description, f)

    # Change the patch location to where it's imported in run.py
    with patch('src.run.FreeSurferWrapper') as mock_wrapper, \
         patch('src.run.BIDSLayout') as mock_layout, \
         patch('subprocess.run') as mock_nidm, \
         patch('src.utils.get_freesurfer_version') as mock_fs_version, \
         patch("os.environ", {"FREESURFER_HOME": "/dummy/path"}):

        # Set up mock returns
        mock_layout_instance = MagicMock()
        mock_layout_instance.get_subjects.return_value = ["001"]
        mock_entity = MagicMock()
        mock_entity.subject = "001"
        mock_layout_instance.get.return_value = [mock_entity]
        mock_layout_instance.get_sessions.return_value = []
        mock_layout.return_value = mock_layout_instance

        # Mock FreeSurfer version
        mock_fs_version.return_value = "8.0.0"

        # Create wrapper mock instance
        mock_wrapper_instance = MagicMock()
        mock_wrapper_instance.process_subject.return_value = True
        mock_wrapper.return_value = mock_wrapper_instance

        # Run the test
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                str(bids_dataset),
                str(output_dir),
                'participant',
                "--participant_label", "sub-001",
                "--freesurfer_license", str(freesurfer_license),
                "--skip-bids-validation",
                "--skip_nidm"
            ],
            catch_exceptions=False
        )

        assert result.exit_code == 0
        mock_wrapper.assert_called_once()
        mock_nidm.assert_not_called()


def test_error_handling(bids_dataset, output_dir, freesurfer_license):
    """Test error handling when processing fails."""
    # Create proper BIDS dataset structure
    subject_dir = Path(bids_dataset) / "sub-001"
    anat_dir = subject_dir / "anat"
    anat_dir.mkdir(parents=True, exist_ok=True)
    (anat_dir / "sub-001_T1w.nii.gz").touch()

    # Create dataset_description.json
    dataset_description = {
        "Name": "Test dataset",
        "BIDSVersion": "1.8.0",
        "DatasetType": "raw"
    }
    with open(Path(bids_dataset) / "dataset_description.json", "w") as f:
        json.dump(dataset_description, f)

    with patch('src.run.BIDSLayout') as mock_layout, \
         patch('src.run.FreeSurferWrapper') as mock_wrapper, \
         patch('subprocess.run') as mock_nidm, \
         patch('src.run.get_version_info') as mock_version_info, \
         patch("os.environ", {"FREESURFER_HOME": "/dummy/path"}):

        # Set up mock returns
        mock_layout_instance = MagicMock()
        mock_layout_instance.get_subjects.return_value = ["001"]  # Without sub- prefix
        mock_layout.return_value = mock_layout_instance

        # Set up wrapper to raise an exception
        mock_wrapper_instance = MagicMock()
        mock_wrapper_instance.process_subject.side_effect = Exception("FreeSurfer processing failed")
        mock_wrapper.return_value = mock_wrapper_instance

        # Mock version info
        mock_version_info.return_value = {
            "freesurfer": {"version": "8.0.0", "build_stamp": None},
            "bids_freesurfer": {"version": "0.1.0"},
            "python": {"version": "3.9.0", "packages": {}}
        }

        # Run the test
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                str(bids_dataset),
                str(output_dir),
                'participant',
                "--participant_label", "sub-001",
                "--freesurfer_license", str(freesurfer_license),
                "--skip-bids-validation"
            ],
            catch_exceptions=False
        )

        # Debug output
        print(f"\nExit code: {result.exit_code}")
        print(f"Output: {result.output}")
        if result.exception:
            print(f"Exception: {result.exception}")

        assert result.exit_code == 1
        assert "Error during processing" in result.output


def test_verbose_output(bids_dataset, output_dir, freesurfer_license):
    """Test verbose output mode."""
    # Create proper BIDS dataset structure
    subject_dir = Path(bids_dataset) / "sub-001"
    anat_dir = subject_dir / "anat"
    anat_dir.mkdir(parents=True, exist_ok=True)
    (anat_dir / "sub-001_T1w.nii.gz").touch()

    # Create dataset_description.json
    dataset_description = {
        "Name": "Test dataset",
        "BIDSVersion": "1.8.0",
        "DatasetType": "raw"
    }
    with open(Path(bids_dataset) / "dataset_description.json", "w") as f:
        json.dump(dataset_description, f)

    with patch("src.run.BIDSLayout") as mock_layout, \
         patch("src.run.FreeSurferWrapper") as mock_wrapper, \
         patch("subprocess.run") as mock_nidm, \
         patch("src.run.get_version_info") as mock_version_info, \
         patch("os.environ", {"FREESURFER_HOME": "/dummy/path"}), \
         patch("src.run.setup_logging") as mock_setup_logging:

        # Set up mock returns
        mock_layout_instance = MagicMock()
        mock_layout_instance.get_subjects.return_value = ["001"]  # Without sub- prefix
        mock_layout.return_value = mock_layout_instance

        # Mock version info
        mock_version_info.return_value = {
            "freesurfer": {"version": "8.0.0", "build_stamp": None},
            "bids_freesurfer": {"version": "0.1.0"},
            "python": {"version": "3.9.0", "packages": {}}
        }

        # Create a mock instance with proper attributes
        mock_wrapper_instance = MagicMock()
        mock_wrapper_instance.process_subject.return_value = True
        mock_wrapper_instance.get_processing_summary.return_value = {
            "total": 1,
            "success": 1,
            "failure": 0,
            "skipped": 0
        }
        mock_wrapper.return_value = mock_wrapper_instance

        # Run the script
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                str(bids_dataset),
                str(output_dir),
                'participant',
                "--participant_label", "sub-001",
                "--freesurfer_license", str(freesurfer_license),
                "--skip-bids-validation",
                "--verbose"
            ],
            catch_exceptions=False
        )

        # Debug output
        print(f"\nExit code: {result.exit_code}")
        print(f"Output: {result.output}")
        if result.exception:
            print(f"Exception: {result.exception}")

        assert result.exit_code == 0
        mock_setup_logging.assert_called_once_with(logging.DEBUG)  # Verify verbose logging was set
        assert mock_wrapper_instance.process_subject.called


def test_processing_summary(bids_dataset, output_dir, freesurfer_license):
    """Test generation of processing summary."""
    # Create proper BIDS dataset structure
    subject_dir = Path(bids_dataset) / "sub-001"
    anat_dir = subject_dir / "anat"
    anat_dir.mkdir(parents=True, exist_ok=True)
    (anat_dir / "sub-001_T1w.nii.gz").touch()

    # Create dataset_description.json
    dataset_description = {
        "Name": "Test dataset",
        "BIDSVersion": "1.8.0",
        "DatasetType": "raw"
    }
    with open(Path(bids_dataset) / "dataset_description.json", "w") as f:
        json.dump(dataset_description, f)

    with patch("src.run.BIDSLayout") as mock_layout, \
         patch("src.run.FreeSurferWrapper") as mock_wrapper, \
         patch("subprocess.run") as mock_nidm, \
         patch("src.run.get_version_info") as mock_version_info, \
         patch("os.environ", {"FREESURFER_HOME": "/dummy/path"}):

        # Set up mock returns
        mock_layout_instance = MagicMock()
        mock_layout_instance.get_subjects.return_value = ["001"]  # Without sub- prefix
        mock_layout.return_value = mock_layout_instance

        # Mock version info
        mock_version_info.return_value = {
            "freesurfer": {"version": "8.0.0", "build_stamp": None},
            "bids_freesurfer": {"version": "0.1.0"},
            "python": {"version": "3.9.0", "packages": {}}
        }

        # Create wrapper mock instance
        mock_wrapper_instance = MagicMock()
        mock_wrapper_instance.process_subject.return_value = True
        summary_data = {
            "total": 1,
            "success": 1,
            "failure": 0,
            "skipped": 0
        }
        mock_wrapper_instance.get_processing_summary.return_value = summary_data
        mock_wrapper.return_value = mock_wrapper_instance

        # Run the script
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                str(bids_dataset),
                str(output_dir),
                'participant',
                "--participant_label", "sub-001",
                "--freesurfer_license", str(freesurfer_license),
                "--skip-bids-validation"
            ],
            catch_exceptions=False
        )

        # Debug output
        print(f"\nExit code: {result.exit_code}")
        print(f"Output: {result.output}")
        if result.exception:
            print(f"Exception: {result.exception}")

        assert result.exit_code == 0
        assert mock_wrapper_instance.process_subject.called
        assert mock_wrapper_instance.get_processing_summary.called


def test_invalid_subject(bids_dataset, output_dir, freesurfer_license):
    """Test handling of invalid subject label."""
    with patch("src.run.BIDSLayout") as mock_layout, \
         patch("src.freesurfer.wrapper.FreeSurferWrapper") as mock_wrapper, \
         patch("os.environ", {"FREESURFER_HOME": "/dummy/path"}):

        # Set up mocks
        mock_layout_instance = MagicMock()
        mock_layout_instance.get_subjects.return_value = ["001"]
        mock_layout_instance.get.return_value = []  # Return empty list for invalid subject
        mock_layout.return_value = mock_layout_instance

        # Run the script
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                str(bids_dataset),
                str(output_dir),
                'participant',
                "--participant_label", "sub-999",  # Add sub- prefix
                "--freesurfer_license", str(freesurfer_license),
                "--skip-bids-validation"
            ],
            catch_exceptions=False
        )

        assert result.exit_code == 1
        assert "Subject sub-999 not found in dataset" in result.output  # Updated error message