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

from click.testing import CliRunner
from src.run import cli


@pytest.fixture
def bids_dataset(tmp_path):
    """Create a temporary BIDS dataset for testing."""
    # Create basic BIDS structure
    bids_dir = tmp_path / "bids_dataset"
    bids_dir.mkdir(parents=True)

    # Create dataset_description.json
    with open(bids_dir / "dataset_description.json", "w") as f:
        json.dump({
            "Name": "Test BIDS Dataset",
            "BIDSVersion": "1.4.0",
            "DatasetType": "raw"
        }, f)

    # Create subject directory
    subject_dir = bids_dir / "sub-001"
    subject_dir.mkdir()

    # Create anat directory
    anat_dir = subject_dir / "anat"
    anat_dir.mkdir()

    # Create dummy T1w image
    t1w_file = anat_dir / "sub-001_T1w.nii.gz"
    t1w_file.touch()

    # Create dummy T2w image
    t2w_file = anat_dir / "sub-001_T2w.nii.gz"
    t2w_file.touch()

    return bids_dir


@pytest.fixture
def freesurfer_license(tmp_path):
    """Create a dummy FreeSurfer license file."""
    license_file = tmp_path / "license.txt"
    license_file.write_text("dummy license")
    return license_file


@pytest.fixture
def output_dir(tmp_path):
    """Create a temporary output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def mock_layout():
    """Mock for BIDSLayout."""
    with patch("src.run.BIDSLayout") as mock:
        layout_instance = MagicMock()
        layout_instance.get_subjects.return_value = ["001"]
        layout_instance.get_sessions.return_value = ["01"]
        mock.return_value = layout_instance
        yield mock


@pytest.fixture
def mock_wrapper():
    """Mock for FreeSurferWrapper."""
    with patch("src.run.FreeSurferWrapper") as mock:
        wrapper_instance = MagicMock()
        wrapper_instance.process_subject.return_value = True
        wrapper_instance.get_processing_summary.return_value = {
            "total": 1,
            "success": 1,
            "failure": 0,
            "skipped": 0
        }
        mock.return_value = wrapper_instance
        yield mock


@pytest.fixture
def mock_nidm():
    """Mock for FreeSurferToNIDM."""
    with patch("src.run.FreeSurferToNIDM") as mock:
        nidm_instance = MagicMock()
        mock.return_value = nidm_instance
        yield mock


@pytest.fixture
def mock_fs_version():
    """Mock for FreeSurfer version."""
    with patch("src.run.get_freesurfer_version") as mock:
        mock.return_value = "7.3.2"
        yield mock


def test_basic_run(bids_dataset, output_dir, freesurfer_license, 
                  mock_layout, mock_wrapper, mock_nidm, mock_fs_version):
    """Test basic run with default options."""
    # Run the script
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "participant",
            str(bids_dataset),
            str(output_dir),
            "--participant_label", "001",
            "--freesurfer_license", str(freesurfer_license),
            "--skip_bids_validator"
        ],
        catch_exceptions=False
    )

    # Check results
    assert result.exit_code == 0

    # Verify BIDSLayout was initialized correctly with string path
    mock_layout.assert_called_once_with(str(bids_dataset), validate=False)
    
    # Verify FreeSurferWrapper was initialized correctly with Path objects
    mock_wrapper.assert_called_once_with(
        bids_dataset,  # Path object
        output_dir,    # Path object
        freesurfer_license,  # Path object
        None  # fs_options
    )
    
    # Verify subject processing was attempted
    wrapper_instance = mock_wrapper.return_value
    wrapper_instance.process_subject.assert_called_once_with(
        "001",  # subject
        "01",   # session
        mock_layout.return_value  # layout
    )
    
    # Verify NIDM conversion was attempted
    mock_nidm.assert_called_once()
    
    # Verify summary was generated
    wrapper_instance.get_processing_summary.assert_called_once()


def test_custom_freesurfer_dir(bids_dataset, output_dir, freesurfer_license):
    """Test run with custom FreeSurfer directory."""
    custom_fs_dir = output_dir / "custom_fs"
    custom_fs_dir.mkdir()

    with patch("src.run.BIDSLayout") as mock_layout, \
         patch("src.run.FreeSurferWrapper") as mock_wrapper, \
         patch("src.run.FreeSurferToNIDM") as mock_nidm, \
         patch("src.run.get_freesurfer_version") as mock_fs_version:

        # Set up mocks
        mock_layout_instance = MagicMock()
        mock_layout_instance.get_subjects.return_value = ["001"]
        mock_layout_instance.get_sessions.return_value = ["01"]
        mock_layout.return_value = mock_layout_instance

        # Mock FreeSurfer version
        mock_fs_version.return_value = "7.3.2"
        
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
                "participant",  # Command must come first
                str(bids_dataset),
                str(output_dir),
                "--participant_label", "001",
                "--freesurfer_license", str(freesurfer_license),
                "--skip_bids_validator",
                "--freesurfer_dir", str(custom_fs_dir)
            ],
            catch_exceptions=False
        )

        # Check results
        assert result.exit_code == 0

        # Verify BIDSLayout was initialized correctly
        mock_layout.assert_called_once_with(str(bids_dataset), validate=False)
        
        # Verify FreeSurferWrapper was initialized correctly
        mock_wrapper.assert_called_once_with(
            bids_dataset,
            output_dir,
            freesurfer_license,
            None  # fs_options
        )

        # Verify subject processing was attempted
        wrapper_instance = mock_wrapper.return_value
        wrapper_instance.process_subject.assert_called_once_with(
            "001",  # subject
            "01",   # session
            mock_layout.return_value  # layout
        )

        # Verify NIDM conversion was attempted with custom FreeSurfer directory
        mock_nidm.assert_called_once_with(
            custom_fs_dir,
            "001",
            "01",
            output_dir / "nidm"
        )


def test_skip_nidm(bids_dataset, output_dir, freesurfer_license,
                  mock_layout, mock_wrapper, mock_nidm, mock_fs_version):
    """Test that NIDM conversion is skipped when requested."""
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "participant",
            str(bids_dataset),
            str(output_dir),
            "--participant_label", "001",
            "--freesurfer_license", str(freesurfer_license),
            "--skip_bids_validator",
            "--skip_nidm"
        ],
        catch_exceptions=False
    )

    assert result.exit_code == 0
    mock_nidm.assert_not_called()


def test_session_less_mode(bids_dataset, output_dir, freesurfer_license):
    """Test run with session-less BIDS dataset."""
    with patch("src.run.BIDSLayout") as mock_layout, \
         patch("src.run.FreeSurferWrapper") as mock_wrapper, \
         patch("src.run.FreeSurferToNIDM") as mock_nidm, \
         patch("src.run.get_freesurfer_version") as mock_fs_version:

        # Set up mocks
        mock_layout_instance = MagicMock()
        mock_layout_instance.get_subjects.return_value = ["001"]
        mock_layout_instance.get_sessions.return_value = []  # Session-less mode
        mock_layout.return_value = mock_layout_instance

        # Mock FreeSurfer version
        mock_fs_version.return_value = "7.3.2"
        
        # Create a mock instance with proper attributes
        mock_wrapper_instance = MagicMock()
        mock_wrapper_instance.process_subject.return_value = True  # Processing succeeds
        mock_wrapper_instance.get_processing_summary.return_value = {
            "total": 1,
            "success": 1,
            "failure": 0,
            "skipped": 0
        }
        mock_wrapper.return_value = mock_wrapper_instance

        # Create a mock instance for NIDM
        mock_nidm_instance = MagicMock()
        mock_nidm.return_value = mock_nidm_instance

        # Run the script
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "participant",  # Command must come first
                str(bids_dataset),
                str(output_dir),
                "--participant_label", "001",
                "--freesurfer_license", str(freesurfer_license),
                "--skip_bids_validator"
            ],
            catch_exceptions=False
        )

        # Check results
        assert result.exit_code == 0

        # Verify BIDSLayout was initialized correctly
        mock_layout.assert_called_once_with(str(bids_dataset), validate=False)
        
        # Verify FreeSurferWrapper was initialized correctly
        mock_wrapper.assert_called_once_with(
            bids_dataset,
            output_dir,
            freesurfer_license,
            None  # fs_options
        )

        # Verify subject processing was attempted with None session
        wrapper_instance = mock_wrapper.return_value
        wrapper_instance.process_subject.assert_called_once_with(
            "001",  # subject
            None,   # session (None in session-less mode)
            mock_layout.return_value  # layout
        )

        # Verify NIDM conversion was attempted
        mock_nidm.assert_called_once_with(
            output_dir / "freesurfer",
            "001",
            None,  # session is None in session-less mode
            output_dir / "nidm"
        )

        # Verify NIDM conversion was completed
        mock_nidm_instance.convert.assert_called_once()


def test_specific_subjects_sessions(bids_dataset, output_dir, freesurfer_license):
    """Test run with specific subjects and sessions."""
    with patch("src.run.BIDSLayout") as mock_layout, \
         patch("src.run.FreeSurferWrapper") as mock_wrapper, \
         patch("src.run.FreeSurferToNIDM") as mock_nidm, \
         patch("src.run.get_freesurfer_version") as mock_fs_version:

        # Set up mocks
        mock_layout_instance = MagicMock()
        mock_layout_instance.get_subjects.return_value = ["001", "002"]
        mock_layout_instance.get_sessions.return_value = ["01", "02"]
        mock_layout.return_value = mock_layout_instance

        # Mock FreeSurfer version
        mock_fs_version.return_value = "7.3.2"
        
        # Create a mock instance with proper attributes
        mock_wrapper_instance = MagicMock()
        mock_wrapper_instance.process_subject.return_value = True  # Processing succeeds
        mock_wrapper_instance.get_processing_summary.return_value = {
            "total": 1,
            "success": 1,
            "failure": 0,
            "skipped": 0
        }
        mock_wrapper.return_value = mock_wrapper_instance

        # Create a mock instance for NIDM
        mock_nidm_instance = MagicMock()
        mock_nidm.return_value = mock_nidm_instance

        # Run the script
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "participant",  # Command must come first
                str(bids_dataset),
                str(output_dir),
                "--participant_label", "001",
                "--session_label", "01",
                "--freesurfer_license", str(freesurfer_license),
                "--skip_bids_validator"
            ],
            catch_exceptions=False
        )

        # Check results
        assert result.exit_code == 0

        # Verify BIDSLayout was initialized correctly
        mock_layout.assert_called_once_with(str(bids_dataset), validate=False)
        
        # Verify FreeSurferWrapper was initialized correctly
        mock_wrapper.assert_called_once_with(
            bids_dataset,
            output_dir,
            freesurfer_license,
            None  # fs_options
        )

        # Verify subject processing was attempted with specific subject and session
        wrapper_instance = mock_wrapper.return_value
        assert wrapper_instance.process_subject.call_count == 1
        wrapper_instance.process_subject.assert_called_once_with(
            "001",  # specific subject
            "01",   # specific session
            mock_layout.return_value  # layout
        )

        # Verify NIDM conversion was attempted
        mock_nidm.assert_called_once_with(
            output_dir / "freesurfer",
            "001",
            "01",  # specific session
            output_dir / "nidm"
        )

        # Verify NIDM conversion was completed
        mock_nidm_instance.convert.assert_called_once()


def test_error_handling(bids_dataset, output_dir, freesurfer_license,
                       mock_layout, mock_wrapper, mock_nidm, mock_fs_version):
    """Test error handling when processing fails."""
    # Make the process_subject call fail
    wrapper_instance = mock_wrapper.return_value
    wrapper_instance.process_subject.return_value = False
    
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "participant",
            str(bids_dataset),
            str(output_dir),
            "--participant_label", "001",
            "--freesurfer_license", str(freesurfer_license),
            "--skip_bids_validator"
        ],
        catch_exceptions=False
    )

    assert result.exit_code == 0  # Should still exit cleanly
    mock_nidm.assert_not_called()  # NIDM conversion should be skipped on failure


def test_verbose_output(bids_dataset, output_dir, freesurfer_license):
    """Test verbose output mode."""
    with patch("src.run.BIDSLayout") as mock_layout, \
         patch("src.run.FreeSurferWrapper") as mock_wrapper, \
         patch("src.run.FreeSurferToNIDM") as mock_nidm, \
         patch("src.run.get_freesurfer_version") as mock_fs_version, \
         patch("src.run.setup_logging") as mock_setup_logging:

        # Set up mocks
        mock_layout_instance = MagicMock()
        mock_layout_instance.get_subjects.return_value = ["001"]
        mock_layout_instance.get_sessions.return_value = ["01"]
        mock_layout.return_value = mock_layout_instance

        # Mock FreeSurfer version
        mock_fs_version.return_value = "7.3.2"
        
        # Create a mock instance with proper attributes
        mock_wrapper_instance = MagicMock()
        mock_wrapper_instance.process_subject.return_value = True  # Processing succeeds
        mock_wrapper_instance.get_processing_summary.return_value = {
            "total": 1,
            "success": 1,
            "failure": 0,
            "skipped": 0
        }
        mock_wrapper.return_value = mock_wrapper_instance

        # Create a mock instance for NIDM
        mock_nidm_instance = MagicMock()
        mock_nidm.return_value = mock_nidm_instance

        # Run the script with output capture
        runner = CliRunner()
        with runner.isolation():
            result = runner.invoke(
                cli,
                [
                    "participant",  # Command must come first
                    str(bids_dataset),
                    str(output_dir),
                    "--participant_label", "001",
                    "--freesurfer_license", str(freesurfer_license),
                    "--skip_bids_validator",
                    "--verbose"
                ],
                catch_exceptions=False
            )

        # Check results
        assert result.exit_code == 0
        
        # Verify logging was set up with debug level
        mock_setup_logging.assert_called_once_with(logging.DEBUG)

        # Verify BIDSLayout was initialized correctly
        mock_layout.assert_called_once_with(str(bids_dataset), validate=False)
        
        # Verify FreeSurferWrapper was initialized correctly
        mock_wrapper.assert_called_once_with(
            bids_dataset,
            output_dir,
            freesurfer_license,
            None  # fs_options
        )

        # Verify subject processing was attempted
        wrapper_instance = mock_wrapper.return_value
        wrapper_instance.process_subject.assert_called_once_with(
            "001",  # subject
            "01",   # session
            mock_layout.return_value  # layout
        )

        # Verify NIDM conversion was attempted
        mock_nidm.assert_called_once_with(
            output_dir / "freesurfer",
            "001",
            "01",
            output_dir / "nidm"
        )

        # Verify NIDM conversion was completed
        mock_nidm_instance.convert.assert_called_once()


def test_processing_summary(bids_dataset, output_dir, freesurfer_license):
    """Test generation of processing summary."""
    with patch("src.run.BIDSLayout") as mock_layout, \
         patch("src.run.FreeSurferWrapper") as mock_wrapper, \
         patch("src.run.FreeSurferToNIDM") as mock_nidm, \
         patch("src.run.get_freesurfer_version") as mock_fs_version, \
         patch("src.freesurfer.wrapper.datetime") as mock_datetime:

        # Set up mocks
        mock_layout_instance = MagicMock()
        mock_layout_instance.get_subjects.return_value = ["001"]
        mock_layout_instance.get_sessions.return_value = ["01"]
        mock_layout.return_value = mock_layout_instance

        # Mock FreeSurfer version
        mock_fs_version.return_value = "7.3.2"

        # Mock datetime
        mock_now = MagicMock()
        mock_now.isoformat.return_value = "2024-03-14T12:00:00"
        mock_datetime.now.return_value = mock_now
        
        # Create a mock instance with proper attributes
        mock_wrapper_instance = MagicMock()
        mock_wrapper_instance.process_subject.return_value = True  # Processing succeeds
        summary_data = {
            "total": 1,
            "success": 1,
            "failure": 0,
            "skipped": 0,
            "timestamp": "2024-03-14T12:00:00",
            "freesurfer_version": "7.3.2"
        }
        mock_wrapper_instance.get_processing_summary.return_value = summary_data

        # Mock the save_processing_summary method to actually create the file
        def mock_save_summary(output_path=None):
            if output_path is None:
                output_path = output_dir / "processing_summary.json"
            with open(output_path, "w") as f:
                json.dump(summary_data, f, indent=2)
            return output_path

        mock_wrapper_instance.save_processing_summary = mock_save_summary
        mock_wrapper.return_value = mock_wrapper_instance

        # Create a mock instance for NIDM
        mock_nidm_instance = MagicMock()
        mock_nidm.return_value = mock_nidm_instance

        # Run the script
        runner = CliRunner()
        with runner.isolation():
            result = runner.invoke(
                cli,
                [
                    "participant",  # Command must come first
                    str(bids_dataset),
                    str(output_dir),
                    "--participant_label", "001",
                    "--freesurfer_license", str(freesurfer_license),
                    "--skip_bids_validator"
                ],
                catch_exceptions=False
            )

        # Check results
        assert result.exit_code == 0

        # Verify BIDSLayout was initialized correctly
        mock_layout.assert_called_once_with(str(bids_dataset), validate=False)
        
        # Verify FreeSurferWrapper was initialized correctly
        mock_wrapper.assert_called_once_with(
            bids_dataset,
            output_dir,
            freesurfer_license,
            None  # fs_options
        )

        # Verify subject processing was attempted
        wrapper_instance = mock_wrapper.return_value
        wrapper_instance.process_subject.assert_called_once_with(
            "001",  # subject
            "01",   # session
            mock_layout.return_value  # layout
        )

        # Verify NIDM conversion was attempted
        mock_nidm.assert_called_once_with(
            output_dir / "freesurfer",
            "001",
            "01",
            output_dir / "nidm"
        )

        # Verify NIDM conversion was completed
        mock_nidm_instance.convert.assert_called_once()

        # Verify summary file exists and contains correct data
        summary_file = output_dir / "processing_summary.json"
        assert summary_file.exists()
        with open(summary_file, "r") as f:
            saved_summary = json.load(f)
        
        # Verify all fields in the summary
        assert saved_summary == summary_data


def test_invalid_subject(bids_dataset, output_dir, freesurfer_license):
    """Test handling of invalid subject label."""
    with patch("src.run.BIDSLayout") as mock_layout, \
         patch("src.run.FreeSurferWrapper") as mock_wrapper, \
         patch("src.run.get_freesurfer_version") as mock_fs_version:

        # Set up mocks
        mock_layout_instance = MagicMock()
        mock_layout_instance.get_subjects.return_value = ["001"]  # Only subject 001 exists
        mock_layout_instance.get_sessions.return_value = ["01"]
        mock_layout.return_value = mock_layout_instance

        # Mock FreeSurfer version
        mock_fs_version.return_value = "7.3.2"
        
        # Create a mock instance with proper attributes
        mock_wrapper_instance = MagicMock()
        # We don't need to set process_subject.return_value because it should never be called
        mock_wrapper.return_value = mock_wrapper_instance

        # Run the script
        runner = CliRunner()
        with runner.isolation():
            result = runner.invoke(
                cli,
                [
                    "participant",  # Command must come first
                    str(bids_dataset),
                    str(output_dir),
                    "--participant_label", "999",  # Non-existent subject
                    "--freesurfer_license", str(freesurfer_license),
                    "--skip_bids_validator"
                ],
                catch_exceptions=False
            )

        # Check results
        assert result.exit_code == 1  # Should fail because subject doesn't exist
        assert "Subject(s) 999 not found in dataset" in result.output

        # Verify BIDSLayout was initialized correctly
        mock_layout.assert_called_once_with(str(bids_dataset), validate=False)
        
        # Verify FreeSurferWrapper was initialized correctly
        mock_wrapper.assert_called_once_with(
            bids_dataset,
            output_dir,
            freesurfer_license,
            None  # fs_options
        )

        # Verify process_subject was never called because subject doesn't exist
        mock_wrapper_instance.process_subject.assert_not_called()


def test_group_level_analysis(bids_dataset, output_dir, freesurfer_license):
    """Test group level analysis (should be not implemented)."""
    with patch("src.run.BIDSLayout") as mock_layout, \
         patch("src.run.get_freesurfer_version") as mock_fs_version:

        # Set up mocks
        mock_layout_instance = MagicMock()
        mock_layout_instance.get_subjects.return_value = ["001", "002"]
        mock_layout.return_value = mock_layout_instance

        # Mock FreeSurfer version
        mock_fs_version.return_value = "7.3.2"

        # Create FreeSurfer output directory with some subjects
        freesurfer_dir = output_dir / "freesurfer"
        freesurfer_dir.mkdir(parents=True)
        (freesurfer_dir / "001").mkdir()
        (freesurfer_dir / "002").mkdir()

        # Run the script
        runner = CliRunner()
        with runner.isolation():
            result = runner.invoke(
                cli,
                [
                    "group",  # Command must come first
                    str(bids_dataset),
                    str(output_dir),
                    "--freesurfer_license", str(freesurfer_license),
                    "--skip_bids_validator"
                ],
                catch_exceptions=False
            )

        # Check results
        assert result.exit_code == 0
        assert "Found 2 processed subjects" in result.output
        assert "Group level analysis not implemented yet" in result.output
        assert "In the future, this will include:" in result.output