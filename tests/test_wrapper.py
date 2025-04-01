#!/usr/bin/env python3
"""
Tests for the FreeSurfer wrapper class.
"""

import json
import os
import pytest
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.freesurfer.wrapper import FreeSurferWrapper


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
def mock_bids_layout():
    """Create a mock BIDS layout."""
    layout = MagicMock()
    
    def get_mock(**kwargs):
        if kwargs.get('suffix') == 'T1w':
            return ["sub-001/anat/sub-001_T1w.nii.gz"]
        elif kwargs.get('suffix') == 'T2w':
            return ["sub-001/anat/sub-001_T2w.nii.gz"]
        return []
    
    layout.get.side_effect = get_mock
    return layout


def test_wrapper_initialization(bids_dataset, output_dir, freesurfer_license):
    """Test wrapper initialization."""
    with patch.dict(os.environ, {"FREESURFER_HOME": "/opt/freesurfer"}):
        wrapper = FreeSurferWrapper(
            str(bids_dataset),
            str(output_dir),
            freesurfer_license=str(freesurfer_license)
        )

        assert wrapper.bids_dir == Path(bids_dataset)
        assert wrapper.output_dir == Path(output_dir)
        assert wrapper.freesurfer_dir == Path(output_dir) / "freesurfer"
        assert wrapper.freesurfer_license == str(freesurfer_license)
        assert wrapper.fs_options == ""
        assert wrapper.parallel is False
        assert wrapper.num_threads is None
        assert wrapper.memory_limit is None


def test_wrapper_with_options(bids_dataset, output_dir, freesurfer_license):
    """Test wrapper initialization with additional options."""
    with patch.dict(os.environ, {"FREESURFER_HOME": "/opt/freesurfer"}):
        wrapper = FreeSurferWrapper(
            str(bids_dataset),
            str(output_dir),
            freesurfer_license=str(freesurfer_license),
            fs_options="-parallel",
            parallel=True,
            num_threads=4,
            memory_limit=8
        )

        assert wrapper.fs_options == "-parallel"
        assert wrapper.parallel is True
        assert wrapper.num_threads == 4
        assert wrapper.memory_limit == 8


def test_create_recon_all_command(bids_dataset, output_dir, freesurfer_license):
    """Test creation of recon-all command."""
    with patch.dict(os.environ, {"FREESURFER_HOME": "/opt/freesurfer"}):
        wrapper = FreeSurferWrapper(
            str(bids_dataset),
            str(output_dir),
            freesurfer_license=str(freesurfer_license)
        )

        t1w_images = ["sub-001/anat/sub-001_T1w.nii.gz"]
        t2w_images = ["sub-001/anat/sub-001_T2w.nii.gz"]
        cmd = wrapper._create_recon_all_command("001", t1w_images, t2w_images)

        assert "recon-all" in cmd
        assert "-subjid" in cmd
        assert "001" in cmd[cmd.index("-subjid") + 1]
        assert "-all" in cmd
        assert "-i" in cmd
        assert "-T2" in cmd
        assert "-T2pial" in cmd


def test_create_recon_all_command_with_options(bids_dataset, output_dir, freesurfer_license):
    """Test creation of recon-all command with additional options."""
    with patch.dict(os.environ, {"FREESURFER_HOME": "/opt/freesurfer"}):
        wrapper = FreeSurferWrapper(
            str(bids_dataset),
            str(output_dir),
            freesurfer_license=str(freesurfer_license),
            fs_options="-parallel",
            parallel=True,
            num_threads=4,
            memory_limit=8
        )

        t1w_images = ["sub-001/anat/sub-001_T1w.nii.gz"]
        cmd = wrapper._create_recon_all_command("001", t1w_images)

        assert "-parallel" in cmd
        assert "-openmp" in cmd
        assert "4" in cmd[cmd.index("-openmp") + 1]
        assert "-max-threads" in cmd
        assert "8" in cmd[cmd.index("-max-threads") + 1]


def test_find_t1w_images(bids_dataset, output_dir, freesurfer_license, mock_bids_layout):
    """Test finding T1w images."""
    with patch.dict(os.environ, {"FREESURFER_HOME": "/opt/freesurfer"}):
        wrapper = FreeSurferWrapper(
            str(bids_dataset),
            str(output_dir),
            freesurfer_license=str(freesurfer_license)
        )

        t1w_images = wrapper._find_t1w_images(mock_bids_layout, "001")
        assert len(t1w_images) == 1
        assert "T1w.nii.gz" in t1w_images[0]


def test_find_t2w_images(bids_dataset, output_dir, freesurfer_license, mock_bids_layout):
    """Test finding T2w images."""
    with patch.dict(os.environ, {"FREESURFER_HOME": "/opt/freesurfer"}):
        wrapper = FreeSurferWrapper(
            str(bids_dataset),
            str(output_dir),
            freesurfer_license=str(freesurfer_license)
        )

        t2w_images = wrapper._find_t2w_images(mock_bids_layout, "001")
        assert len(t2w_images) == 1
        assert "T2w.nii.gz" in t2w_images[0]


def test_organize_bids_output(bids_dataset, output_dir, freesurfer_license):
    """Test organizing outputs in BIDS format."""
    with patch.dict(os.environ, {"FREESURFER_HOME": "/opt/freesurfer"}):
        wrapper = FreeSurferWrapper(
            str(bids_dataset),
            str(output_dir),
            freesurfer_license=str(freesurfer_license)
        )

        # Create dummy FreeSurfer output structure
        fs_subject_dir = wrapper.freesurfer_dir / "001"
        fs_subject_dir.mkdir(parents=True)
        (fs_subject_dir / "mri").mkdir()
        (fs_subject_dir / "stats").mkdir()
        (fs_subject_dir / "mri" / "brain.mgz").touch()
        (fs_subject_dir / "mri" / "aparc.DKTatlas+aseg.mgz").touch()
        (fs_subject_dir / "mri" / "wmparc.mgz").touch()
        (fs_subject_dir / "stats" / "aseg.stats").touch()

        # Organize outputs
        wrapper._organize_bids_output("001", "001")

        # Check BIDS structure
        assert (output_dir / "sub-001" / "anat").exists()
        assert (output_dir / "sub-001" / "stats").exists()
        assert (output_dir / "dataset_description.json").exists()
        assert (output_dir / "README").exists()


def test_organize_bids_output_with_session(bids_dataset, output_dir, freesurfer_license):
    """Test organizing outputs in BIDS format with session."""
    with patch.dict(os.environ, {"FREESURFER_HOME": "/opt/freesurfer"}):
        wrapper = FreeSurferWrapper(
            str(bids_dataset),
            str(output_dir),
            freesurfer_license=str(freesurfer_license)
        )

        # Create dummy FreeSurfer output structure
        fs_subject_dir = wrapper.freesurfer_dir / "001_01"
        fs_subject_dir.mkdir(parents=True)
        (fs_subject_dir / "mri").mkdir()
        (fs_subject_dir / "stats").mkdir()
        (fs_subject_dir / "mri" / "brain.mgz").touch()
        (fs_subject_dir / "mri" / "aparc.DKTatlas+aseg.mgz").touch()
        (fs_subject_dir / "mri" / "wmparc.mgz").touch()
        (fs_subject_dir / "stats" / "aseg.stats").touch()

        # Organize outputs
        wrapper._organize_bids_output("001_01", "001", "01")

        # Check BIDS structure
        assert (output_dir / "sub-001" / "ses-01" / "anat").exists()
        assert (output_dir / "sub-001" / "ses-01" / "stats").exists()


def test_process_subject(bids_dataset, output_dir, freesurfer_license, mock_bids_layout):
    """Test processing a subject."""
    with patch.dict(os.environ, {"FREESURFER_HOME": "/opt/freesurfer"}):
        wrapper = FreeSurferWrapper(
            str(bids_dataset),
            str(output_dir),
            freesurfer_license=str(freesurfer_license)
        )

        # Mock subprocess.run
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="",
                stderr=""
            )

            # Process subject
            success = wrapper.process_subject("001", layout=mock_bids_layout)

            assert success is True
            assert "001" in wrapper.results["success"]
            mock_run.assert_called_once()


def test_process_subject_failure(bids_dataset, output_dir, freesurfer_license, mock_bids_layout):
    """Test processing a subject with failure."""
    with patch.dict(os.environ, {"FREESURFER_HOME": "/opt/freesurfer"}):
        wrapper = FreeSurferWrapper(
            str(bids_dataset),
            str(output_dir),
            freesurfer_license=str(freesurfer_license)
        )

        # Mock subprocess.run to raise an exception
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = Exception("Processing failed")

            # Process subject
            success = wrapper.process_subject("001", layout=mock_bids_layout)

            assert success is False
            assert "001" in wrapper.results["failure"]


def test_processing_summary(bids_dataset, output_dir, freesurfer_license):
    """Test generation of processing summary."""
    with patch.dict(os.environ, {"FREESURFER_HOME": "/opt/freesurfer"}):
        wrapper = FreeSurferWrapper(
            str(bids_dataset),
            str(output_dir),
            freesurfer_license=str(freesurfer_license)
        )

        # Add some results
        wrapper.results["success"].append("001")
        wrapper.results["failure"].append("002")
        wrapper.results["skipped"].append("003")

        # Get summary
        summary = wrapper.get_processing_summary()

        assert summary["total"] == 3
        assert summary["success"] == 1
        assert summary["failure"] == 1
        assert summary["skipped"] == 1
        assert "001" in summary["success_list"]
        assert "002" in summary["failure_list"]
        assert "003" in summary["skipped_list"]


def test_save_processing_summary(bids_dataset, output_dir, freesurfer_license):
    """Test saving processing summary to file."""
    with patch.dict(os.environ, {"FREESURFER_HOME": "/opt/freesurfer"}):
        wrapper = FreeSurferWrapper(
            str(bids_dataset),
            str(output_dir),
            freesurfer_license=str(freesurfer_license)
        )

        # Add some results
        wrapper.results["success"].append("001")
        wrapper.results["failure"].append("002")
        wrapper.results["skipped"].append("003")

        # Save summary
        summary_file = wrapper.save_processing_summary()

        assert os.path.exists(summary_file)
        with open(summary_file) as f:
            summary = json.load(f)
            assert "total" in summary
            assert "success" in summary
            assert "failure" in summary
            assert "skipped" in summary
            assert "timestamp" in summary
            assert "freesurfer_version" in summary


def test_cleanup(bids_dataset, output_dir, freesurfer_license):
    """Test cleanup of temporary files."""
    with patch.dict(os.environ, {"FREESURFER_HOME": "/opt/freesurfer"}):
        wrapper = FreeSurferWrapper(
            str(bids_dataset),
            str(output_dir),
            freesurfer_license=str(freesurfer_license)
        )

        # Create a temporary file
        temp_file = output_dir / "temp.txt"
        temp_file.touch()
        wrapper.temp_files.append(str(temp_file))

        # Run cleanup
        wrapper.cleanup()

        assert not temp_file.exists() 