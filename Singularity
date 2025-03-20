Bootstrap: docker
From: vnmd/freesurfer_8.0.0

%post
    # Install additional Python dependencies
    apt-get update && apt-get install -y \
        python3 \
        python3-pip \
        python3-setuptools \
        git \
        && rm -rf /var/lib/apt/lists/*

    # Create opt directory for application code
    mkdir -p /opt
    
    # Install Python dependencies
    cd /opt
    pip3 install --no-cache-dir -r requirements.txt
    # If you have a setup.py, also install the package
    pip3 install -e .

%files
    ./src /opt/src
    ./requirements.txt /opt/requirements.txt
    ./setup.py /opt/setup.py
    ./setup.cfg /opt/setup.cfg

%environment
    # Set runtime license path to match BABS mount point
    export FS_LICENSE=/SGLR/FREESURFER_HOME/license.txt
    # Add opt to Python path
    export PYTHONPATH=/opt:$PYTHONPATH

%runscript
    # Execute the Python entry point directly, assuming input/output paths are provided as arguments
    python3 /opt/src/run.py "$@"

%help
    FreeSurfer 8.0.0 BIDS App

    This container is designed to work with BABS.

    Usage:
      singularity run -B [workdir] -B [license.txt]:/SGLR/FREESURFER_HOME/license.txt [container] [input_dir] [output_dir] participant --fs-license-file /SGLR/FREESURFER_HOME/license.txt [options]

    Example:
      singularity run -B $PWD -B license.txt:/SGLR/FREESURFER_HOME/license.txt freesurfer.sif $PWD/inputs/data/BIDS $PWD/outputs/freesurfer participant --fs-license-file /SGLR/FREESURFER_HOME/license.txt --skip-bids-validation --n_cpus 16 --participant-label sub-001