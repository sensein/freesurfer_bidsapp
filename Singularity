Bootstrap: docker
From: vnmd/freesurfer_8.0.0

%files
    ./src /opt/src
    ./requirements.txt /opt/requirements.txt
    ./setup.py /opt/setup.py
    ./setup.cfg /opt/setup.cfg

%post
    # Create opt directory for application code
    mkdir -p /opt
    
    # Install pip using get-pip.py
    curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    python3 get-pip.py --user
    rm get-pip.py
    
    # Install Python dependencies in user mode to avoid permission issues
    cd /opt
    python3 -m pip install --user -r requirements.txt
    python3 -m pip install --user -e .

%environment
    # Set runtime license path to match BABS mount point
    export FS_LICENSE=/SGLR/FREESURFER_HOME/license.txt
    # Add opt and local Python packages to path
    export PYTHONPATH=/opt:$PYTHONPATH
    export PATH=/root/.local/bin:$PATH

%runscript
    # Execute the Python entry point directly
    python3 /opt/src/run.py "$@"

%help
    FreeSurfer 8.0.0 BIDS App

    This container is designed to work with BABS.

    Usage:
      singularity run -B [workdir] -B [license.txt]:/SGLR/FREESURFER_HOME/license.txt [container] [input_dir] [output_dir] participant --fs-license-file /SGLR/FREESURFER_HOME/license.txt [options]

    Example:
      singularity run -B $PWD -B license.txt:/SGLR/FREESURFER_HOME/license.txt freesurfer.sif $PWD/inputs/data/BIDS $PWD/outputs/freesurfer participant --fs-license-file /SGLR/FREESURFER_HOME/license.txt --skip-bids-validation --n_cpus 16 --participant-label sub-001