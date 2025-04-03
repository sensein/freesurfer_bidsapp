Bootstrap: docker
From: vnmd/freesurfer_8.0.0

%files
    ./src /opt/src
    ./requirements.txt /opt/requirements.txt
    ./setup.py /opt/setup.py
    ./setup.cfg /opt/setup.cfg
    ./VERSION /opt/VERSION

%post
    # Create opt directory for application code
    mkdir -p /opt
    
    # Install pip using get-pip.py
    curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    python3 get-pip.py
    rm get-pip.py
    
    # Install Python dependencies
    cd /opt
    pip install -r requirements.txt
    pip install -e .

%environment
    # Set runtime license path to match BABS mount point
    export FS_LICENSE=/SGLR/FREESURFER_HOME/license.txt
    # Add opt to Python path
    export PYTHONPATH=/opt:$PYTHONPATH
    # Add Python packages to path
    export PATH=/usr/local/bin:$PATH

%runscript
    # Execute the Python entry point directly
    python3 /opt/src/run.py "$@"

%help
    FreeSurfer BIDS App 0.1.0

    This container is designed to work with BABS.

    Version Information:
    - BIDS-FreeSurfer: 0.1.0
    - FreeSurfer: 8.0.0
    - Base Image: vnmd/freesurfer_8.0.0

    Usage:
      singularity run -B [workdir] -B [license.txt]:/SGLR/FREESURFER_HOME/license.txt [container] [input_dir] [output_dir] participant --fs-license-file /SGLR/FREESURFER_HOME/license.txt [options]

    Example:
      singularity run -B $PWD -B license.txt:/SGLR/FREESURFER_HOME/license.txt freesurfer_bids_app.sif $PWD/inputs/data/BIDS $PWD/outputs/freesurfer participant --fs-license-file /SGLR/FREESURFER_HOME/license.txt --skip-bids-validation 