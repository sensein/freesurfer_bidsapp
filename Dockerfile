FROM ubuntu:22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    python3 \
    python3-pip \
    python3-setuptools \
    tcsh \
    bc \
    tar \
    git \
    libgomp1 \
    libxmu6 \
    libxt6 \
    perl \
    bzip2 \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Set up FreeSurfer environment
ENV FREESURFER_HOME=/opt/freesurfer
ENV SUBJECTS_DIR=$FREESURFER_HOME/subjects
ENV FUNCTIONALS_DIR=$FREESURFER_HOME/sessions
ENV MINC_BIN_DIR=$FREESURFER_HOME/mni/bin
ENV MINC_LIB_DIR=$FREESURFER_HOME/mni/lib
ENV MNI_DIR=$FREESURFER_HOME/mni
ENV PERL5LIB=$FREESURFER_HOME/mni/lib/perl5/5.8.5
ENV MNI_PERL5LIB=$FREESURFER_HOME/mni/lib/perl5/5.8.5
ENV PATH=$FREESURFER_HOME/bin:$FREESURFER_HOME/tktools:$MINC_BIN_DIR:$PATH

# Install FreeSurfer 8.0.0 (license will be mounted at runtime)
RUN wget -qO- https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/8.0.0/freesurfer_ubuntu22-8.0.0.tar.gz | tar xz -C /opt \
    && echo "Finished extracting FreeSurfer"

# Set up the BIDS app environment
WORKDIR /app

# Copy BIDS App files
COPY . /app/

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Install the app as a package (allowing for importable modules)
RUN pip3 install -e .

# Set FS_LICENSE path for runtime
ENV FS_LICENSE=/license.txt

# Default command shows help
ENTRYPOINT ["python3", "/app/src/run.py"]
CMD ["--help"]