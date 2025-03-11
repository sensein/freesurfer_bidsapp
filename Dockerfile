FROM ubuntu:22.04

# =======================================
# System Dependencies
# =======================================
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
    dpkg \
    && rm -rf /var/lib/apt/lists/*

# =======================================
# FreeSurfer 8.0.0 Installation
# =======================================
# Download and install FreeSurfer 8.0.0 .deb package
RUN mkdir -p /tmp/fs && \
    echo "Downloading FreeSurfer 8.0.0 DEB package for Ubuntu 22..." && \
    wget --progress=dot:giga \
         -O /tmp/fs/freesurfer.deb \
         "https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/8.0.0/freesurfer_ubuntu22-8.0.0_amd64.deb" && \
    echo "Verifying download size..." && \
    if [ $(stat -c%s "/tmp/fs/freesurfer.deb") -gt 1000000000 ]; then \
        echo "Installing FreeSurfer 8.0.0..." && \
        dpkg -i /tmp/fs/freesurfer.deb || true && \
        apt-get update && \
        apt-get -f install -y && \
        echo "FreeSurfer 8.0.0 installed successfully"; \
    else \
        echo "Error: FreeSurfer 8.0.0 download incomplete. File too small." && \
        exit 1; \
    fi && \
    rm -rf /tmp/fs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# =======================================
# Environment Configuration
# =======================================
# FreeSurfer Environment Variables
ENV FREESURFER_HOME=/usr/local/freesurfer
ENV SUBJECTS_DIR=$FREESURFER_HOME/subjects
ENV FUNCTIONALS_DIR=$FREESURFER_HOME/sessions
ENV MINC_BIN_DIR=$FREESURFER_HOME/mni/bin
ENV MINC_LIB_DIR=$FREESURFER_HOME/mni/lib
ENV MNI_DIR=$FREESURFER_HOME/mni
ENV PERL5LIB=$FREESURFER_HOME/mni/lib/perl5/5.8.5
ENV MNI_PERL5LIB=$FREESURFER_HOME/mni/lib/perl5/5.8.5
ENV PATH=$FREESURFER_HOME/bin:$FREESURFER_HOME/tktools:$MINC_BIN_DIR:$PATH
ENV FS_LICENSE=$FREESURFER_HOME/license.txt

# Set runtime license path (for mounted license)
ENV FS_LICENSE=/license.txt

# =======================================
# BIDS App Setup
# =======================================
WORKDIR /app

# Copy application files
COPY . /app/

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install -e .

# =======================================
# Runtime Configuration
# =======================================
# Default command shows help
ENTRYPOINT ["python3", "/app/src/run.py"]
CMD ["--help"]