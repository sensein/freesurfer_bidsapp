FROM vnmd/freesurfer_8.0.0

# Install additional Python dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-setuptools \
    git \
    && rm -rf /var/lib/apt/lists/*

# =======================================
# Environment Configuration
# =======================================
# Make license path match BABS mount point
ENV FS_LICENSE=/SGLR/FREESURFER_HOME/license.txt

# =======================================
# BIDS App Setup
# =======================================
# Copy application files to a location that won't conflict
COPY . /opt/

# Install Python dependencies
WORKDIR /opt
RUN pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install -e .

# =======================================
# Runtime Configuration
# =======================================
# Entrypoint that expects input/output paths as arguments
ENTRYPOINT ["python3", "/opt/src/run.py"]