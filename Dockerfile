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
