FROM ubuntu:22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    python3 \
    python3-pip \
    tcsh \
    bc \
    tar \
    libgomp1 \
    libxmu6 \
    libxt6 \
    perl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /tmp/
RUN pip3 install -r /tmp/requirements.txt

# Install FreeSurfer 8.0.0 (license will be mounted at runtime)
ENV FREESURFER_HOME=/opt/freesurfer
RUN wget -qO- https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/8.0.0/freesurfer_ubuntu22-8.0.0.tar.gz | tar xvz -C /opt
ENV PATH=$PATH:$FREESURFER_HOME/bin

# Create entrypoint script
COPY src/run.py /app/run.py
ENTRYPOINT ["python3", "/app/run.py"]