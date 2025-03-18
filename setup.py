from setuptools import setup, find_packages
import subprocess
import sys

def build_docker():
    """Build Docker container"""
    print("Building Docker image...")
    try:
        subprocess.run(["docker", "build", "-t", "bids-freesurfer", "."], check=True)
        print("Docker image built successfully")
    except subprocess.CalledProcessError as e:
        print(f"Docker build failed: {e}")
        return False
    return True

def build_singularity():
    """Build Singularity container"""
    print("Building Singularity image...")
    try:
        # Try singularity command first, then apptainer if available
        if subprocess.run(["which", "singularity"], capture_output=True).returncode == 0:
            subprocess.run(["singularity", "build", "freesurfer.sif", "Singularity"], check=True)
        elif subprocess.run(["which", "apptainer"], capture_output=True).returncode == 0:
            subprocess.run(["apptainer", "build", "freesurfer.sif", "Singularity"], check=True)
        else:
            print("Neither singularity nor apptainer found. Cannot build image.")
            return False
        print("Singularity image built successfully")
    except subprocess.CalledProcessError as e:
        print(f"Singularity build failed: {e}")
        return False
    return True

# Check if we're being called with a container build command
if len(sys.argv) > 1 and sys.argv[1] in ["docker", "singularity", "containers"]:
    command = sys.argv[1]
    # Remove the custom argument so setup() doesn't see it
    sys.argv.pop(1)
    
    if command == "docker":
        build_docker()
    elif command == "singularity":
        build_singularity()
    elif command == "containers":
        build_docker()
        build_singularity()
    
    # Exit if we were just building containers
    if len(sys.argv) == 1:
        sys.exit(0)

setup(
    name="bids-freesurfer",
    version="0.1.0",
    description="BIDS App for FreeSurfer with NIDM Output",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    include_package_data=True,
    license="MIT",
    url="https://github.com/yourusername/bids-freesurfer",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    entry_points={
        "console_scripts": [
            "bids-freesurfer=src.run:main",
        ],
    },
    install_requires=[
        "click>=8.0.0",
        "pybids>=0.15.1",
        "nipype>=1.8.5",
        "nibabel>=5.0.0",
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "prov>=2.0.0",
        "rdflib>=6.0.0",
        'rapidfuzz>=2.0.0',
        "PyLD>=2.0.0",
    ],
    python_requires=">=3.9",
)