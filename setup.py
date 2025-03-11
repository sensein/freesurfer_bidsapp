from setuptools import setup, find_packages

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
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    entry_points={
        "console_scripts": [
            "bids-freesurfer=src.run:main",
        ],
    },
    install_requires=[
        "click>=8.0.0",  # Using standard Click instead of ClickBit
        "pybids>=0.15.1",
        "nipype>=1.8.5",
        "nibabel>=5.0.0",
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "prov>=2.0.0",
        "rdflib>=6.0.0",
        "PyLD>=2.0.0",  # For better JSON-LD support
        "jsonld>=1.0.0",
    ],
    python_requires=">=3.8",
)