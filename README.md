# BIDS FreeSurfer App

A BIDS App implementation for FreeSurfer 8.0.0 that provides standardized surface reconstruction and morphometric analysis with NIDM output.

## Description

This BIDS App runs FreeSurfer's `recon-all` pipeline on structural T1w and (optionally) T2w images from a BIDS-valid dataset. It organizes outputs in a BIDS-compliant derivatives structure and provides additional NIDM format outputs for improved interoperability.

The app implements:
1. Automatic identification and processing of T1w images (required)
2. Utilization of T2w images when available (optional)
3. Multi-session data handling with appropriate processing paths
4. NIDM format output generation for standardized data exchange
5. BIDS provenance documentation for reproducibility

## Installation

### Requirements

- Docker (for containerized execution)
- FreeSurfer license file (obtainable from https://surfer.nmr.mgh.harvard.edu/registration.html)

### Quick Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/bids-freesurfer.git
cd bids-freesurfer

# Build the Docker image
docker build -t bids/freesurfer:8.0.0 .
```

## Usage

### Basic Command

```bash
docker run -v /path/to/bids_dataset:/bids_dataset:ro \
           -v /path/to/output:/output \
           -v /path/to/freesurfer/license.txt:/license.txt \
           bids/freesurfer:8.0.0 \
           /bids_dataset /output participant
```

### Command-Line Arguments

- Positional arguments:
  - `bids_dir`: The directory with the input BIDS dataset
  - `output_dir`: The directory where the output files should be stored
  - `analysis_level`: Level of the analysis that will be performed. Options are: participant, group

- Optional arguments:
  - `--participant_label`: The label(s) of the participant(s) to analyze (without "sub-" prefix)
  - `--session_label`: The label(s) of the session(s) to analyze (without "ses-" prefix)
  - `--freesurfer_license`: Path to FreeSurfer license file
  - `--skip_bids_validator`: Skip BIDS validation
  - `--fs_options`: Additional options to pass to recon-all (e.g., "-parallel -openmp 4")
  - `--skip_nidm`: Skip NIDM output generation

### Examples

Process a single subject:
```bash
docker run -v /path/to/bids_dataset:/bids_dataset:ro \
           -v /path/to/output:/output \
           -v /path/to/freesurfer/license.txt:/license.txt \
           bids/freesurfer:8.0.0 \
           /bids_dataset /output participant --participant_label 01
```

Process multiple subjects in parallel (using GNU parallel):
```bash
docker run -v /path/to/bids_dataset:/bids_dataset:ro \
           -v /path/to/output:/output \
           -v /path/to/freesurfer/license.txt:/license.txt \
           bids/freesurfer:8.0.0 \
           /bids_dataset /output participant --fs_options="-parallel -openmp 4" \
           --participant_label 01 02 03
```

## Outputs

### Output Directory Structure

```
<output_dir>/
├── dataset_description.json
├── freesurfer/
│   ├── dataset_description.json
│   ├── sub-<participant_label>/
│   │   ├── anat/
│   │   │   ├── aparc+aseg.mgz
│   │   │   ├── aseg.mgz
│   │   │   ├── brainmask.mgz
│   │   │   └── T1.mgz
│   │   ├── label/
│   │   ├── stats/
│   │   │   ├── aseg.stats
│   │   │   ├── lh.aparc.stats
│   │   │   └── rh.aparc.stats
│   │   ├── surf/
│   │   │   ├── lh.pial
│   │   │   ├── lh.white
│   │   │   ├── rh.pial
│   │   │   └── rh.white
│   │   └── provenance.json
└── nidm/
    ├── dataset_description.json
    └── sub-<participant_label>/
        └── prov.jsonld
```

### FreeSurfer Output

The FreeSurfer outputs follow standard FreeSurfer conventions but are organized in a BIDS-compliant directory structure. Key output files include:

- Segmentation volumes (`aparc+aseg.mgz`, `aseg.mgz`)
- Surface meshes (`lh.white`, `rh.white`, `lh.pial`, `rh.pial`)
- Statistical measures (`aseg.stats`, `lh.aparc.stats`, `rh.aparc.stats`)

### NIDM Output

The NIDM outputs are provided in JSON-LD format (`prov.jsonld`), which includes:

- FreeSurfer version information
- Processing provenance
- Volume measurements for brain structures
- Cortical thickness and surface area measurements
- Standard identifiers for interoperability

## License

This BIDS App is licensed under [MIT License](LICENSE).

## Acknowledgments

- FreeSurfer (https://surfer.nmr.mgh.harvard.edu/)
- BIDS (https://bids.neuroimaging.io/)
- NIDM (http://nidm.nidash.org/)

## References

If you use this BIDS App in your research, please cite:

1. Fischl B. (2012). FreeSurfer. NeuroImage, 62(2), 774–781. https://doi.org/10.1016/j.neuroimage.2012.01.021
2. Gorgolewski, K. J., Auer, T., Calhoun, V. D., Craddock, R. C., Das, S., Duff, E. P., Flandin, G., Ghosh, S. S., Glatard, T., Halchenko, Y. O., Handwerker, D. A., Hanke, M., Keator, D., Li, X., Michael, Z., Maumet, C., Nichols, B. N., Nichols, T. E., Pellman, J., Poline, J. B., … Poldrack, R. A. (2016). The brain imaging data structure, a format for organizing and describing outputs of neuroimaging experiments. Scientific data, 3, 160044. https://doi.org/10.1038/sdata.2016.44
3. Maumet, C., Auer, T., Bowring, A., Chen, G., Das, S., Flandin, G., Ghosh, S., Glatard, T., Gorgolewski, K. J., Helmer, K. G., Jenkinson, M., Keator, D. B., Nichols, B. N., Poline, J. B., Reynolds, R., Sochat, V., Turner, J., & Nichols, T. E. (2016). Sharing brain mapping statistical results with the neuroimaging data model. Scientific data, 3, 160102. https://doi.org/10.1038/sdata.2016.102