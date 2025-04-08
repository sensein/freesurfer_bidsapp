#!/usr/bin/env python3
"""
FreeSurfer-specific utility functions and configurations.
"""

# File mapping for FreeSurfer outputs to BIDS-compliant structure
FREESURFER_FILE_MAPPING = {
    # Critical volumes in MRI directory
    "mri/T1.mgz": "anat/T1.mgz",
    "mri/aparc+aseg.mgz": "anat/aparc+aseg.mgz",
    "mri/aparc.a2009s+aseg.mgz": "anat/aparc.a2009s+aseg.mgz",
    "mri/aparc.DKTatlas+aseg.mgz": "anat/aparc.DKTatlas+aseg.mgz",
    "mri/brainmask.mgz": "anat/brainmask.mgz",
    "mri/brain.mgz": "anat/brain.mgz",
    "mri/aseg.mgz": "anat/aseg.mgz",
    "mri/wm.mgz": "anat/wm.mgz",
    "mri/wmparc.mgz": "anat/wmparc.mgz",
    "mri/ribbon.mgz": "anat/ribbon.mgz",
    "mri/entowm.mgz": "anat/entowm.mgz",
    # Important surfaces
    "surf/lh.white": "surf/lh.white",
    "surf/rh.white": "surf/rh.white",
    "surf/lh.pial": "surf/lh.pial",
    "surf/rh.pial": "surf/rh.pial",
    "surf/lh.inflated": "surf/lh.inflated",
    "surf/rh.inflated": "surf/rh.inflated",
    "surf/lh.sphere.reg": "surf/lh.sphere.reg",
    "surf/rh.sphere.reg": "surf/rh.sphere.reg",
    "surf/lh.thickness": "surf/lh.thickness",
    "surf/rh.thickness": "surf/rh.thickness",
    "surf/lh.area": "surf/lh.area",
    "surf/rh.area": "surf/rh.area",
    "surf/lh.curv": "surf/lh.curv",
    "surf/rh.curv": "surf/rh.curv",
    "surf/lh.sulc": "surf/lh.sulc",
    "surf/rh.sulc": "surf/rh.sulc",
    # Essential stats
    "stats/aseg.stats": "stats/aseg.stats",
    "stats/lh.aparc.stats": "stats/lh.aparc.stats",
    "stats/rh.aparc.stats": "stats/rh.aparc.stats",
    "stats/lh.aparc.a2009s.stats": "stats/lh.aparc.a2009s.stats",
    "stats/rh.aparc.a2009s.stats": "stats/rh.aparc.a2009s.stats",
    "stats/lh.aparc.DKTatlas.stats": "stats/lh.aparc.DKTatlas.stats",
    "stats/rh.aparc.DKTatlas.stats": "stats/rh.aparc.DKTatlas.stats",
    "stats/wmparc.stats": "stats/wmparc.stats",
    "stats/brainvol.stats": "stats/brainvol.stats",
    "stats/entowm.stats": "stats/entowm.stats",
    # Critical labels/annotations
    "label/lh.aparc.annot": "label/lh.aparc.annot",
    "label/rh.aparc.annot": "label/rh.aparc.annot",
    "label/lh.aparc.a2009s.annot": "label/lh.aparc.a2009s.annot",
    "label/rh.aparc.a2009s.annot": "label/rh.aparc.a2009s.annot",
    "label/lh.aparc.DKTatlas.annot": "label/lh.aparc.DKTatlas.annot",
    "label/rh.aparc.DKTatlas.annot": "label/rh.aparc.DKTatlas.annot",
    # Processing logs and scripts
    "scripts/recon-all.log": "scripts/recon-all.log",
    "scripts/recon-all.done": "scripts/recon-all.done",
    "scripts/recon-all.env": "scripts/recon-all.env",
    "scripts/build-stamp.txt": "scripts/build-stamp.txt",
} 