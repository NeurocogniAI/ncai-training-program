"""
rsfMRI Functional Connectivity Analysis - FULL Python Solution
------------------------------------------------------------
This single-file solution provides a reproducible pipeline for:
 1. Running (or preparing) fMRIPrep preprocessing (Docker/Singularity command shown)
 2. Loading fMRIPrep outputs
 3. Quality control (FD, tSNR)
 4. Denoising (aCompCor + motion regression) and temporal filtering
 5. ROI-ROI connectivity matrices (Fisher Z)
 6. Seed-based correlation maps (PCC example)
 7. Basic group statistics on ROI edges (t-test + FDR)
 8. Saving outputs (NIfTI maps, CSVs, .npy matrices)

Usage:
 - Edit the CONFIG section below (paths, subjects list, atlas, TR, t_r)
 - Ensure Python environment with required packages (nilearn, nibabel, numpy, pandas,
   scipy, statsmodels, matplotlib) is active.
 - fMRIPrep must be run separately (command included) or you can call run_fmriprep() in this
   script (requires Docker/Singularity available and configured).

Notes:
 - This script assumes fMRIPrep outputs are in BIDS-derivatives layout produced by fMRIPrep
   with spatial normalization to MNI (e.g., space-MNI152NLin2009cAsym).
 - Adjust bandpass (high_pass, low_pass) and confounds selection to taste.

"""

import os
import sys
import json
import glob
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import image, masking
from nilearn.input_data import NiftiLabelsMasker, NiftiMasker, NiftiSpheresMasker
from nilearn.signal import clean
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt

# -----------------------
# CONFIG (edit these)
# -----------------------
BIDS_ROOT = '/path/to/bids'            # BIDS root (raw dataset)
DERIV_ROOT = '/path/to/derivatives'    # derivatives output (fmriprep output)
FMRIPREP_IMAGE = 'poldracklab/fmriprep:latest'  # Docker image name (if running)
FS_LICENSE = '/path/to/license.txt'    # FreeSurfer license if needed
SUBJECTS = [                            # list of subject IDs as in BIDS (without 'sub-')
    '01','02','03','04','05','06','07','08'
]
SPACE = 'MNI152NLin2009cAsym'           # space used by fmriprep outputs
TR = 2.0                                # repetition time in seconds
HIGH_PASS = 0.01
LOW_PASS = 0.08
ATLAS_IMG = '/path/to/atlas_labels.nii.gz'  # atlas for ROI-ROI (labels must be integers)
ATLAS_LABELS = '/path/to/atlas_labels.csv'  # CSV with columns ['label','name']
SEED_COORDS = [(0, -52, 26)]            # PCC seed in MNI coords (example)
SEED_RADIUS = 6                          # mm
OUTPUT_DIR = './fc_results'

# confound selection settings
N_COMPCOR = 5   # number of aCompCor components to use
USE_GLOBAL_SIGNAL = False
MOTION_PARAMS = True

# create outputs dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------
# Helper functions
# -----------------------

def run_fmriprep(bids_root=BIDS_ROOT, out_root=DERIV_ROOT, working_dir=None, subjects=None):
    """Run fMRIPrep using Docker. Change to Singularity if desired.
    This function constructs and executes a docker run command. It doesn't capture
    logs elegantly; for long runs prefer running on the command line.
    """
    if subjects is None:
        subj_flag = ''
    else:
        subj_flag = ' '.join([f'--participant-label {s}' for s in subjects])

    cmd = (
        f"docker run --rm -it "
        f"-v {bids_root}:/data:ro "
        f"-v {out_root}:/out "
        f"-v {FS_LICENSE}:{FS_LICENSE} "
        f"{FMRIPREP_IMAGE} "/data /out participant "
        f"--fs-license-file {FS_LICENSE} --nthreads 8 --omp-nthreads 8 "
        f"--output-spaces {SPACE}:res-2 --use-aroma "
        f"{subj_flag}"
    )
    print('Running fmriprep with command:')
    print(cmd)
    subprocess.run(cmd, shell=True, check=True)


def find_bold_and_confounds(sub):
    """Locate fMRIPrep preprocessed bold and confounds TSV for a subject's resting run.
       Returns list of dicts per run: {'bold':path, 'confounds':path, 'mask':path}
    """
    subj = f'sub-{sub}'
    pattern = os.path.join(DERIV_ROOT, 'fmriprep', subj, 'func', f'{subj}_*_task-rest_*space-{SPACE}_*desc-preproc_bold.nii.gz')
    bolds = sorted(glob.glob(pattern))
    results = []
    for b in bolds:
        base = Path(b).name
        # confounds path pattern
        conf_pattern = os.path.join(DERIV_ROOT, 'fmriprep', subj, 'func', base.replace('_desc-preproc_bold.nii.gz', '_desc-confounds_regressors.tsv'))
        mask_pattern = os.path.join(DERIV_ROOT, 'fmriprep', subj, 'func', base.replace('_desc-preproc_bold.nii.gz', '_space-'+SPACE+'_desc-brain_mask.nii.gz'))
        confs = conf_pattern if os.path.exists(conf_pattern) else None
        mask = mask_pattern if os.path.exists(mask_pattern) else None
        results.append({'bold': b, 'confounds': confs, 'mask': mask})
    return results


def load_confounds(confounds_tsv):
    """Load fmriprep confounds TSV and select desired columns.
    Returns a DataFrame.
    """
    df = pd.read_csv(confounds_tsv, sep='\t')
    # Generate basic FD metrics if present
    # Select motion params
    conf_list = []
    if MOTION_PARAMS:
        mot_cols = [c for c in df.columns if c.startswith('trans_') or c.startswith('rot_')]
        conf_list += mot_cols
    # aCompCor components: columns like 'a_comp_cor_00'
    acomp = [c for c in df.columns if c.startswith('a_comp_cor')]
    acol = acomp[:N_COMPCOR]
    conf_list += acol
    # Global signal
    if USE_GLOBAL_SIGNAL and 'global_signal' in df.columns:
        conf_list += ['global_signal']
    # Also include derivatives if available
    derivs = [c for c in df.columns if c.endswith('_derivative1')]
    # We'll include derivatives of motion
    conf_list += [c for c in df.columns if c.endswith('_derivative1') and (c.replace('_derivative1','').startswith('trans_') or c.replace('_derivative1','').startswith('rot_'))]

    # intersection and drop NaNs
    conf_list = [c for c in conf_list if c in df.columns]
    conf_df = df[conf_list].fillna(0.0)
    # also keep framewise_displacement for QC
    if 'framewise_displacement' in df.columns:
        conf_df['framewise_displacement'] = df['framewise_displacement']
    return conf_df


def compute_fd_metrics(conf_df, fd_thresh=0.5):
    """Compute mean FD, max FD, and count of FD > threshold.
    conf_df should contain 'framewise_displacement' column.
    """
    if 'framewise_displacement' not in conf_df.columns:
        return {'mean_FD': np.nan, 'max_FD': np.nan, 'n_fd_gt': np.nan}
    fd = conf_df['framewise_displacement'].values
    return {'mean_FD': float(np.nanmean(fd)), 'max_FD': float(np.nanmax(fd)), 'n_fd_gt': int(np.sum(fd > fd_thresh))}


def compute_tSNR(bold_img, mask_img=None):
    """Compute tSNR: mean signal / std across time within mask.
    Returns mean tSNR within mask.
    """
    img = image.load_img(bold_img)
    data = img.get_fdata()
    # data shape: X Y Z T
    mean_img = np.mean(data, axis=3)
    std_img = np.std(data, axis=3, ddof=1)
    tSNR = np.zeros(mean_img.shape)
    with np.errstate(divide='ignore', invalid='ignore'):
        tSNR = np.where(std_img != 0, mean_img / std_img, 0)
    if mask_img is None:
        mask = np.any(data != 0, axis=3)
    else:
        mask = image.load_img(mask_img).get_fdata().astype(bool)
    # return mean within mask
    vals = tSNR[mask]
    return float(np.nanmean(vals))


def denoise_and_filter(bold_img, conf_df, mask_img, t_r=TR, low_pass=LOW_PASS, high_pass=HIGH_PASS):
    """Denoise using selected confounds and apply band-pass filtering using nilearn.signal.clean.
    Returns a 2D array: (n_timepoints, n_voxels) standardized.
    """
    # Load bold and mask
    masker = NiftiMasker(mask_img=mask_img, standardize=True, detrend=True,
                         low_pass=low_pass, high_pass=high_pass, t_r=t_r)
    # Use confounds matrix (columns already selected
    confounds = conf_df.drop(columns=[c for c in conf_df.columns if c=='framewise_displacement'], errors='ignore').values
    # fit_transform returns (n_timepoints, n_voxels)
    time_series = masker.fit_transform(bold_img, confounds=confounds)
    return time_series, masker


def extract_roi_timeseries_from_img(time_series_img, atlas_img, t_r=TR, low_pass=LOW_PASS, high_pass=HIGH_PASS):
    """Extract ROI time series using a labels atlas image.
    time_series_img: path to 4D bold image (preprocessed / denoised)
    Returns (time_series matrix: n_timepoints x n_rois, labels list)
    """
    labels_df = pd.read_csv(ATLAS_LABELS)
    labels = labels_df['label'].tolist()
    names = labels_df['name'].tolist()
    masker = NiftiLabelsMasker(labels_img=ATLAS_IMG, standardize=True, detrend=True,
                               low_pass=low_pass, high_pass=high_pass, t_r=t_r)
    ts = masker.fit_transform(time_series_img)
    return ts, labels, names


def compute_connectivity_matrix(timeseries):
    """Compute Pearson correlation matrix and return Fisher Z transformed matrix.
    timeseries: (n_timepoints x n_rois)
    returns: (n_rois x n_rois) fisher z
    """
    # ensure (timepoints, rois)
    if timeseries.shape[0] < timeseries.shape[1]:
        # likely (n_rois, n_timepoints) - transpose
        ts = timeseries.T
    else:
        ts = timeseries
    corr = np.corrcoef(ts.T)
    # numerical safety
    corr = np.nan_to_num(corr)
    # Fisher Z
    with np.errstate(divide='ignore', invalid='ignore'):
        z = np.arctanh(np.clip(corr, -0.999999, 0.999999))
    return z


def save_matrix(mat, fname):
    np.save(fname, mat)


def seed_based_map(bold_img, mask_img, seed_coords, seed_radius, t_r=TR, low_pass=LOW_PASS, high_pass=HIGH_PASS):
    """Compute seed-based correlation map for one seed coordinate.
    Returns a nilearn image (z-map) of Fisher Z values.
    """
    # Seed time series
    seed_masker = NiftiSpheresMasker(seed_coords, radius=seed_radius, standardize=True,
                                     detrend=True, low_pass=low_pass, high_pass=high_pass, t_r=t_r)
    seed_ts = seed_masker.fit_transform(bold_img)
    # whole-brain
    brain_masker = NiftiMasker(mask_img=mask_img, standardize=True, detrend=True,
                               low_pass=low_pass, high_pass=high_pass, t_r=t_r)
    brain_ts = brain_masker.fit_transform(bold_img)
    # correlation: brain voxels x 1
    # brain_ts shape: n_time x n_voxels
    # seed_ts shape: n_time x 1
    # compute Pearson as dot product if standardized
    corr = np.dot(brain_ts.T, seed_ts)[:, 0] / (brain_ts.shape[0] - 1)
    # But nilearn's standardize True will ensure zero mean unit std so dot gives correlation * (n-1)
    # To be safe, compute correlation explicitly
    from scipy.stats import pearsonr
    # compute voxel-by-voxel correlation might be slow; we'll compute using matrix ops
    # calculate corr = (brain_ts.T @ seed_ts) / (n_timepoints - 1) is not exact; better to compute manually
    # Use np.corrcoef between seed and each voxel
    seed = seed_ts[:, 0]
    n_time = brain_ts.shape[0]
    mean_seed = seed.mean()
    std_seed = seed.std(ddof=0)
    mean_brain = brain_ts.mean(axis=0)
    std_brain = brain_ts.std(axis=0, ddof=0)
    denom = std_seed * std_brain * n_time
    # covariance
    cov = (brain_ts - mean_brain) * (seed[:, None] - mean_seed)
    cov = cov.sum(axis=0)
    # correlation
    corr_vox = cov / denom
    corr_vox = np.nan_to_num(corr_vox)
    # Fisher z
    z_vox = np.arctanh(np.clip(corr_vox, -0.999999, 0.999999))
    # convert back to image
    corr_img = brain_masker.inverse_transform(z_vox)
    return corr_img


def group_stats_roi(edge_matrices_control, edge_matrices_patient, alpha=0.05):
    """Perform t-test per edge and FDR correction.
    Inputs: lists or arrays (n_sub x n_edges) for each group
    Returns: DataFrame with columns ['edge','t','p','p_fdr','significant']
    """
    Xc = np.vstack(edge_matrices_control)
    Xp = np.vstack(edge_matrices_patient)
    tvals, pvals = ttest_ind(Xc, Xp, axis=0, equal_var=False, nan_policy='omit')
    # FDR
    rej, pvals_fdr, _, _ = multipletests(pvals, alpha=alpha, method='fdr_bh')
    df = pd.DataFrame({'t': tvals, 'p': pvals, 'p_fdr': pvals_fdr, 'signif': rej})
    return df

# -----------------------
# Main processing workflow
# -----------------------

def main():
    subjects = SUBJECTS

    # QC summary list
    qc_rows = []

    # containers for ROI matrices per subject
    roi_matrices = {}
    seed_maps = {}

    for sub in subjects:
        print(f'Processing {sub}')
        runs = find_bold_and_confounds(sub)
        if len(runs) == 0:
            print(f'No resting runs found for subject {sub} in {DERIV_ROOT}. Skipping.')
            continue
        # For simplicity take the first resting run
        run = runs[0]
        bold = run['bold']
        conf = run['confounds']
        mask = run['mask']
        if conf is None or mask is None:
            print(f'Missing confounds or mask for {sub}. Paths: ', conf, mask)
            continue
        conf_df = load_confounds(conf)
        fd_metrics = compute_fd_metrics(conf_df)
        tSNR = compute_tSNR(bold, mask)
        qc_rows.append({'subject': sub, 'mean_FD': fd_metrics['mean_FD'], 'max_FD': fd_metrics['max_FD'], 'n_fd_gt_0.5': fd_metrics['n_fd_gt'], 'tSNR': tSNR})

        # Denoise & filter
        print('  Denoising and filtering...')
        ts, masker = denoise_and_filter(bold, conf_df, mask, t_r=TR, low_pass=LOW_PASS, high_pass=HIGH_PASS)
        # ts shape: n_time x n_voxels

        # Save cleaned 4D image (reconstruct)
        clean_img = masker.inverse_transform(ts)
        clean_img_path = os.path.join(OUTPUT_DIR, f'sub-{sub}_task-rest_cleaned_space-{SPACE}.nii.gz')
        clean_img.to_filename(clean_img_path)
        print('  Saved cleaned image to', clean_img_path)

        # ROI timeseries
        print('  Extracting ROI timeseries...')
        roi_ts, labels, names = extract_roi_timeseries_from_img(clean_img_path, ATLAS_IMG, t_r=TR)
        # compute connectivity
        zmat = compute_connectivity_matrix(roi_ts)
        roi_matrices[sub] = zmat
        save_matrix(zmat, os.path.join(OUTPUT_DIR, f'sub-{sub}_roi_zcorr.npy'))

        # seed-based map(s)
        print('  Computing seed-based maps...')
        seed_imgs = []
        for i, coords in enumerate(SEED_COORDS):
            zimg = seed_based_map(clean_img_path, mask, [coords], SEED_RADIUS, t_r=TR)
            out_seed = os.path.join(OUTPUT_DIR, f'sub-{sub}_seed{i}_zmap_space-{SPACE}.nii.gz')
            zimg.to_filename(out_seed)
            seed_imgs.append(out_seed)
        seed_maps[sub] = seed_imgs

    # Save QC
    qc_df = pd.DataFrame(qc_rows)
    qc_df.to_csv(os.path.join(OUTPUT_DIR, 'qc_summary.csv'), index=False)
    print('QC table saved to', os.path.join(OUTPUT_DIR, 'qc_summary.csv'))

    # --- Group statistics example ---
    # For demonstration we assume first half of SUBJECTS = control, second half = patient
    n = len(subjects)
    mid = n // 2
    controls = subjects[:mid]
    patients = subjects[mid:]

    # Build edge vectors: flatten upper triangle (excluding diagonal)
    def upper_tri_flat(mat):
        iu = np.triu_indices(mat.shape[0], k=1)
        return mat[iu]

    control_edges = []
    patient_edges = []
    for s in controls:
        if s in roi_matrices:
            control_edges.append(upper_tri_flat(roi_matrices[s]))
    for s in patients:
        if s in roi_matrices:
            patient_edges.append(upper_tri_flat(roi_matrices[s]))

    if len(control_edges) > 0 and len(patient_edges) > 0:
        print('Running group statistics on ROI edges...')
        res_df = group_stats_roi(control_edges, patient_edges, alpha=0.05)
        res_df.to_csv(os.path.join(OUTPUT_DIR, 'roi_edge_group_stats.csv'), index=False)
        print('Group stats saved to', os.path.join(OUTPUT_DIR, 'roi_edge_group_stats.csv'))

    print('All done. Outputs in', OUTPUT_DIR)


if __name__ == '__main__':
    main()
