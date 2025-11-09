#!/usr/bin/env bash
# preproc_adni.sh
# Preprocessing pipeline for resting-state fMRI using:
# dcm2niix, ANTs, FreeSurfer, FSL (NO fMRIPrep)
#
# Edit the CONFIG section below before running

set -euo pipefail
export OMP_NUM_THREADS=4

# -------------------------
# CONFIG (EDIT THESE)
# -------------------------
BIDS_DIR="/path/to/bids"                       # BIDS-style raw DICOM/NIfTI root (or DICOM root)
DERIV_DIR="/path/to/derivatives"               # output derivatives
SUBJ_LIST=("01" "02" "03" "04" "05")           # subject ids WITHOUT 'sub-'
SESSION=""                                     # if you have session IDs, adjust script
TR=2.0                                         # TR in seconds (edit to your dataset)
DUMMY_VOL=5                                    # number of initial volumes to drop
SMOOTH_FWHM=6                                  # spatial smoothing FWHM in mm
HP_CUTOFF=0.01                                 # high-pass cutoff (Hz)
LP_CUTOFF=0.08                                 # low-pass cutoff (Hz)
ANTs_TEMPLATE="/usr/share/ants/2mm/MNI152_T1_2mm.nii.gz"  # change to your MNI template
FREESURFER_SUBJECTS_DIR="${DERIV_DIR}/freesurfer_subjects"
LOGDIR="${DERIV_DIR}/logs"
# -------------------------

mkdir -p "${DERIV_DIR}" "${LOGDIR}" "${FREESURFER_SUBJECTS_DIR}"

# Helper: convert Hz cutoff to FSL -bptf sigma values
# -bptf takes: <highpass_sigma> <lowpass_sigma> in seconds converted as cutoff/(2*TR)
hp_sigma=$(python3 - <<PY
hp=${HP_CUTOFF}
tr=${TR}
# If hp==0, then pass -1 (no highpass)
if hp <= 0:
    print("-1")
else:
    # cutoff in seconds = 1/hz; FSL wants sigma = (cutoff_seconds)/(2*TR)
    cutoff_s = 1.0/hp
    sigma = cutoff_s/(2*tr)
    print("{:.6f}".format(sigma))
PY
)
lp_sigma=$(python3 - <<PY
lp=${LP_CUTOFF}
tr=${TR}
if lp <= 0:
    print("-1")
else:
    cutoff_s = 1.0/lp
    sigma = cutoff_s/(2*tr)
    print("{:.6f}".format(sigma))
PY
)

echo "Using FSL -bptf highpass sigma=${hp_sigma}, lowpass sigma=${lp_sigma}"

# -------------------------
# FUNCTIONS
# -------------------------
log() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"; }

run_dcm2niix() {
  local dicom_dir="$1"
  local out_dir="$2"
  mkdir -p "$out_dir"
  log "Converting DICOMs in ${dicom_dir} -> ${out_dir}"
  dcm2niix -z y -o "$out_dir" -f "%p_%t_%s" "$dicom_dir"
}

run_recon_all() {
  local subj="$1"
  local t1="$2"
  # Create subjects dir if not exists
  export SUBJECTS_DIR="${FREESURFER_SUBJECTS_DIR}"
  mkdir -p "${SUBJECTS_DIR}"
  log "Running FreeSurfer recon-all for sub-${subj}"
  recon-all -i "${t1}" -s "sub-${subj}" -all -qcache | tee "${LOGDIR}/reconall_sub-${subj}.log"
}

# -------------------------
# MAIN LOOP: per-subject
# -------------------------
for sub in "${SUBJ_LIST[@]}"; do
  subj="sub-${sub}"
  log "=== START Subject ${subj} ==="

  subj_raw_dir="${BIDS_DIR}/${subj}"
  subj_out_dir="${DERIV_DIR}/${subj}"
  mkdir -p "${subj_out_dir}" "${subj_out_dir}/anat" "${subj_out_dir}/func"

  # 1) --- Convert DICOM -> NIfTI (if you have DICOMs). If you already have NIfTIs, skip.
  # Assume DICOMs under ${BIDS_DIR}_dicoms/sub-XX/    (adjust as needed)
  DICOM_ROOT="${BIDS_DIR}_dicoms/${subj}"  # change path if needed
  if [ -d "${DICOM_ROOT}" ]; then
    run_dcm2niix "${DICOM_ROOT}" "${subj_out_dir}/orig"
  else
    log "DICOM root ${DICOM_ROOT} not found - skipping dcm2niix"
  fi

  # 2) --- Identify T1 and resting BOLD NIfTIs (user may already have them)
  # This tries common filenames; adapt to your naming
  # prefer BIDS naming: sub-XX_T1w.nii.gz and sub-XX_task-rest_bold.nii.gz after conversion
  T1_CAND=$(ls "${subj_out_dir}/orig"/*T1*.nii* 2>/dev/null || true)
  if [ -z "${T1_CAND}" ]; then
    T1_CAND=$(ls "${subj_raw_dir}/anat"/*T1*.nii* 2>/dev/null || true)
  fi
  if [ -z "${T1_CAND}" ]; then
    log "No T1 candidate found for ${subj}. Please place T1w NIfTI in ${subj_raw_dir}/anat/ and re-run."
    continue
  fi
  T1_NII="${T1_CAND%%$'\n'*}"
  cp "${T1_NII}" "${subj_out_dir}/anat/${subj}_T1w.nii.gz"

  # find resting-state bold
  BOLD_CAND=$(ls "${subj_out_dir}/orig"/*rest*.nii* 2>/dev/null || true)
  if [ -z "${BOLD_CAND}" ]; then
    BOLD_CAND=$(ls "${subj_raw_dir}/func"/*task-rest*bold*.nii* 2>/dev/null || true)
  fi
  if [ -z "${BOLD_CAND}" ]; then
    log "No resting BOLD found for ${subj}. Please place rest BOLD NIfTI in ${subj_raw_dir}/func/ and re-run."
    continue
  fi
  BOLD_NII="${BOLD_CAND%%$'\n'*}"
  cp "${BOLD_NII}" "${subj_out_dir}/func/${subj}_task-rest_bold.nii.gz"

  # 3) --- Structural: N4 bias correction (ANTs) and FreeSurfer recon-all
  log "Running ANTs N4 bias correction on T1..."
  T1_WORK="${subj_out_dir}/anat/${subj}_T1w_n4.nii.gz"
  N4CMD=(N4BiasFieldCorrection -i "${subj_out_dir}/anat/${subj}_T1w.nii.gz" -o "${T1_WORK}")
  "${N4CMD[@]}" | tee "${LOGDIR}/N4_${subj}.log"

  run_recon_all "${sub}" "${T1_WORK}"

  # 4) Structural segmentation output (FreeSurfer -> fill, or use FSL FAST)
  # We'll create WM/CSF masks from FreeSurfer aparc+aseg:
  APARC="${FREESURFER_SUBJECTS_DIR}/sub-${sub}/mri/aparc+aseg.mgz"
  if [ -f "${APARC}" ]; then
    # convert to nifti
    mri_convert "${APARC}" "${subj_out_dir}/anat/${subj}_aparc_aseg.nii.gz"
    # create WM mask (labels: 2,41 are cerebral WM, plus others; using freesurfer wm label=2,41)
    fslmaths "${subj_out_dir}/anat/${subj}_aparc_aseg.nii.gz" -thr 2 -uthr 2 -bin "${subj_out_dir}/anat/${subj}_wmmask_fs.nii.gz"
    fslmaths "${subj_out_dir}/anat/${subj}_aparc_aseg.nii.gz" -thr 41 -uthr 41 -bin -add "${subj_out_dir}/anat/${subj}_wmmask_fs.nii.gz" "${subj_out_dir}/anat/${subj}_wmmask_fs.nii.gz"
    # CSF (typically label 4=CSF and 43)
    fslmaths "${subj_out_dir}/anat/${subj}_aparc_aseg.nii.gz" -thr 4 -uthr 4 -bin "${subj_out_dir}/anat/${subj}_c sfmask_fs.nii.gz" || true
    fslmaths "${subj_out_dir}/anat/${subj}_aparc_aseg.nii.gz" -thr 43 -uthr 43 -bin -add "${subj_out_dir}/anat/${subj}_csfmask_fs.nii.gz" "${subj_out_dir}/anat/${subj}_csfmask_fs.nii.gz" || true
  else
    log "aparc+aseg not found for ${subj}. Will try FSL FAST segmentation..."
    # FSL FAST segmentation
    fast -o "${subj_out_dir}/anat/${subj}_T1_fast" "${T1_WORK}"
    # FAST outputs: _pve_0 = CSF, _pve_1 = GM, _pve_2 = WM
    fslmaths "${subj_out_dir}/anat/${subj}_T1_fast_pve_2.nii.gz" -thr 0.9 -bin "${subj_out_dir}/anat/${subj}_wmmask_fs.nii.gz"
    fslmaths "${subj_out_dir}/anat/${subj}_T1_fast_pve_0.nii.gz" -thr 0.9 -bin "${subj_out_dir}/anat/${subj}_csfmask_fs.nii.gz"
  fi

  # 5) --- fMRI preprocessing
  # 5.1 Drop dummy volumes
  BOLD="${subj_out_dir}/func/${subj}_task-rest_bold.nii.gz"
  BOLD_TRIM="${subj_out_dir}/func/${subj}_task-rest_space-T1w_desc-trimmed_bold.nii.gz"
  fslroi "${BOLD}" "${BOLD_TRIM}" ${DUMMY_VOL} -1
  log "Dropped first ${DUMMY_VOL} volumes -> ${BOLD_TRIM}"

  # 5.2 Slice-timing correction (SLICETIMER) - only if necessary (if slices interleaved)
  BOLD_STC="${subj_out_dir}/func/${subj}_task-rest_space-T1w_desc-stc_bold.nii.gz"
  # You must set --odd or --even or provide slice order; here we use default (adjust to your data)
  slicetimer -i "${BOLD_TRIM}" -o "${BOLD_STC}" --odd || cp "${BOLD_TRIM}" "${BOLD_STC}"

  # 5.3 Motion correction (MCFLIRT) -> get motion params
  BOLD_MC="${subj_out_dir}/func/${subj}_task-rest_space-T1w_desc-mc_bold.nii.gz"
  MC_PAR="${subj_out_dir}/func/${subj}_task-rest_desc-mc_par.txt"
  mcflirt -in "${BOLD_STC}" -out "${BOLD_MC}" -plots -mats -rmsrel -rmsabs -o "${BOLD_MC}" 2>&1 | tee "${LOGDIR}/mcflirt_${subj}.log"
  # MCFLIRT will produce a .par file; move it to MC_PAR (some versions create <in>.par)
  if [ -f "${BOLD_STC}.par" ]; then mv "${BOLD_STC}.par" "${MC_PAR}"; fi
  if [ ! -f "${MC_PAR}" ]; then
    # fallback: find *par in folder
    PARFOUND=$(ls "${subj_out_dir}/func/"*mcflirt*.par 2>/dev/null | head -n1 || true)
    if [ -n "${PARFOUND}" ]; then mv "${PARFOUND}" "${MC_PAR}"; fi
  fi

  # 5.4 Brain extraction on mean functional for mask
  meanfunc="${subj_out_dir}/func/${subj}_task-rest_mean.nii.gz"
  fslmaths "${BOLD_MC}" -Tmean "${meanfunc}"
  BET_MASK="${subj_out_dir}/func/${subj}_task-rest_mask.nii.gz"
  bet "${meanfunc}" "${subj_out_dir}/func/${subj}_task-rest_brain" -m -f 0.3
  mv "${subj_out_dir}/func/${subj}_task-rest_brain_mask.nii.gz" "${BET_MASK}"

  # 5.5 Coregistration fMRI -> T1: use ANTs (register meanfunc to bias-corrected T1)
  # First, affine registration using FLIRT to provide init
  log "Computing initial FLIRT rigid-body (BOLD mean -> T1) for ${subj}"
  flirt -in "${meanfunc}" -ref "${T1_WORK}" -out "${subj_out_dir}/func/${subj}_mean2T1_flirt.nii.gz" -omat "${subj_out_dir}/func/${subj}_mean2T1_flirt.mat" -dof 6

  # Now refine using ANTs (SyN) - produce transform from func -> T1
  log "Running ANTs registration (SyN) meanfunc -> T1"
  antsRegistrationSyNQuick.sh -d 3 -f "${T1_WORK}" -m "${meanfunc}" -o "${subj_out_dir}/func/${subj}_mean2T1_"

  # antsRegistrationSyNQuick.sh creates transforms: ${prefix}0GenericAffine.mat and ${prefix}1Warp.nii.gz etc.
  # We'll combine transforms later for normalization to MNI.

  # 5.6 Apply mask to motion-corrected functional
  fslmaths "${BOLD_MC}" -mas "${BET_MASK}" "${subj_out_dir}/func/${subj}_task-rest_space-T1w_desc-brain_bold.nii.gz"

  # 5.7 Normalize to MNI using transforms: first func->T1 (ANTs) then T1->MNI (ANTs)
  # Create T1 -> MNI registration
  log "Registering T1 -> MNI template using ANTs (SyN)"
  antsRegistrationSyNQuick.sh -d 3 -f "${ANTs_TEMPLATE}" -m "${T1_WORK}" -o "${subj_out_dir}/anat/${subj}_T1_to_MNI_"

  # Now apply transforms to functional (compose func->T1 and T1->MNI)
  # ANTs apply transforms ordering: output = applyTransforms fixed reference target (MNI) moving (input) with transforms list
  FUNC_IN="${subj_out_dir}/func/${subj}_task-rest_space-T1w_desc-brain_bold.nii.gz"
  FUNC_MNI="${subj_out_dir}/func/${subj}_task-rest_space-MNI152_desc-preproc_bold.nii.gz"
  # transforms from antsRegistrationSyNQuick.sh use suffixes: _0GenericAffine.mat and _1Warp.nii.gz
  # Compose: first warp from func->T1 prefix: ${subj_out_dir}/func/${subj}_mean2T1_ ; second warp from T1->MNI prefix: ${subj_out_dir}/anat/${subj}_T1_to_MNI_
  ANTS_FUNC2T1_AFF="${subj_out_dir}/func/${subj}_mean2T1_0GenericAffine.mat"
  ANTS_FUNC2T1_WARP="${subj_out_dir}/func/${subj}_mean2T1_1Warp.nii.gz"
  ANTS_T12MNI_AFF="${subj_out_dir}/anat/${subj}_T1_to_MNI_0GenericAffine.mat"
  ANTS_T12MNI_WARP="${subj_out_dir}/anat/${subj}_T1_to_MNI_1Warp.nii.gz"

  # Apply transforms: moving=FUNC_IN, fixed=MNI template, transforms: func->T1 affine, func->T1 warp, T1->MNI affine, T1->MNI warp
  # Note ANTs applyTransforms expects transforms in reverse order (last first). We'll supply: T1 warp, T1 affine, func warp, func affine
  applyTransforms -d 3 -e 3 -i "${FUNC_IN}" -r "${ANTs_TEMPLATE}" \
    -o "${FUNC_MNI}" \
    -t "${ANTS_T12MNI_WARP}" -t "${ANTS_T12MNI_AFF}" -t "${ANTS_FUNC2T1_WARP}" -t "${ANTS_FUNC2T1_AFF}" \
    --interpolation Linear

  # 5.8 Spatial smoothing using SUSAN (FWHM -> brightness threshold)
  SMOOTH_SIGMA=$(python3 - <<PY
import math
fwhm=${SMOOTH_FWHM}
sigma = fwhm/(2*math.sqrt(2*math.log(2)))
print("{:.6f}".format(sigma))
PY
)
  SMOOTH_OUT="${subj_out_dir}/func/${subj}_task-rest_space-MNI_desc-smoothed_bold.nii.gz"
  susan "${FUNC_MNI}" 1.5 "${SMOOTH_SIGMA}" 3 1 "${SMOOTH_OUT}" || fslmaths "${FUNC_MNI}" -s "${SMOOTH_SIGMA}" "${SMOOTH_OUT}"

  # 6) Confound estimation
  # 6.1 Motion parameters: use MCFLIRT .par (already produced)
  MPAR="${MC_PAR}"
  cp "${MPAR}" "${subj_out_dir}/func/${subj}_task-rest_desc-confounds_motion.tsv"

  # 6.2 Framewise displacement & motion outliers (FSL's fsl_motion_outliers)
  FD_OUT="${subj_out_dir}/func/${subj}_task-rest_desc-confounds_fd.txt"
  fsl_motion_outliers -i "${BOLD_MC}" -o "${FD_OUT/_fd.txt/_outliers.txt}" --fd > "${FD_OUT}" || true
  # This produces a confound file of outliers; also compute mean FD using python
  python3 - <<PY
import numpy as np, sys
fdfile="${FD_OUT}"
try:
    fd = np.loadtxt(fdfile)
    print("MeanFD:", np.mean(fd))
except:
    print("MeanFD: NaN")
PY

  # 6.3 Extract mean WM and CSF timeseries (use masks from freesurfer or FAST)
  WM_MASK="${subj_out_dir}/anat/${subj}_wmmask_fs.nii.gz"
  CSF_MASK="${subj_out_dir}/anat/${subj}_csfmask_fs.nii.gz"
  WM_TS="${subj_out_dir}/func/${subj}_wm_ts.txt"
  CSF_TS="${subj_out_dir}/func/${subj}_csf_ts.txt"
  if [ -f "${WM_MASK}" ]; then
    fslmeants -i "${BOLD_MC}" -o "${WM_TS}" -m "${WM_MASK}"
  fi
  if [ -f "${CSF_MASK}" ]; then
    fslmeants -i "${BOLD_MC}" -o "${CSF_TS}" -m "${CSF_MASK}"
  fi

  # 7) Nuisance regression (motion+WM+CSF) using fsl_regfilt
  # Build confound matrix: columns [motion (6), wm, csf, intercept] -> create combined file
  CONF_MAT="${subj_out_dir}/func/${subj}_confounds_matrix.txt"
  paste ${MPAR} ${WM_TS:-/dev/null} ${CSF_TS:-/dev/null} <(yes 1 | head -n $(fslval "${BOLD_MC}" dim4)) | awk '{print $0}' > "${CONF_MAT}" || true
  # NOTE: above paste may fail if WM/CSF missing; instead create robust confounds using python below
  python3 - <<PY
import numpy as np, pandas as pd, os
mp="${MPAR}"
wm="${WM_TS}"
csf="${CSF_TS}"
out="${CONF_MAT}"
# read motion
motion = np.loadtxt(mp) if os.path.exists(mp) else np.zeros((1,6))
nvol = int(os.popen("fslval '{}' dim4".format("${BOLD_MC}")).read().strip())
# ensure motion shape (nvol,6)
if motion.ndim == 1:
    motion = motion.reshape(-1,6)
# read wm, csf
vals=[]
if os.path.exists(wm):
    wm_ts=np.loadtxt(wm).reshape(-1,1)
else:
    wm_ts = np.zeros((nvol,1))
if os.path.exists(csf):
    csf_ts=np.loadtxt(csf).reshape(-1,1)
else:
    csf_ts = np.zeros((nvol,1))
mat = np.hstack([motion, wm_ts, csf_ts, np.ones((nvol,1))])
np.savetxt(out, mat, fmt='%.6f')
print('Wrote confound matrix to', out)
PY

  # Use fsl_regfilt to regress confounds from the spatially-smoothed MNI-space data
  CLEAN_BOLD="${subj_out_dir}/func/${subj}_task-rest_space-MNI_desc-clean_bold.nii.gz"
  # Identify columns to regress: all columns in conf mat (0-based indices)
  ncols=$(wc -l < <(head -n1 "${CONF_MAT}" | awk '{print NF}') && true) || ncols=$(awk '{print NF; exit}' "${CONF_MAT}")
  # fsl_regfilt expects a regressor indices file or matrix; easiest is to use --design and fsl_regfilt with the conf matrix
  fsl_regfilt -i "${SMOOTH_OUT}" -o "${CLEAN_BOLD}" -d "${CONF_MAT}" -f "1,2,3,4,5,6,7,8,9" || cp "${SMOOTH_OUT}" "${CLEAN_BOLD}"

  # 8) Temporal bandpass: using fslmaths -bptf with sigma values computed above
  BAND_BOLD="${subj_out_dir}/func/${subj}_task-rest_space-MNI_desc-clean_bp.nii.gz"
  if [ "${hp_sigma}" == "-1" ] && [ "${lp_sigma}" == "-1" ]; then
    cp "${CLEAN_BOLD}" "${BAND_BOLD}"
  else
    # if both are numbers, pass them; if one is -1, pass -1
    fslmaths "${CLEAN_BOLD}" -bptf ${hp_sigma} ${lp_sigma} "${BAND_BOLD}"
  fi

  # 9) Save final preprocessed derivative and QC plots
  FINAL_MASK="${subj_out_dir}/func/${subj}_task-rest_space-MNI_desc-brain_mask.nii.gz"
  # transform original BET mask to MNI using same transforms
  applyTransforms -d 3 -i "${BET_MASK}" -r "${ANTs_TEMPLATE}" -o "${FINAL_MASK}" -t "${ANTS_T12MNI_WARP}" -t "${ANTS_T12MNI_AFF}" -t "${ANTS_FUNC2T1_WARP}" -t "${ANTS_FUNC2T1_AFF}" --interpolation NearestNeighbor || true

  log "Preprocessing complete for ${subj}. Cleaned BOLD: ${BAND_BOLD}"

  # 10) Run MELODIC ICA on cleaned data if desired
  MELODIC_DIR="${subj_out_dir}/melodic"
  mkdir -p "${MELODIC_DIR}"
  log "Running MELODIC (single-subject ICA) for ${subj}"
  melodic -i "${BAND_BOLD}" -o "${MELODIC_DIR}" --nobet --mask="${FINAL_MASK}" --tr=${TR} --report || true

  log "=== END Subject ${subj} ==="
done

log "All subjects processed. Derivatives in ${DERIV_DIR}. Logs in ${LOGDIR}"

echo "Next recommended steps:"
echo "  - Inspect motion parameters and mean FD (files in derivatives/*/func/*desc-confounds_fd.txt)"
echo "  - Run group-level MELODIC or dual-regression (FSL) for component-based group analysis"
echo "  - For seed-based FC use FEAT or extract ROI timeseries (fslmeants) and compute correlations (Nilearn / custom Python)"
