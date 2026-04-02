import os
import json
import argparse
import hashlib
import re
import numpy as np
import nibabel as nib
from nilearn.image import resample_to_img
import logging

try:
    import torch
    TORCH_OK = True
except Exception:
    TORCH_OK = False

try:
    from scipy.ndimage import distance_transform_edt
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


def log(msg):
    """Print log messages in a unified format."""
    print(msg, flush=True)


def has_file(path):
    """Check whether the given path exists and is a file."""
    return os.path.exists(path) and os.path.isfile(path)


def find_nii(sub_dir, stem):
    """Find a NIfTI file with the given stem in a directory, preferring .nii.gz over .nii."""
    gz = os.path.join(sub_dir, stem + ".nii.gz")
    ni = os.path.join(sub_dir, stem + ".nii")
    if has_file(gz):
        return gz
    if has_file(ni):
        return ni
    return None


def list_subjects(root):
    """Scan the root directory and identify valid subject folders containing nsdgeneral."""
    subs = []
    for name in sorted(os.listdir(root)):
        sp = os.path.join(root, name)
        if not os.path.isdir(sp):
            continue
        epi = find_nii(sp, "nsdgeneral")
        if epi is not None:
            subs.append({"name": name, "dir": sp, "epi": epi})
    if not subs:
        raise FileNotFoundError("No valid subject was found (each subject folder must contain nsdgeneral.nii[.gz]).")
    return subs


def list_atlas_prefixes(sub_dir):
    """List all atlas prefixes in the current subject directory, excluding nsdgeneral."""
    prefs = set()
    for f in os.listdir(sub_dir):
        if f.endswith(".nii.gz"):
            stem = f[:-7]
        elif f.endswith(".nii"):
            stem = f[:-4]
        else:
            continue
        if stem == "nsdgeneral":
            continue
        prefs.add(stem)
    return prefs


def autodetect_common_atlases(subjects, include_hemis=False):
    """Automatically detect atlas prefixes shared by all subjects."""
    sets = []
    for s in subjects:
        prefs = list_atlas_prefixes(s["dir"])
        if not prefs:
            raise FileNotFoundError(f"Subject {s['name']} does not contain any atlas NIfTI file.")
        sets.append(set(prefs))
    inter = set.intersection(*sets)
    if not include_hemis:
        inter = {p for p in inter if not (p.startswith("lh.") or p.startswith("rh."))}
    return sorted(list(inter))


def get_mask_from_epi(epi_img):
    """Construct a mask from the EPI file by keeping only voxels whose value equals 1."""
    epi = epi_img.get_fdata()
    mask = (epi == 1)
    return mask


def get_mask_from_file(mask_path, epi_img, strict):
    """Construct a boolean mask from an external mask file and resample it to the EPI grid if needed."""
    m_img = nib.load(mask_path)
    same_shape = (m_img.shape == epi_img.shape)
    same_aff = np.allclose(m_img.affine, epi_img.affine, atol=1e-5)

    if not (same_shape and same_aff):
        if strict:
            raise ValueError(
                f"[ERROR] Mask grid mismatch: {os.path.basename(mask_path)} vs {os.path.basename(epi_img.get_filename())}"
            )
        m_img = resample_to_img(m_img, epi_img, interpolation="nearest")

    m = m_img.get_fdata()
    mask = np.isfinite(m) & (m > 0)
    return mask


def load_resampled_labelmap(atlas_path, epi_img, strict):
    """Load an atlas label map and resample it to the EPI grid when necessary."""
    a_img = nib.load(atlas_path)
    same_shape = a_img.shape == epi_img.shape
    same_aff = np.allclose(a_img.affine, epi_img.affine, atol=1e-5)

    if not (same_shape and same_aff):
        if strict:
            raise ValueError(
                f"[ERROR] Grid mismatch: {os.path.basename(atlas_path)} vs {os.path.basename(epi_img.get_filename())}"
            )
        a_img = resample_to_img(a_img, epi_img, interpolation="nearest")

    data = a_img.get_fdata()
    return np.round(data).astype(np.int32)


def voxel_indices_from_mask(mask):
    """Extract voxel coordinate indices from a mask, with output shape [N, 3]."""
    return np.stack(np.where(mask), axis=1).astype(np.int32)


def collect_global_label_ids(subjects, atlas_prefix, strict, exclude_set, mask_name=""):
    """Collect the union of atlas labels appearing inside masks across all subjects and return them in ascending order."""
    union = set()

    for s in subjects:
        epi_img = nib.load(s["epi"])

        arr = epi_img.get_fdata()
        n_val1 = int(np.count_nonzero(np.isfinite(arr) & (arr == 1)))
        n_finite_nonzero = int(np.count_nonzero(np.isfinite(arr) & (arr != 0)))
        logging.info(
            f"[EPI] {s['name']}: nsdgeneral==1 voxel count={n_val1} | (reference) finite and nonzero count={n_finite_nonzero}"
        )

        mask = get_mask_from_epi(epi_img)

        epi_data = epi_img.get_fdata()
        count_val1 = int(np.count_nonzero(epi_data == 1))
        count_finite_nonzero = int(np.count_nonzero(np.isfinite(epi_data) & (epi_data != 0)))
        log(
            f"[INFO] {s['name']}: voxel count with nsdgeneral == 1 = {count_val1} | (reference) finite and nonzero count = {count_finite_nonzero}"
        )

        N = int(mask.sum())
        if N == 0:
            log(f"[WARN] Subject {s['name']} has an empty mask. Skipped.")
            continue

        atlas_path = find_nii(s["dir"], atlas_prefix)
        if atlas_path is None:
            raise FileNotFoundError(f"Subject {s['name']} is missing {atlas_prefix}.nii[.gz]")

        lab = load_resampled_labelmap(atlas_path, epi_img, strict)
        vals = np.unique(lab[mask])

        for v in vals.tolist():
            if v not in exclude_set:
                union.add(int(v))

    union.discard(0)
    label_ids = np.array(sorted(list(union)), dtype=np.int32)
    return label_ids


def onehot_membership(labelmap, mask, label_ids):
    """Construct a one-hot ROI membership matrix from the label map."""
    N = int(mask.sum())
    K = len(label_ids)
    R = np.zeros((N, K), dtype=np.float32)

    col = {lab: i for i, lab in enumerate(label_ids)}
    flat = labelmap[mask]
    labels_present = np.unique(flat)
    labels_present = labels_present[labels_present != 0]

    for lab in labels_present:
        j = col.get(int(lab), None)
        if j is not None:
            R[:, j] = (flat == lab).astype(np.float32)

    return R


def edt_soft_membership(labelmap, mask, label_ids, tau, voxel_sizes):
    """Construct a soft-edt ROI membership matrix using distance transforms."""
    if not SCIPY_OK:
        raise ImportError("scipy is required when using --softmode soft-edt")

    N = int(mask.sum())
    K = len(label_ids)
    R = np.zeros((N, K), dtype=np.float32)

    for j, lab in enumerate(label_ids):
        target = (labelmap == lab)
        dist = distance_transform_edt(~target, sampling=voxel_sizes)
        R[:, j] = -dist[mask] / float(tau)

    R -= R.max(axis=1, keepdims=True)
    np.exp(R, out=R)
    R /= (R.sum(axis=1, keepdims=True) + 1e-8)

    return R


def save_subject_outputs(out_subject_dir, atlas_prefix, R, label_ids, voxel_idx, epi_img, float16):
    """Save the current subject's R matrix, label order, voxel indices, and metadata."""
    os.makedirs(out_subject_dir, exist_ok=True)

    if float16:
        R = R.astype(np.float16)

    if TORCH_OK:
        import torch
        torch.save(torch.from_numpy(R), os.path.join(out_subject_dir, f"{atlas_prefix}_R.pt"))
    else:
        np.savez_compressed(os.path.join(out_subject_dir, f"{atlas_prefix}_R.npz"), R=R)

    np.save(os.path.join(out_subject_dir, f"{atlas_prefix}_label_ids.npy"), label_ids)

    vi = os.path.join(out_subject_dir, "voxel_indices.npy")
    if not has_file(vi):
        np.save(vi, voxel_idx)

    meta_path = os.path.join(out_subject_dir, "meta.json")
    if not has_file(meta_path):
        meta = {
            "epi_shape": tuple(int(x) for x in epi_img.shape),
            "epi_affine_sha1": hashlib.sha1(epi_img.affine.astype(np.float32).tobytes()).hexdigest(),
            "float_dtype": "float16" if R.dtype == np.float16 else "float32",
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    subj_id = os.path.basename(out_subject_dir)
    atlas_name = atlas_prefix
    voxel_indices = voxel_idx
    log(
        f"[SAVE] subj={subj_id} atlas={atlas_name} | R.shape={R.shape} (N,K) "
        f"| label_ids={len(label_ids)} | voxel_indices={voxel_indices.shape}"
    )


def parse_args():
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Input root directory (one subdirectory per subject)")
    ap.add_argument("--out", required=True, help="Output root directory")
    ap.add_argument("--softmode", choices=["onehot", "soft-edt"], default="onehot", help="Soft-ROI construction mode")
    ap.add_argument("--tau", type=float, default=8.0, help="Temperature parameter for soft-edt")
    ap.add_argument("--float16", action="store_true", help="Store R in float16 format")
    ap.add_argument("--strict_grid", action="store_true", help="Raise an error immediately if grids do not match")
    ap.add_argument("--atlases", default="", help="Manually specify atlas prefixes, separated by commas")
    ap.add_argument("--include_hemis", action="store_true", help="Whether to include lh.* / rh.* during auto-detection")
    ap.add_argument("--exclude_labels", default="0,-1", help="Labels to exclude, default is 0,-1")
    ap.add_argument("--mask_name", default="", help="Optional custom mask filename")
    return ap.parse_args()


def main():
    """Run the main pipeline: detect subjects, build global labels, generate and save each subject's R matrices."""
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    exclude_set = set()
    for x in args.exclude_labels.split(","):
        x = x.strip()
        if x == "":
            continue
        try:
            exclude_set.add(int(x))
        except ValueError:
            pass
    exclude_set.add(0)

    subjects = list_subjects(args.root)
    log(f"[INFO] Detected subjects: {[s['name'] for s in subjects]}")

    if args.atlases.strip():
        atlas_list = [a.strip() for a in args.atlases.split(",") if a.strip()]
        log(f"[INFO] Using user-specified atlases: {atlas_list}")
    else:
        atlas_list = autodetect_common_atlases(subjects, include_hemis=args.include_hemis)
        if not atlas_list:
            raise RuntimeError("No atlas prefix shared by all subjects was found.")
        log(f"[INFO] Automatically detected common atlases: {atlas_list}")

    global_label_ids = {}
    for a in atlas_list:
        labs = collect_global_label_ids(
            subjects,
            a,
            strict=args.strict_grid,
            exclude_set=exclude_set,
            mask_name=args.mask_name
        )
        global_label_ids[a] = labs
        np.save(os.path.join(args.out, f"{a}_label_ids_global.npy"), labs)
        log(f"[INFO] {a}: number of global labels = {len(labs)} | example: {labs[:10].tolist()}")

    for s in subjects:
        epi_img = nib.load(s["epi"])

        arr = epi_img.get_fdata()
        n_val1 = int(np.count_nonzero(np.isfinite(arr) & (arr == 1)))
        n_finite_nonzero = int(np.count_nonzero(np.isfinite(arr) & (arr != 0)))
        log(f"[EPI] {s['name']}: nsdgeneral==1 voxel count={n_val1} | (reference) finite and nonzero count={n_finite_nonzero}")

        mask = None
        if args.mask_name:
            mp = find_nii(
                s["dir"],
                args.mask_name[:-7] if args.mask_name.endswith(".nii.gz")
                else args.mask_name[:-4] if args.mask_name.endswith(".nii")
                else args.mask_name
            )
            if mp is not None:
                mask = get_mask_from_file(mp, epi_img, strict=args.strict_grid)

        if mask is None:
            mask = get_mask_from_epi(epi_img)

        n_mask = int(np.count_nonzero(mask))
        log(f"[Mask] {s['name']}: number of voxels with mask == True = {n_mask}")

        N = int(mask.sum())
        if N == 0:
            log(f"[WARN] Subject {s['name']} has an empty mask. Skipped.")
            continue

        voxel_idx = voxel_indices_from_mask(mask)
        log(f"[VoxIdx] {s['name']}: voxel_indices.shape={voxel_idx.shape} (expected to be (N,3) or (N,))")

        voxel_sizes = np.sqrt((epi_img.affine[:3, :3] ** 2).sum(axis=0))

        out_subject_dir = os.path.join(args.out, s["name"])
        os.makedirs(out_subject_dir, exist_ok=True)
        log(f"[INFO] Processing subject {s['name']}: N_mask_voxels = {N}")

        for a in atlas_list:
            atlas_path = find_nii(s["dir"], a)
            if atlas_path is None:
                raise FileNotFoundError(f"Subject {s['name']} is missing {a}.nii[.gz]")

            labelmap = load_resampled_labelmap(atlas_path, epi_img, strict=args.strict_grid)
            labs_global = global_label_ids[a]

            if args.softmode == "onehot":
                R = onehot_membership(labelmap, mask, labs_global)
            else:
                R = edt_soft_membership(labelmap, mask, labs_global, tau=args.tau, voxel_sizes=voxel_sizes)

            assert R.shape[0] == N, f"Row count mismatch: R={R.shape}, N_mask={N}"
            assert R.shape[1] == len(labs_global), f"Column count mismatch: R={R.shape}, labels={len(labs_global)}"

            save_subject_outputs(out_subject_dir, a, R, labs_global, voxel_idx, epi_img, args.float16)

    summary = {
        "subjects": [s["name"] for s in subjects],
        "atlases": list(global_label_ids.keys()),
        "softmode": args.softmode,
        "tau": args.tau if args.softmode == "soft-edt" else None,
        "float16": bool(args.float16),
        "strict_grid": bool(args.strict_grid),
        "exclude_labels": sorted(list(exclude_set)),
        "include_hemis": bool(args.include_hemis),
        "mask_name": args.mask_name or None,
        "per_atlas_global_label_counts": {a: int(len(labs)) for a, labs in global_label_ids.items()},
    }

    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log(f"[DONE] All processing is complete. Output directory: {args.out}")


if __name__ == "__main__":
    main()