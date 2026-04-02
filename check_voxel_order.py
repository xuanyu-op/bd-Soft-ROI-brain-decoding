import argparse
import numpy as np
import nibabel as nib
from collections import Counter


def as_bool_mask_from_label(img_data, label_value=1):
    """Generate a boolean mask according to the given label value."""
    return (img_data == label_value)


def flat_indices_from_mask(mask_bool):
    """Flatten the boolean mask in C-order and return the linear indices of True entries."""
    return np.flatnonzero(mask_bool.ravel(order="C"))


def ensure_zero_based_coords(coords, shape):
    """Convert coordinate indices to 0-based format and automatically detect whether the input is 0-based or 1-based."""
    coords = coords.astype(np.int64)
    Dx, Dy, Dz = shape
    zero_ok = (
        (coords.min() >= 0)
        and np.all(coords[:, 0] < Dx)
        and np.all(coords[:, 1] < Dy)
        and np.all(coords[:, 2] < Dz)
    )
    one_ok = (
        (coords.min() >= 1)
        and np.all(coords[:, 0] <= Dx)
        and np.all(coords[:, 1] <= Dy)
        and np.all(coords[:, 2] <= Dz)
    )

    if zero_ok and not one_ok:
        return coords, "0-based"
    if one_ok and not zero_ok:
        return coords - 1, "1-based→0-based (auto adjusted)"
    if zero_ok and one_ok:
        return coords, "ambiguous (treated as 0-based)"

    raise ValueError("Coordinates out of range for both 0-based and 1-based indexing.")


def flat_from_coords(coords, shape):
    """Convert 3D coordinate indices into 1D linear indices under C-order."""
    Dx, Dy, Dz = shape
    return coords[:, 0] * Dy * Dz + coords[:, 1] * Dz + coords[:, 2]


def ensure_zero_based_linear_idx(idx, total_size):
    """Convert linear indices to 0-based format and automatically detect whether the input is 0-based or 1-based."""
    idx = idx.astype(np.int64)
    zero_ok = (idx.min() >= 0) and (idx.max() < total_size)
    one_ok = (idx.min() >= 1) and (idx.max() <= total_size)

    if zero_ok and not one_ok:
        return idx, "0-based"
    if one_ok and not zero_ok:
        return idx - 1, "1-based→0-based (auto adjusted)"
    if zero_ok and one_ok:
        return idx, "ambiguous (treated as 0-based)"

    raise ValueError("Linear indices out of range for both 0-based and 1-based.")


def build_perm_from_to(src_linear, dst_linear):
    """Build a permutation perm such that dst_linear[perm] == src_linear."""
    if src_linear.shape != dst_linear.shape:
        raise ValueError("src and dst must have same shape to build a permutation.")

    pos = {v: i for i, v in enumerate(dst_linear.tolist())}
    perm = np.empty_like(src_linear)
    missing = []

    for i, v in enumerate(src_linear.tolist()):
        j = pos.get(v, None)
        if j is None:
            missing.append(v)
            j = -1
        perm[i] = j

    return perm, missing


def main():
    """Run consistency checks on voxel count, voxel set, and voxel order."""
    ap = argparse.ArgumentParser(description="Check voxel_indices.npy count & order against nsdgeneral.nii.gz mask(view-C).")
    ap.add_argument("--mask", required=True, help="Path to nsdgeneral.nii.gz")
    ap.add_argument("--voxel-indices", required=True, help="Path to voxel_indices.npy (Nx3 coords or 1D linear indices)")
    ap.add_argument("--label", type=int, default=1, help="Mask label value to include (default: 1)")
    ap.add_argument("--save-report", default=None, help="Optional CSV to save mismatch samples")
    ap.add_argument("--max-show", type=int, default=20, help="Max samples to print for mismatches")
    args = ap.parse_args()

    img = nib.load(args.mask)
    data = img.get_fdata()
    if data.ndim != 3:
        raise ValueError(f"Mask image must be 3D, got shape={data.shape}")

    Dx, Dy, Dz = data.shape
    mask_bool = as_bool_mask_from_label(data, label_value=args.label)
    flat_mask = flat_indices_from_mask(mask_bool)
    N_mask = flat_mask.size

    vxi = np.load(args.voxel_indices, allow_pickle=False)
    if vxi.ndim == 2 and vxi.shape[1] == 3:
        coords0, msg = ensure_zero_based_coords(vxi, (Dx, Dy, Dz))
        flat_vxi = flat_from_coords(coords0, (Dx, Dy, Dz))
        coord_mode = f"coords Nx3 ({msg})"
    elif vxi.ndim == 1:
        flat_vxi, msg = ensure_zero_based_linear_idx(vxi, Dx * Dy * Dz)
        coord_mode = f"linear N ({msg})"
    else:
        raise ValueError(f"Unsupported voxel_indices.npy shape: {vxi.shape}")

    N_vxi = flat_vxi.size

    print("=== Basic Info ===")
    print(f"Mask path         : {args.mask}")
    print(f"Voxel indices path: {args.voxel_indices}  [{coord_mode}]")
    print(f"Volume shape      : (Dx,Dy,Dz)=({Dx},{Dy},{Dz})  total={Dx*Dy*Dz}")
    print(f"Mask count(label={args.label}): {N_mask}")
    print(f"voxel_indices count           : {N_vxi}")

    counts_equal = (N_mask == N_vxi)
    print(f"\n[Count] Equal? {counts_equal}")
    if not counts_equal:
        print(f"  -> Difference: mask {N_mask} vs voxel_indices {N_vxi}")

    set_mask = set(flat_mask.tolist())
    set_vxi = set(flat_vxi.tolist())
    only_in_mask = set_mask - set_vxi
    only_in_vxi = set_vxi - set_mask
    sets_equal = (len(only_in_mask) == 0 and len(only_in_vxi) == 0)

    print(f"\n[Set] Equal (ignoring order)? {sets_equal}")
    if not sets_equal:
        print(f"  -> in mask not in vxi : {len(only_in_mask)}")
        print(f"  -> in vxi  not in mask: {len(only_in_vxi)}")

    N_cmp = min(N_mask, N_vxi)
    order_equal = np.array_equal(flat_mask[:N_cmp], flat_vxi[:N_cmp]) and counts_equal
    mismatch_positions = np.where(flat_mask[:N_cmp] != flat_vxi[:N_cmp])[0]
    n_mismatch = int(mismatch_positions.size)

    print(f"\n[Order] Equal (element-wise & count-equal)? {order_equal}")
    print(f"  -> mismatches: {n_mismatch} / {N_cmp}")

    if n_mismatch > 0:
        show = min(args.max_show, n_mismatch)
        print(f"\n[Samples of mismatches] (showing {show})")
        for i in mismatch_positions[:show]:
            print(f"  idx {i:>7d} : mask={flat_mask[i]:>9d} | vxi={flat_vxi[i]:>9d}")

    perm = None
    missing = []
    if sets_equal:
        perm, missing = build_perm_from_to(src_linear=flat_vxi, dst_linear=flat_mask)
        neg = int((perm < 0).sum())
        out_of_place = int((perm != np.arange(perm.size)).sum())

        print(f"\n[Permutation vxi→mask]")
        print(f"  -> missing (should be 0 if sets_equal=True): {neg}")
        print(f"  -> out-of-place positions: {out_of_place} / {perm.size}")

        if args.save_report:
            bad = np.where(perm != np.arange(perm.size))[0]
            k = min(args.max_show, bad.size)
            rows = []
            for i in bad[:k]:
                rows.append((i, int(flat_vxi[i]), int(flat_mask[perm[i]])))

            import csv
            with open(args.save_report, "w", newline="") as f:
                cw = csv.writer(f)
                cw.writerow(["idx_in_vxi", "linear_from_vxi", "linear_at_same_pos_in_mask_perm"])
                cw.writerows(rows)

            print(f"  -> mismatch samples saved to: {args.save_report}")

    dup_vxi = [k for k, v in Counter(flat_vxi.tolist()).items() if v > 1]
    dup_mask = [k for k, v in Counter(flat_mask.tolist()).items() if v > 1]

    if dup_vxi or dup_mask:
        print("\n[Duplicate Warning]")
        if dup_vxi:
            print(f"  -> voxel_indices has duplicates: {len(dup_vxi)}")
        if dup_mask:
            print(f"  -> mask(flat) has duplicates: {len(dup_mask)}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()